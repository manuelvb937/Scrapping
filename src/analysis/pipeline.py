"""Analysis pipeline for social-listening topic and sentiment outputs.

The current flow is:
1. Preprocess posts for clustering.
2. Run transformer sentiment per post as ``Sentiment_Transformer``.
3. Embed posts.
4. Cluster with UMAP + HDBSCAN.
5. Extract c-TF-IDF keywords per cluster.
6. Send posts to the LLM in cluster-local batches for ``Sentiment_LLM`` and
   raw post topics.
7. Consolidate raw topic labels algorithmically inside each cluster.
8. Refine each cluster taxonomy with a second LLM consolidation step.
9. Attach canonical topics back to posts.
10. Make one final LLM marketing call over cluster summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clustering import Cluster, cluster_embeddings
from .embeddings import TextEmbedder
from .llm_labeling import LLMTopicSentimentLabeler
from .post_llm_annotation import MarketingSummaryGenerator, PostBatchAnnotator, TopicConsolidationRefiner
from .sentiment_transformer import (
    TransformerSentimentAnalyzer,
    contains_ambiguity_marker,
    normalize_sentiment_label,
)
from .structured_output import build_structured_output
from .text_preparation import (
    clean_for_llm,
    extract_cluster_keywords_ctfidf,
    extract_search_term_fragments,
    extract_top_keywords,
    prepare_clustering_texts,
    select_representative_posts,
)
from .topic_consolidation import consolidate_cluster_topics
from social_listening_pipeline.config import load_settings

LOGGER = logging.getLogger(__name__)
SENTIMENT_VALUES = ("positive", "neutral", "negative")


FREE_MODE_DEFAULTS = {
    "SENTIMENT_METHOD": "transformer",
    "ENABLE_LLM_SENTIMENT_FALLBACK": "false",
    "SENTIMENT_CONFIDENCE_THRESHOLD": "0.55",
    "HDBSCAN_MIN_CLUSTER_SIZE": "6",
    "HDBSCAN_MIN_SAMPLES": "2",
    "HDBSCAN_CLUSTER_SELECTION_METHOD": "eom",
    "POST_TOPIC_LLM_BATCH_SIZE": "15",
    "LLM_TOPIC_CONSOLIDATION_MAX_TOPICS": "6",
    "LLM_TOPIC_CONSOLIDATION_MAX_INPUT_TOPICS": "40",
    "LLM_POST_BATCH_SIZE": "15",
    "LLM_POST_TOPIC_MAX_OUTPUT_TOKENS": "4096",
    "LLM_MARKETING_MAX_OUTPUT_TOKENS": "4096",
    "LLM_MARKETING_MAX_CLUSTERS": "60",
    "TOPIC_LABELING_DELAY_SECONDS": "0",
    "TOPIC_LABELING_429_COOLDOWN_SECONDS": "120",
    "GEMINI_FREE_TIER_LIMITING": "true",
    "GEMINI_REQUESTS_PER_MINUTE": "10",
    "GEMINI_TOKENS_PER_MINUTE": "250000",
    "GEMINI_REQUESTS_PER_DAY": "250",
    "GEMINI_MIN_SECONDS_BETWEEN_REQUESTS": "6",
    "GEMINI_THINKING_BUDGET": "0",
    "LLM_TEMPERATURE": "0",
    "LLM_TOP_P": "1",
}


def apply_free_mode_defaults() -> None:
    """Tune environment defaults for Gemini/OpenAI free-tier-style runs."""
    for name, value in FREE_MODE_DEFAULTS.items():
        os.environ.setdefault(name, value)


def load_processed_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sentiment_counts_to_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {
        sentiment: round(counts.get(sentiment, 0) / total, 4)
        for sentiment in SENTIMENT_VALUES
    }


def _dominant_sentiment(counts: dict[str, int]) -> str:
    tie_break = {"positive": 1, "neutral": 2, "negative": 0}
    return max(SENTIMENT_VALUES, key=lambda item: (counts.get(item, 0), tie_break[item]))


def _ordered_sentiment_counts(counter: Counter) -> dict[str, int]:
    return {sentiment: int(counter.get(sentiment, 0)) for sentiment in SENTIMENT_VALUES}


def _sentiment_counts_for_indices(indices: list[int], sentiment_by_index: dict[int, str]) -> dict[str, int]:
    counter = Counter(normalize_sentiment_label(sentiment_by_index.get(index)) for index in indices)
    return _ordered_sentiment_counts(counter)


def _match_representative_indices(
    cluster_indices: list[int],
    cluster_texts: list[str],
    representative_texts: list[str],
) -> set[int]:
    matched: set[int] = set()
    for representative in representative_texts:
        for original_index, text in zip(cluster_indices, cluster_texts):
            if original_index in matched:
                continue
            if text[:300] == representative:
                matched.add(original_index)
                break
    return matched


def _cluster_clustering_texts(
    cluster: Cluster,
    *,
    clustered_original_indices: set[int],
    valid_index_positions: dict[int, int],
    clustering_texts: list[str],
    all_texts: list[str],
) -> list[str]:
    texts: list[str] = []
    for index in cluster.indices:
        if index in clustered_original_indices and index in valid_index_positions:
            texts.append(clustering_texts[valid_index_positions[index]])
        else:
            texts.append(all_texts[index])
    return texts


def _analyze_transformer_sentiments(texts: list[str], settings: Any) -> list[dict[str, Any]]:
    """Run transformer sentiment, falling back only when the model cannot load."""
    method = str(getattr(settings, "sentiment_method", "transformer")).strip().lower()
    if method in {"llm", "heuristic", "legacy"}:
        LOGGER.info("Using heuristic sentiment fallback because SENTIMENT_METHOD=%s", method)
        return _fallback_sentiment_results(
            texts,
            settings.sentiment_model_name,
            f"SENTIMENT_METHOD={method}",
        )

    transformer = TransformerSentimentAnalyzer(model_name=settings.sentiment_model_name)
    try:
        transformer_results = transformer.analyze(texts)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Transformer sentiment failed; using heuristic fallback: %s", exc)
        return _fallback_sentiment_results(texts, settings.sentiment_model_name, str(exc))

    return [
        {
            "text": item["text"],
            "transformer_sentiment": {
                "label": normalize_sentiment_label(item.get("label")),
                "confidence": item.get("confidence", 0.0),
                "raw_scores": item.get("raw_scores", {}),
                "model": item.get("model", settings.sentiment_model_name),
            },
            "llm_review": None,
            "final_sentiment": normalize_sentiment_label(item.get("label")),
            "sentiment_source": "transformer",
            "ambiguity_triggered": contains_ambiguity_marker(str(item.get("text", ""))),
        }
        for item in transformer_results
    ]


def _fallback_sentiment_results(texts: list[str], model_name: str, reason: str) -> list[dict[str, Any]]:
    labeler = LLMTopicSentimentLabeler()
    results = []
    for text in texts:
        sentiment = normalize_sentiment_label(labeler._heuristic_sentiment(text))
        results.append(
            {
                "text": text,
                "transformer_sentiment": {
                    "label": sentiment,
                    "confidence": 0.0,
                    "raw_scores": {
                        "positive": 0.0,
                        "neutral": 0.0,
                        "negative": 0.0,
                    },
                    "model": model_name,
                    "fallback_reason": reason,
                },
                "llm_review": None,
                "final_sentiment": sentiment,
                "sentiment_source": "heuristic_fallback",
                "ambiguity_triggered": contains_ambiguity_marker(text),
            }
        )
    return results


def _primary_topic(topics: list[dict[str, Any]], fallback_keywords: list[tuple[str, int | float]]) -> dict[str, Any]:
    if topics:
        return topics[0]
    fallback_label = str(fallback_keywords[0][0]) if fallback_keywords else "general conversation"
    return {
        "topic_id": None,
        "topic_label": fallback_label,
        "topic_description": f"Conversation around {fallback_label}",
    }


def _marketing_result_for_cluster(marketing_summary: dict[str, Any], cluster_id: int) -> dict[str, Any]:
    for item in marketing_summary.get("clusters", []):
        if int(item.get("cluster_id", -999)) == cluster_id:
            return item
    return {
        "marketing_interpretation": "",
        "risk_level": "low",
        "opportunity_level": "low",
        "recommended_actions": [],
    }


def _apply_refined_topic_maps(
    post_topics_by_index: dict[int, list[dict[str, Any]]],
    *,
    post_cluster_lookup: dict[int, int],
    topic_id_maps_by_cluster: dict[int, dict[str, dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    refined: dict[int, list[dict[str, Any]]] = {}
    for post_index, topics in post_topics_by_index.items():
        cluster_id = post_cluster_lookup.get(post_index)
        topic_map = topic_id_maps_by_cluster.get(cluster_id, {})
        updated_topics: list[dict[str, Any]] = []
        for topic in topics:
            old_topic_id = str(topic.get("topic_id") or "")
            mapped = topic_map.get(old_topic_id)
            if mapped:
                next_topic = {
                    "topic_id": mapped.get("topic_id"),
                    "topic_label": mapped.get("topic_label"),
                    "topic_description": mapped.get("topic_description"),
                    "raw_topic_label": topic.get("raw_topic_label") or topic.get("topic_label"),
                    "preliminary_topic_id": old_topic_id,
                }
            else:
                next_topic = dict(topic)
            if not any(existing.get("topic_id") == next_topic.get("topic_id") for existing in updated_topics):
                updated_topics.append(next_topic)
            if len(updated_topics) >= 3:
                break
        refined[post_index] = updated_topics
    return refined


def run_analysis(input_path: str | Path, output_dir: str | Path = "data/reports") -> tuple[Path, Path]:
    settings = load_settings()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    posts = load_processed_jsonl(input_path)

    # --- 1. Preprocess posts for clustering ---
    LOGGER.info("Preparing clustering texts (stripping titles and noise)...")
    clustering_texts, valid_indices = prepare_clustering_texts(posts)
    search_term_stopwords = extract_search_term_fragments(posts)
    all_texts = [str(post.get("cleaned_content") or post.get("content") or "") for post in posts]
    llm_texts = [clean_for_llm(text) for text in all_texts]

    # --- 2. Transformer sentiment per post ---
    LOGGER.info("Running transformer sentiment on %d posts...", len(llm_texts))
    post_sentiment_results = _analyze_transformer_sentiments(llm_texts, settings)
    sentiment_by_index: dict[int, str] = {}
    sentiment_result_by_index: dict[int, dict[str, Any]] = {}
    for index, result in enumerate(post_sentiment_results):
        sentiment = normalize_sentiment_label(result.get("final_sentiment"))
        sentiment_by_index[index] = sentiment
        sentiment_result_by_index[index] = result
        if index < len(posts):
            posts[index]["Sentiment_Transformer"] = sentiment
            posts[index]["sentiment_transformer_result"] = result.get("transformer_sentiment")

    # --- 3. Embeddings ---
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(clustering_texts)

    # --- 4. UMAP + HDBSCAN clustering ---
    clusters, diagnostics = cluster_embeddings(embeddings)
    coordinates_by_index: dict[int, tuple[float, float]] = {}
    for embedding_index, coordinate in enumerate(diagnostics.umap_coordinates):
        if embedding_index >= len(valid_indices) or len(coordinate) < 2:
            continue
        coordinates_by_index[valid_indices[embedding_index]] = (coordinate[0], coordinate[1])

    remapped_clusters: list[Cluster] = []
    clustered_original_indices: set[int] = set()
    for cluster in clusters:
        original_indices = [valid_indices[i] for i in cluster.indices]
        remapped_clusters.append(Cluster(cluster_id=cluster.cluster_id, indices=original_indices))
        clustered_original_indices.update(original_indices)

    skipped_indices = [index for index in range(len(posts)) if index not in clustered_original_indices]
    if skipped_indices:
        unclustered_id = max((cluster.cluster_id for cluster in remapped_clusters), default=-1) + 1
        remapped_clusters.append(Cluster(cluster_id=unclustered_id, indices=skipped_indices))
        LOGGER.info(
            "%d posts were too short after title stripping; grouped as cluster %d",
            len(skipped_indices),
            unclustered_id,
        )

    valid_index_positions = {idx: pos for pos, idx in enumerate(valid_indices)}
    cluster_clustering_texts_by_id = {
        cluster.cluster_id: _cluster_clustering_texts(
            cluster,
            clustered_original_indices=clustered_original_indices,
            valid_index_positions=valid_index_positions,
            clustering_texts=clustering_texts,
            all_texts=all_texts,
        )
        for cluster in remapped_clusters
    }

    # --- 5. c-TF-IDF keywords per cluster ---
    ctfidf_keywords_by_cluster = extract_cluster_keywords_ctfidf(
        cluster_clustering_texts_by_id,
        top_n=10,
        search_terms=search_term_stopwords,
    )
    cluster_contexts: dict[int, dict[str, Any]] = {}
    representative_indices_by_cluster: dict[int, set[int]] = {}
    for cluster in remapped_clusters:
        cluster_texts = cluster_clustering_texts_by_id.get(cluster.cluster_id, [])
        top_keywords = ctfidf_keywords_by_cluster.get(cluster.cluster_id)
        if not top_keywords:
            top_keywords = extract_top_keywords(cluster_texts, top_n=10, search_terms=search_term_stopwords)
        representative = select_representative_posts(cluster_texts, limit=5)
        representative_indices = _match_representative_indices(cluster.indices, cluster_texts, representative)
        representative_indices_by_cluster[cluster.cluster_id] = representative_indices
        cluster_contexts[cluster.cluster_id] = {
            "top_keywords": top_keywords,
            "representative_posts": representative,
            "representative_indices": representative_indices,
        }

    # --- 6. LLM batches inside each cluster: Sentiment_LLM + raw topics ---
    annotator = PostBatchAnnotator(model=settings.topic_labeling_model)
    post_annotations_by_index: dict[int, dict[str, Any]] = {}
    for cluster in remapped_clusters:
        context = cluster_contexts.get(cluster.cluster_id, {})
        batch_posts = [
            {
                "source_row_index": index,
                "text": llm_texts[index] or all_texts[index],
            }
            for index in cluster.indices
        ]
        post_annotations_by_index.update(
            annotator.annotate_cluster_posts(
                cluster_id=cluster.cluster_id,
                posts=batch_posts,
                top_keywords=context.get("top_keywords"),
                search_terms=search_term_stopwords,
                fallback_sentiments_by_index=sentiment_by_index,
            )
        )

    for index, post in enumerate(posts):
        annotation = post_annotations_by_index.get(index) or {
            "Sentiment_LLM": sentiment_by_index.get(index, "neutral"),
            "sentiment_llm": sentiment_by_index.get(index, "neutral"),
            "raw_topics_llm": [],
            "used_llm": False,
        }
        post["Sentiment_LLM"] = normalize_sentiment_label(annotation.get("Sentiment_LLM"))
        post["raw_topics_llm"] = annotation.get("raw_topics_llm") or []
        post["llm_topic_annotation_used"] = bool(annotation.get("used_llm"))
        post["post_sentiment"] = post["Sentiment_LLM"]
        post["post_sentiment_result"] = sentiment_result_by_index.get(index, {})

    # --- 7-8. Consolidate topics inside each cluster and attach to posts ---
    cluster_topics_by_id, post_topics_by_index = consolidate_cluster_topics(
        clusters=remapped_clusters,
        post_annotations_by_index=post_annotations_by_index,
        post_texts=all_texts,
        embedder=embedder,
    )
    topic_refiner = TopicConsolidationRefiner(model=settings.topic_labeling_model)
    topic_id_maps_by_cluster: dict[int, dict[str, dict[str, Any]]] = {}
    refined_cluster_topics_by_id: dict[int, list[dict[str, Any]]] = {}
    for cluster in remapped_clusters:
        context = cluster_contexts.get(cluster.cluster_id, {})
        refined_topics, topic_id_map = topic_refiner.refine_cluster_topics(
            cluster_id=cluster.cluster_id,
            preliminary_topics=cluster_topics_by_id.get(cluster.cluster_id, []),
            top_keywords=context.get("top_keywords"),
            representative_texts=context.get("representative_posts", []),
        )
        refined_cluster_topics_by_id[cluster.cluster_id] = refined_topics
        topic_id_maps_by_cluster[cluster.cluster_id] = topic_id_map

    cluster_topics_by_id = refined_cluster_topics_by_id
    post_cluster_lookup = {
        index: cluster.cluster_id
        for cluster in remapped_clusters
        for index in cluster.indices
    }
    post_topics_by_index = _apply_refined_topic_maps(
        post_topics_by_index,
        post_cluster_lookup=post_cluster_lookup,
        topic_id_maps_by_cluster=topic_id_maps_by_cluster,
    )
    for index, post in enumerate(posts):
        canonical_topics = post_topics_by_index.get(index, [])
        post["topics"] = canonical_topics
        if canonical_topics:
            post["topic_id"] = canonical_topics[0]["topic_id"]
            post["topic_label"] = canonical_topics[0]["topic_label"]
            post["topic_description"] = canonical_topics[0]["topic_description"]

    # --- 9. Final LLM marketing call ---
    cluster_summaries_for_marketing: list[dict[str, Any]] = []
    for cluster in remapped_clusters:
        context = cluster_contexts.get(cluster.cluster_id, {})
        llm_sentiment_by_index = {
            index: normalize_sentiment_label(posts[index].get("Sentiment_LLM"))
            for index in cluster.indices
        }
        cluster_summaries_for_marketing.append(
            {
                "cluster_id": cluster.cluster_id,
                "cluster_size": len(cluster.indices),
                "top_keywords": [str(keyword) for keyword, _ in context.get("top_keywords", [])],
                "sentiment_llm_counts": _sentiment_counts_for_indices(cluster.indices, llm_sentiment_by_index),
                "sentiment_transformer_counts": _sentiment_counts_for_indices(cluster.indices, sentiment_by_index),
                "topics": cluster_topics_by_id.get(cluster.cluster_id, []),
                "representative_texts": context.get("representative_posts", []),
            }
        )
    marketing_summary = MarketingSummaryGenerator(model=settings.topic_labeling_model).generate(
        cluster_summaries_for_marketing
    )

    clusters_payload: list[dict[str, Any]] = []
    analysis_clusters: list[dict[str, Any]] = []
    for cluster in remapped_clusters:
        context = cluster_contexts.get(cluster.cluster_id, {})
        top_keywords = context.get("top_keywords", [])
        canonical_topics = cluster_topics_by_id.get(cluster.cluster_id, [])
        primary_topic = _primary_topic(canonical_topics, top_keywords)
        llm_sentiment_by_index = {
            index: normalize_sentiment_label(posts[index].get("Sentiment_LLM"))
            for index in cluster.indices
        }
        sentiment_counts = _sentiment_counts_for_indices(cluster.indices, llm_sentiment_by_index)
        sentiment_dist = _sentiment_counts_to_distribution(sentiment_counts)
        dominant_sentiment = _dominant_sentiment(sentiment_counts)
        transformer_counts = _sentiment_counts_for_indices(cluster.indices, sentiment_by_index)
        marketing = _marketing_result_for_cluster(marketing_summary, cluster.cluster_id)

        cluster_record = {
            "cluster_id": cluster.cluster_id,
            "size": len(cluster.indices),
            "post_indices": cluster.indices,
            "posts": [posts[index] for index in cluster.indices],
            "sentiment_counts": sentiment_counts,
            "sentiment_distribution": sentiment_dist,
            "dominant_sentiment": dominant_sentiment,
            "sentiment_llm_counts": sentiment_counts,
            "sentiment_llm_distribution": sentiment_dist,
            "sentiment_transformer_counts": transformer_counts,
            "topic_representation_method": "llm_batch_topics_consolidated_with_ctfidf_keywords",
            "top_keywords": [
                {"keyword": keyword, "count": score, "score": score}
                for keyword, score in top_keywords
            ],
            "topics": canonical_topics,
            "topic_id": primary_topic.get("topic_id"),
            "topic_label": primary_topic.get("topic_label"),
            "topic_description": primary_topic.get("topic_description"),
            "representative_posts": context.get("representative_posts", []),
            "representative_post_indices": sorted(context.get("representative_indices", set())),
            "marketing_interpretation": marketing.get("marketing_interpretation"),
            "risk_level": marketing.get("risk_level"),
            "opportunity_level": marketing.get("opportunity_level"),
            "recommended_actions": marketing.get("recommended_actions", []),
            "umap_coordinates": [
                {
                    "post_index": index,
                    "x": coordinates_by_index[index][0],
                    "y": coordinates_by_index[index][1],
                }
                for index in cluster.indices
                if index in coordinates_by_index
            ],
        }
        clusters_payload.append(cluster_record)
        analysis_clusters.append(
            {
                "cluster_id": cluster.cluster_id,
                "topic_id": primary_topic.get("topic_id"),
                "topic_label": primary_topic.get("topic_label"),
                "topic_summary": primary_topic.get("topic_description"),
                "topic_description": primary_topic.get("topic_description"),
                "topics": canonical_topics,
                "rationale": "Post-level LLM topics were consolidated automatically inside the cluster.",
                "marketing_interpretation": marketing.get("marketing_interpretation"),
                "risk_level": marketing.get("risk_level"),
                "opportunity_level": marketing.get("opportunity_level"),
                "recommended_actions": marketing.get("recommended_actions", []),
                "sentiment": dominant_sentiment,
                "dominant_sentiment": dominant_sentiment,
                "size": len(cluster.indices),
                "sentiment_counts": sentiment_counts,
                "sentiment_distribution": sentiment_dist,
                "sentiment_llm_counts": sentiment_counts,
                "sentiment_llm_distribution": sentiment_dist,
                "sentiment_transformer_counts": transformer_counts,
                "topic_representation_method": "llm_batch_topics_consolidated_with_ctfidf_keywords",
                "top_keywords": [
                    {"keyword": keyword, "count": score, "score": score}
                    for keyword, score in top_keywords
                ],
                "representative_posts": context.get("representative_posts", []),
            }
        )

    analysis_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "total_posts": len(posts),
        "structured_output_file": str(output_dir / "structured_output.json"),
        "diagnostics": {
            "cluster_count": diagnostics.cluster_count,
            "cluster_sizes": diagnostics.cluster_sizes,
            "largest_cluster_ratio": round(diagnostics.largest_cluster_ratio, 3),
            "noise_count": diagnostics.noise_count,
            "clustered_posts": len(valid_indices),
            "skipped_posts": len(skipped_indices) if skipped_indices else 0,
            "coordinate_method": diagnostics.coordinate_method,
            "warning": diagnostics.warning,
        },
        "pipeline_steps": [
            "preprocess_posts",
            "transformer_sentiment_per_post",
            "embeddings",
            "umap_hdbscan_clustering",
            "ctfidf_keywords_per_cluster",
            "llm_post_batches_for_sentiment_llm_and_raw_topics",
            "automatic_topic_consolidation_per_cluster",
            "llm_topic_taxonomy_refinement_per_cluster",
            "attach_canonical_topics_to_posts",
            "final_llm_marketing_summary",
        ],
        "sentiment_method": "transformer_plus_llm_post_batches",
        "sentiment_transformer_model": settings.sentiment_model_name,
        "sentiment_model": settings.sentiment_model_name,
        "llm_provider": annotator.provider,
        "llm_model": annotator.model,
        "llm_post_batch_size": annotator.batch_size,
        "llm_has_api_key": annotator.has_api_key,
        "topic_consolidation": {
            "method": "normalized_label_plus_embedding_similarity_then_llm_taxonomy_refinement",
            "scope": "inside_each_cluster",
            "llm_max_topics_per_cluster": topic_refiner.max_topics,
            "llm_max_input_topics_per_cluster": topic_refiner.max_input_topics,
            "llm_refinement_has_api_key": topic_refiner.has_api_key,
        },
        "marketing_report": marketing_summary,
        "clusters": analysis_clusters,
    }

    clusters_path = output_dir / "clusters.json"
    analysis_path = output_dir / "analysis.json"
    structured_output_path = output_dir / "structured_output.json"

    with clusters_path.open("w", encoding="utf-8") as file:
        json.dump({"clusters": clusters_payload}, file, ensure_ascii=False, indent=2)

    with analysis_path.open("w", encoding="utf-8") as file:
        json.dump(analysis_payload, file, ensure_ascii=False, indent=2)

    structured_output = build_structured_output(
        posts=posts,
        input_path=input_path,
        clusters_payload=clusters_payload,
        analysis_payload=analysis_payload,
        sentiment_results_by_index=sentiment_result_by_index,
        coordinates_by_index=coordinates_by_index,
        representative_indices_by_cluster=representative_indices_by_cluster,
    )
    with structured_output_path.open("w", encoding="utf-8") as file:
        json.dump(structured_output, file, ensure_ascii=False, indent=2)

    return clusters_path, analysis_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis pipeline on processed JSONL posts.")
    parser.add_argument("--input", required=True, help="Input processed JSONL path.")
    parser.add_argument("--output-dir", default="data/reports", help="Output directory for JSON reports.")
    parser.add_argument(
        "--free",
        action="store_true",
        help="Use conservative LLM batching/delays for free-tier API limits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.free:
        apply_free_mode_defaults()
    clusters_path, analysis_path = run_analysis(args.input, args.output_dir)
    print(f"Saved clusters to: {clusters_path}")
    print(f"Saved analysis to: {analysis_path}")
    print(f"Saved structured output to: {Path(args.output_dir) / 'structured_output.json'}")


if __name__ == "__main__":
    main()
