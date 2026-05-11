"""Analysis pipeline: embeddings, clustering, topic labeling, and sentiment.

Updated to use:
  - TextEmbedder (BGE-M3) for semantic embeddings (#1)
  - UMAP + HDBSCAN for clustering (#2)
  - Transformer-first per-post sentiment analysis (#4)
  - LLM topic labeling per cluster
  - Text preparation for clustering (title/noise stripping)
  - Diagnostics, representative posts, and top keywords per cluster
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

from .clustering import cluster_embeddings, Cluster
from .embeddings import TextEmbedder
from .llm_labeling import LLMTopicSentimentLabeler
from .sentiment_transformer import (
    HybridSentimentAnalyzer,
    LLMSentimentReviewer,
    TransformerSentimentAnalyzer,
    contains_ambiguity_marker,
    normalize_sentiment_label,
)
from .text_preparation import (
    prepare_clustering_texts,
    extract_top_keywords,
    extract_search_term_fragments,
    select_representative_posts,
    clean_for_llm,
)
from .topic_labeling import TopicLabeler
from social_listening_pipeline.config import load_settings

LOGGER = logging.getLogger(__name__)
SENTIMENT_VALUES = ("positive", "neutral", "negative")


def load_processed_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _sentiment_counts_to_distribution(counts: dict[str, int]) -> dict[str, float]:
    """Convert sentiment counts into stable positive/neutral/negative ratios."""
    total = sum(counts.values()) or 1
    return {
        sentiment: round(counts.get(sentiment, 0) / total, 4)
        for sentiment in SENTIMENT_VALUES
    }


def _dominant_sentiment(counts: dict[str, int]) -> str:
    """Pick the most common sentiment, preferring neutral for exact ties."""
    tie_break = {"positive": 1, "neutral": 2, "negative": 0}
    return max(SENTIMENT_VALUES, key=lambda item: (counts.get(item, 0), tie_break[item]))


def _analyze_post_sentiments(texts: list[str], settings: Any) -> list[dict[str, Any]]:
    """Analyze post sentiment according to SENTIMENT_METHOD.

    Supported modes:
    - transformer: Hugging Face model only
    - llm: legacy LLM batch sentiment behavior
    - hybrid: transformer first, LLM review for low confidence or ambiguous slang
    """
    method = str(settings.sentiment_method).strip().lower()
    if method not in {"llm", "transformer", "hybrid"}:
        raise ValueError("SENTIMENT_METHOD must be one of: llm, transformer, hybrid")

    if method == "llm":
        labeler = LLMTopicSentimentLabeler()
        legacy_results = labeler.analyze_posts_batch(texts)
        sentiment_by_index = {
            item.index: normalize_sentiment_label(item.sentiment)
            for item in legacy_results
        }
        return [
            {
                "text": text,
                "transformer_sentiment": None,
                "llm_review": {
                    "label": sentiment_by_index.get(index, "neutral"),
                    "rationale": "Legacy LLM sentiment mode.",
                },
                "final_sentiment": sentiment_by_index.get(index, "neutral"),
                "sentiment_source": "llm",
                "ambiguity_triggered": contains_ambiguity_marker(text),
            }
            for index, text in enumerate(texts)
        ]

    transformer = TransformerSentimentAnalyzer(model_name=settings.sentiment_model_name)

    if method == "transformer":
        transformer_results = transformer.analyze(texts)
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

    hybrid = HybridSentimentAnalyzer(
        transformer=transformer,
        llm_reviewer=LLMSentimentReviewer(),
        confidence_threshold=settings.sentiment_confidence_threshold,
        enable_llm_fallback=settings.enable_llm_sentiment_fallback,
    )
    return hybrid.analyze(texts)


def run_analysis(input_path: str | Path, output_dir: str | Path = "data/reports") -> tuple[Path, Path]:
    settings = load_settings()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    posts = load_processed_jsonl(input_path)

    # --- Text preparation ---
    # Create clustering_text by stripping search terms, usernames, timestamps
    # The original cleaned_content is preserved for display and sentiment analysis
    LOGGER.info("Preparing clustering texts (stripping titles and noise)...")
    clustering_texts, valid_indices = prepare_clustering_texts(posts)

    # Extract search-term fragments for dynamic stopword filtering
    search_term_stopwords = extract_search_term_fragments(posts)
    LOGGER.info(
        "Search-term stopwords extracted: %s",
        search_term_stopwords,
    )

    # Full texts for sentiment analysis (use cleaned_content, not clustering_text)
    all_texts = [str(post.get("cleaned_content") or post.get("content") or "") for post in posts]

    # --- Embeddings ---
    # Embed clustering_text (title-stripped) instead of cleaned_content
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(clustering_texts)

    # --- Clustering ---
    clusters, diagnostics = cluster_embeddings(embeddings)

    # Map cluster indices back to original post indices
    # (clustering was done on valid_indices subset, not all posts)
    remapped_clusters: list[Cluster] = []
    clustered_original_indices: set[int] = set()
    for cluster in clusters:
        original_indices = [valid_indices[i] for i in cluster.indices]
        remapped_clusters.append(Cluster(cluster_id=cluster.cluster_id, indices=original_indices))
        clustered_original_indices.update(original_indices)

    # Add unclustered posts (those skipped during text preparation) as a separate cluster
    skipped_indices = [i for i in range(len(posts)) if i not in clustered_original_indices]
    if skipped_indices:
        unclustered_id = max(c.cluster_id for c in remapped_clusters) + 1 if remapped_clusters else 0
        remapped_clusters.append(Cluster(cluster_id=unclustered_id, indices=skipped_indices))
        LOGGER.info(
            "%d posts were too short after title stripping — grouped as cluster %d",
            len(skipped_indices), unclustered_id,
        )

    # --- Per-post sentiment analysis ---
    # Clean texts for LLM input (strip URLs, usernames, timestamps to save tokens)
    llm_texts = [clean_for_llm(t) for t in all_texts]
    LOGGER.info(
        "Running per-post sentiment analysis on %d posts with method=%s...",
        len(llm_texts),
        settings.sentiment_method,
    )
    post_sentiment_results = _analyze_post_sentiments(llm_texts, settings)

    # Build sentiment lookup (index -> sentiment)
    sentiment_by_index: dict[int, str] = {}
    sentiment_result_by_index: dict[int, dict[str, Any]] = {}
    for index, result in enumerate(post_sentiment_results):
        sentiment = normalize_sentiment_label(result.get("final_sentiment"))
        sentiment_by_index[index] = sentiment
        sentiment_result_by_index[index] = result

    topic_labeler = TopicLabeler(model=settings.topic_labeling_model)

    # --- Build output ---
    clusters_payload: list[dict] = []
    analysis_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "total_posts": len(posts),
        "diagnostics": {
            "cluster_count": diagnostics.cluster_count,
            "cluster_sizes": diagnostics.cluster_sizes,
            "largest_cluster_ratio": round(diagnostics.largest_cluster_ratio, 3),
            "noise_count": diagnostics.noise_count,
            "clustered_posts": len(valid_indices),
            "skipped_posts": len(skipped_indices) if skipped_indices else 0,
            "warning": diagnostics.warning,
        },
        "sentiment_method": settings.sentiment_method,
        "sentiment_model": (
            settings.sentiment_model_name
            if settings.sentiment_method in {"transformer", "hybrid"}
            else None
        ),
        "sentiment_confidence_threshold": settings.sentiment_confidence_threshold,
        "clusters": [],
    }

    for cluster in remapped_clusters:
        cluster_posts = [posts[idx] for idx in cluster.indices]
        cluster_texts = [all_texts[idx] for idx in cluster.indices]

        # Get clustering_text versions for keyword extraction
        cluster_clustering_texts = []
        for idx in cluster.indices:
            if idx in clustered_original_indices and idx in set(valid_indices):
                # Find position in valid_indices
                try:
                    vi_pos = valid_indices.index(idx)
                    cluster_clustering_texts.append(clustering_texts[vi_pos])
                except ValueError:
                    cluster_clustering_texts.append(all_texts[idx])
            else:
                cluster_clustering_texts.append(all_texts[idx])

        # Annotate each post with its individual sentiment
        for idx in cluster.indices:
            if idx < len(posts):
                posts[idx]["post_sentiment"] = sentiment_by_index.get(idx, "neutral")
                posts[idx]["post_sentiment_result"] = sentiment_result_by_index.get(idx, {})

        # Compute per-post sentiment distribution for this cluster
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for idx in cluster.indices:
            s = sentiment_by_index.get(idx, "neutral")
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        sentiment_dist = _sentiment_counts_to_distribution(sentiment_counts)
        dominant_sentiment = _dominant_sentiment(sentiment_counts)

        # Extract top keywords and representative posts for this cluster
        top_keywords = extract_top_keywords(
            cluster_clustering_texts, top_n=10, search_terms=search_term_stopwords,
        )
        representative = select_representative_posts(cluster_clustering_texts, limit=5)

        clusters_payload.append(
            {
                "cluster_id": cluster.cluster_id,
                "size": len(cluster.indices),
                "post_indices": cluster.indices,
                "posts": cluster_posts,
                "sentiment_counts": sentiment_counts,
                "sentiment_distribution": sentiment_dist,
                "dominant_sentiment": dominant_sentiment,
                "top_keywords": [{"keyword": kw, "count": cnt} for kw, cnt in top_keywords],
                "representative_posts": representative,
            }
        )

        # Cluster-level topic labeling via LLM. Sentiment is already aggregated
        # above from individual post-level sentiment results.
        topic_analysis = topic_labeler.analyze_cluster(
            cluster.cluster_id,
            cluster_texts,
            top_keywords=top_keywords,
            representative_texts=representative,
            search_terms=search_term_stopwords,
            metadata={"size": len(cluster.indices)},
        )
        analysis_payload["clusters"].append(
            {
                "cluster_id": topic_analysis.cluster_id,
                "topic_label": topic_analysis.topic_label,
                "topic_summary": topic_analysis.topic_summary,
                "rationale": topic_analysis.rationale,
                "marketing_interpretation": topic_analysis.marketing_interpretation,
                "sentiment": dominant_sentiment,
                "dominant_sentiment": dominant_sentiment,
                "size": len(cluster.indices),
                "sentiment_counts": sentiment_counts,
                "sentiment_distribution": sentiment_dist,
                "top_keywords": [{"keyword": kw, "count": cnt} for kw, cnt in top_keywords],
                "representative_posts": representative,
            }
        )

        # Delay between cluster LLM calls to stay under Gemini free-tier rate limit
        has_topic_api = (
            (topic_labeler.provider == "gemini" and topic_labeler.gemini_api_key)
            or (topic_labeler.provider == "openai" and topic_labeler.openai_api_key)
        )
        if has_topic_api:
            time.sleep(10.0)

    clusters_path = output_dir / "clusters.json"
    analysis_path = output_dir / "analysis.json"

    with clusters_path.open("w", encoding="utf-8") as file:
        json.dump({"clusters": clusters_payload}, file, ensure_ascii=False, indent=2)

    with analysis_path.open("w", encoding="utf-8") as file:
        json.dump(analysis_payload, file, ensure_ascii=False, indent=2)

    return clusters_path, analysis_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis pipeline on processed JSONL posts.")
    parser.add_argument("--input", required=True, help="Input processed JSONL path.")
    parser.add_argument("--output-dir", default="data/reports", help="Output directory for JSON reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clusters_path, analysis_path = run_analysis(args.input, args.output_dir)
    print(f"Saved clusters to: {clusters_path}")
    print(f"Saved analysis to: {analysis_path}")


if __name__ == "__main__":
    main()
