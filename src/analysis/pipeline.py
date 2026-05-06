"""Analysis pipeline: real embeddings, UMAP+HDBSCAN clustering, and LLM labeling.

Updated to use:
  - TextEmbedder (BGE-M3) for semantic embeddings (#1)
  - UMAP + HDBSCAN for clustering (#2)
  - Per-post sentiment analysis (#4)
  - Text preparation for clustering (title/noise stripping)
  - Diagnostics, representative posts, and top keywords per cluster
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .clustering import cluster_embeddings, Cluster
from .embeddings import TextEmbedder
from .llm_labeling import LLMTopicSentimentLabeler
from .text_preparation import (
    prepare_clustering_texts,
    extract_top_keywords,
    extract_search_term_fragments,
    select_representative_posts,
    clean_for_llm,
)

LOGGER = logging.getLogger(__name__)


def load_processed_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_analysis(input_path: str | Path, output_dir: str | Path = "data/reports") -> tuple[Path, Path]:
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
    labeler = LLMTopicSentimentLabeler()
    LOGGER.info("Running per-post sentiment analysis on %d posts...", len(llm_texts))
    post_sentiments = labeler.analyze_posts_batch(llm_texts)

    # Build sentiment lookup (index -> sentiment)
    sentiment_by_index: dict[int, str] = {}
    for ps in post_sentiments:
        sentiment_by_index[ps.index] = ps.sentiment

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

        # Compute per-post sentiment distribution for this cluster
        sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
        for idx in cluster.indices:
            s = sentiment_by_index.get(idx, "neutral")
            sentiment_dist[s] = sentiment_dist.get(s, 0) + 1

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
                "sentiment_distribution": sentiment_dist,
                "top_keywords": [{"keyword": kw, "count": cnt} for kw, cnt in top_keywords],
                "representative_posts": representative,
            }
        )

        # Cluster-level topic + sentiment analysis via LLM
        cluster_analysis = labeler.analyze_cluster(
            cluster.cluster_id,
            cluster_texts,
            top_keywords=top_keywords,
            representative_texts=representative,
            search_terms=search_term_stopwords,
        )
        analysis_payload["clusters"].append(
            {
                "cluster_id": cluster_analysis.cluster_id,
                "topic_label": cluster_analysis.topic_label,
                "sentiment": cluster_analysis.sentiment,
                "rationale": cluster_analysis.rationale,
                "size": len(cluster.indices),
                "sentiment_distribution": sentiment_dist,
                "top_keywords": [{"keyword": kw, "count": cnt} for kw, cnt in top_keywords],
                "representative_posts": representative,
            }
        )

        # Delay between cluster LLM calls to stay under Gemini free-tier rate limit
        import time
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
