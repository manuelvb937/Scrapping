"""Analysis pipeline: real embeddings, UMAP+HDBSCAN clustering, and LLM labeling.

Updated to use:
  - TextEmbedder (BGE-M3) for semantic embeddings (#1)
  - UMAP + HDBSCAN for clustering (#2)
  - Per-post sentiment analysis (#4)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .clustering import cluster_embeddings
from .embeddings import TextEmbedder
from .llm_labeling import LLMTopicSentimentLabeler

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
    texts = [str(post.get("cleaned_content") or post.get("content") or "") for post in posts]

    # Improvement #1: Use real multilingual embeddings (BGE-M3)
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(texts)

    # Improvement #2: Use UMAP + HDBSCAN clustering
    clusters = cluster_embeddings(embeddings)

    labeler = LLMTopicSentimentLabeler()

    # Improvement #4: Run per-post sentiment analysis
    LOGGER.info("Running per-post sentiment analysis on %d posts...", len(texts))
    post_sentiments = labeler.analyze_posts_batch(texts)

    # Build sentiment lookup (index -> sentiment)
    sentiment_by_index: dict[int, str] = {}
    for ps in post_sentiments:
        sentiment_by_index[ps.index] = ps.sentiment

    clusters_payload: list[dict] = []
    analysis_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "total_posts": len(posts),
        "clusters": [],
    }

    for cluster in clusters:
        cluster_posts = [posts[idx] for idx in cluster.indices]
        cluster_texts = [texts[idx] for idx in cluster.indices]

        # Annotate each post with its individual sentiment
        for idx in cluster.indices:
            if idx < len(posts):
                posts[idx]["post_sentiment"] = sentiment_by_index.get(idx, "neutral")

        # Compute per-post sentiment distribution for this cluster
        sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
        for idx in cluster.indices:
            s = sentiment_by_index.get(idx, "neutral")
            sentiment_dist[s] = sentiment_dist.get(s, 0) + 1

        clusters_payload.append(
            {
                "cluster_id": cluster.cluster_id,
                "size": len(cluster.indices),
                "post_indices": cluster.indices,
                "posts": cluster_posts,
                "sentiment_distribution": sentiment_dist,
            }
        )

        # Cluster-level topic + sentiment analysis via LLM
        cluster_analysis = labeler.analyze_cluster(cluster.cluster_id, cluster_texts)
        analysis_payload["clusters"].append(
            {
                "cluster_id": cluster_analysis.cluster_id,
                "topic_label": cluster_analysis.topic_label,
                "sentiment": cluster_analysis.sentiment,
                "rationale": cluster_analysis.rationale,
                "size": len(cluster.indices),
                "sentiment_distribution": sentiment_dist,
            }
        )

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
