"""Analysis pipeline: embeddings, clustering, and LLM labeling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .clustering import cluster_embeddings
from .embeddings import embed_text
from .llm_labeling import LLMTopicSentimentLabeler


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
    embeddings = [embed_text(text) for text in texts]
    clusters = cluster_embeddings(embeddings)

    clusters_payload: list[dict] = []
    labeler = LLMTopicSentimentLabeler()
    analysis_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "clusters": [],
    }

    for cluster in clusters:
        cluster_posts = [posts[idx] for idx in cluster.indices]
        cluster_texts = [texts[idx] for idx in cluster.indices]

        clusters_payload.append(
            {
                "cluster_id": cluster.cluster_id,
                "size": len(cluster.indices),
                "post_indices": cluster.indices,
                "posts": cluster_posts,
            }
        )

        cluster_analysis = labeler.analyze_cluster(cluster.cluster_id, cluster_texts)
        analysis_payload["clusters"].append(
            {
                "cluster_id": cluster_analysis.cluster_id,
                "topic_label": cluster_analysis.topic_label,
                "sentiment": cluster_analysis.sentiment,
                "rationale": cluster_analysis.rationale,
                "size": len(cluster.indices),
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
