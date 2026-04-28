"""Generate markdown/html/csv reports from clustering analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sentiment_distribution(analysis_clusters: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for cluster in analysis_clusters:
        sentiment = str(cluster.get("sentiment", "neutral"))
        size = int(cluster.get("size", 1))
        counter[sentiment] += size
    return dict(counter)


def get_representative_posts(cluster: dict, limit: int = 3) -> list[str]:
    posts = cluster.get("posts", [])
    snippets: list[str] = []
    for post in posts[:limit]:
        text = post.get("cleaned_content") or post.get("content") or ""
        text = str(text).strip().replace("\n", " ")
        if text:
            snippets.append(text[:220])
    return snippets


def build_cluster_rows(clusters_data: dict, analysis_data: dict) -> list[dict]:
    analysis_by_id = {int(item["cluster_id"]): item for item in analysis_data.get("clusters", [])}
    rows: list[dict] = []

    for cluster in clusters_data.get("clusters", []):
        cluster_id = int(cluster.get("cluster_id", -1))
        summary = analysis_by_id.get(cluster_id, {})
        rep_posts = get_representative_posts(cluster)

        rows.append(
            {
                "cluster_id": cluster_id,
                "topic_label": summary.get("topic_label", "unknown"),
                "sentiment": summary.get("sentiment", "neutral"),
                "size": int(cluster.get("size", len(cluster.get("posts", [])))),
                "representative_posts": rep_posts,
                "rationale": summary.get("rationale", ""),
            }
        )

    rows.sort(key=lambda x: x["size"], reverse=True)
    return rows


def generate_clusters_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["cluster_id", "topic_label", "sentiment", "size", "representative_posts", "rationale"],
        )
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["representative_posts"] = " | ".join(row_copy["representative_posts"])
            writer.writerow(row_copy)


def generate_markdown(rows: list[dict], sentiment_counts: dict[str, int]) -> str:
    positive_topics = [r for r in rows if r["sentiment"] == "positive"]
    negative_topics = [r for r in rows if r["sentiment"] == "negative"]

    lines: list[str] = [
        "# Social Listening Marketing Report",
        "",
        "## Executive Summary",
        "This report summarizes cluster-level conversation themes and sentiment for marketing decisions.",
        "",
        "## Sentiment Distribution",
    ]

    total = sum(sentiment_counts.values()) or 1
    for sentiment in ["positive", "neutral", "negative"]:
        count = sentiment_counts.get(sentiment, 0)
        pct = (count / total) * 100
        lines.append(f"- **{sentiment.title()}**: {count} posts ({pct:.1f}%)")

    lines.extend(["", "## Positive Topics (Opportunity Areas)"])
    if positive_topics:
        for topic in positive_topics[:10]:
            lines.append(f"- **{topic['topic_label']}** (cluster {topic['cluster_id']}, n={topic['size']})")
    else:
        lines.append("- No clearly positive clusters detected.")

    lines.extend(["", "## Negative Topics (Risk Areas)"])
    if negative_topics:
        for topic in negative_topics[:10]:
            lines.append(f"- **{topic['topic_label']}** (cluster {topic['cluster_id']}, n={topic['size']})")
    else:
        lines.append("- No clearly negative clusters detected.")

    lines.extend(["", "## Cluster Details"]) 
    for row in rows:
        lines.extend(
            [
                f"### Cluster {row['cluster_id']}: {row['topic_label']}",
                f"- Sentiment: **{row['sentiment']}**",
                f"- Volume: **{row['size']}** posts",
                f"- Interpretation: {row['rationale']}",
                "- Representative posts:",
            ]
        )
        if row["representative_posts"]:
            for post in row["representative_posts"]:
                lines.append(f"  - {post}")
        else:
            lines.append("  - (No representative text available)")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def markdown_to_html(markdown: str) -> str:
    """Minimal markdown-to-HTML conversion for generated report sections."""
    html_lines: list[str] = ["<html><head><meta charset='utf-8'><title>Social Listening Report</title></head><body>"]

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("- "):
            html_lines.append(f"<p>• {stripped[2:]}</p>")
        elif stripped.startswith("  - "):
            html_lines.append(f"<p style='margin-left:20px;'>• {stripped[4:]}</p>")
        elif stripped == "":
            html_lines.append("<br/>")
        else:
            html_lines.append(f"<p>{stripped}</p>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def generate_reports(
    clusters_json_path: str | Path,
    analysis_json_path: str | Path,
    output_dir: str | Path = "data/reports",
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters_data = load_json(Path(clusters_json_path))
    analysis_data = load_json(Path(analysis_json_path))

    rows = build_cluster_rows(clusters_data, analysis_data)
    sentiment_counts = sentiment_distribution(analysis_data.get("clusters", []))

    report_md = generate_markdown(rows, sentiment_counts)
    report_html = markdown_to_html(report_md)

    report_md_path = output_dir / "report.md"
    report_html_path = output_dir / "report.html"
    clusters_csv_path = output_dir / "clusters.csv"

    report_md_path.write_text(report_md, encoding="utf-8")
    report_html_path.write_text(report_html, encoding="utf-8")
    generate_clusters_csv(rows, clusters_csv_path)

    return report_md_path, report_html_path, clusters_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report.md/report.html/clusters.csv from analysis outputs.")
    parser.add_argument("--clusters", required=True, help="Path to clusters.json")
    parser.add_argument("--analysis", required=True, help="Path to analysis.json")
    parser.add_argument("--output-dir", default="data/reports", help="Output directory for report artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_path, html_path, csv_path = generate_reports(args.clusters, args.analysis, args.output_dir)
    print(f"Saved markdown report to: {md_path}")
    print(f"Saved HTML report to: {html_path}")
    print(f"Saved clusters CSV to: {csv_path}")


if __name__ == "__main__":
    main()
