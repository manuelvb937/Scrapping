"""Generate markdown/html/csv reports from clustering analysis outputs.

Improvements:
  #5 — LLM-generated marketing strategy recommendations
  Per-post sentiment distribution in reports
  Fixed HTML bold rendering
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sentiment_distribution(analysis_clusters: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for cluster in analysis_clusters:
        # Use per-post sentiment distribution if available, else fall back to cluster-level
        dist = cluster.get("sentiment_distribution")
        if dist:
            for sentiment, count in dist.items():
                counter[sentiment] += count
        else:
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

        # Per-post sentiment distribution
        sentiment_dist = (
            summary.get("sentiment_distribution")
            or cluster.get("sentiment_distribution")
            or {}
        )

        rows.append(
            {
                "cluster_id": cluster_id,
                "topic_label": summary.get("topic_label", "unknown"),
                "sentiment": summary.get("sentiment", "neutral"),
                "size": int(cluster.get("size", len(cluster.get("posts", [])))),
                "representative_posts": rep_posts,
                "rationale": summary.get("rationale", ""),
                "sentiment_distribution": sentiment_dist,
            }
        )

    rows.sort(key=lambda x: x["size"], reverse=True)
    return rows


def generate_clusters_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "cluster_id", "topic_label", "sentiment", "size",
                "positive_pct", "negative_pct", "neutral_pct",
                "representative_posts", "rationale",
            ],
        )
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            dist = row_copy.pop("sentiment_distribution", {})
            total = sum(dist.values()) or 1
            row_copy["positive_pct"] = f"{(dist.get('positive', 0) / total) * 100:.1f}%"
            row_copy["negative_pct"] = f"{(dist.get('negative', 0) / total) * 100:.1f}%"
            row_copy["neutral_pct"] = f"{(dist.get('neutral', 0) / total) * 100:.1f}%"
            row_copy["representative_posts"] = " | ".join(row_copy["representative_posts"])
            writer.writerow(row_copy)


def generate_marketing_strategies(
    rows: list[dict],
    sentiment_counts: dict[str, int],
) -> str:
    """Generate LLM-based marketing strategy recommendations.

    Improvement #5: Calls the LLM with the full analysis summary to produce
    actionable marketing strategies. Falls back to template-based strategies
    when no API key is available.
    """
    from analysis.llm_labeling import LLMTopicSentimentLabeler

    positive_topics = [r for r in rows if r["sentiment"] == "positive"]
    negative_topics = [r for r in rows if r["sentiment"] == "negative"]

    total = sum(sentiment_counts.values()) or 1
    pos_pct = (sentiment_counts.get("positive", 0) / total) * 100
    neg_pct = (sentiment_counts.get("negative", 0) / total) * 100
    neu_pct = (sentiment_counts.get("neutral", 0) / total) * 100

    summary_text = (
        f"Social Listening Analysis Summary:\n"
        f"- Total posts analyzed: {total}\n"
        f"- Sentiment: {pos_pct:.0f}% positive, {neg_pct:.0f}% negative, {neu_pct:.0f}% neutral\n"
        f"- Top positive topics: {', '.join(r['topic_label'] for r in positive_topics[:5]) or 'None'}\n"
        f"- Top negative topics: {', '.join(r['topic_label'] for r in negative_topics[:5]) or 'None'}\n"
        f"- Largest topic clusters:\n"
    )
    for r in rows[:5]:
        summary_text += f"  - {r['topic_label']} ({r['size']} posts, {r['sentiment']})\n"

    labeler = LLMTopicSentimentLabeler()
    has_api = (
        (labeler.provider == "gemini" and labeler.gemini_api_key)
        or (labeler.provider == "openai" and labeler.openai_api_key)
    )

    if not has_api:
        LOGGER.info("No LLM API key — using template-based marketing strategies")
        return _template_strategies(rows, positive_topics, negative_topics)

    strategy_prompt = (
        "You are a marketing strategist analyzing social media listening data. "
        "Based on the following analysis, generate exactly 5 specific, actionable "
        "marketing strategies. For each strategy, provide:\n"
        "1. Strategy name\n"
        "2. Target audience\n"
        "3. Recommended action\n"
        "4. Expected impact\n"
        "5. Risk assessment\n\n"
        "If the data is in Japanese, write strategies in Japanese.\n"
        "日本語のデータの場合、戦略も日本語で記述してください。\n\n"
        f"{summary_text}"
    )

    try:
        if labeler.provider == "gemini" and labeler.gemini_api_key:
            result = labeler._call_with_retry(
                lambda: _call_gemini_strategy(labeler, strategy_prompt)
            )
        else:
            result = labeler._call_with_retry(
                lambda: _call_openai_strategy(labeler, strategy_prompt)
            )
        return result
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("LLM strategy generation failed: %s. Using template fallback.", exc)
        return _template_strategies(rows, positive_topics, negative_topics)


def _call_gemini_strategy(labeler, prompt: str) -> str:
    """Call Gemini API for marketing strategy text."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
    }

    params = urlencode({"key": labeler.gemini_api_key})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{labeler.model}:generateContent?{params}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))

    candidates = body.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return text
    return ""


def _call_openai_strategy(labeler, prompt: str) -> str:
    """Call OpenAI API for marketing strategy text."""
    from urllib.request import Request, urlopen

    payload = {
        "model": labeler.model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
    }

    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {labeler.openai_api_key}",
        },
        method="POST",
    )

    with urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))

    output = body.get("output", [])
    for entry in output:
        for content in entry.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    return ""


def _template_strategies(
    rows: list[dict],
    positive_topics: list[dict],
    negative_topics: list[dict],
) -> str:
    """Generate template-based strategies when no LLM API is available."""
    lines: list[str] = []

    if positive_topics:
        top = positive_topics[0]
        lines.extend([
            f"**Strategy 1: Amplify \"{top['topic_label']}\"**",
            f"- Target: Users discussing {top['topic_label']}",
            f"- Action: Create content that reinforces this positive narrative",
            f"- Impact: Leverage existing positive sentiment ({top['size']} posts)",
            f"- Risk: Low — aligns with organic positive conversation",
            "",
        ])

    if negative_topics:
        top_neg = negative_topics[0]
        lines.extend([
            f"**Strategy 2: Address \"{top_neg['topic_label']}\" Concerns**",
            f"- Target: Users expressing negative sentiment about {top_neg['topic_label']}",
            f"- Action: Create response content addressing concerns directly",
            f"- Impact: Mitigate negative perception ({top_neg['size']} posts)",
            f"- Risk: Medium — requires careful messaging to avoid amplifying negativity",
            "",
        ])

    if rows:
        largest = rows[0]
        lines.extend([
            f"**Strategy 3: Engage the Largest Community (\"{largest['topic_label']}\")**",
            f"- Target: The {largest['size']} users in the largest discussion cluster",
            f"- Action: Participate in or sponsor discussions around this topic",
            f"- Impact: Maximum reach due to volume",
            f"- Risk: Low — go where the conversation already is",
            "",
        ])

    lines.extend([
        "**Strategy 4: Monitor & Alert System**",
        "- Target: Brand and marketing team",
        "- Action: Set up recurring pipeline runs to track sentiment trends over time",
        "- Impact: Early detection of sentiment shifts enables proactive response",
        "- Risk: Low — operational improvement",
        "",
        "**Strategy 5: Influencer Identification**",
        "- Target: High-engagement users in positive clusters",
        "- Action: Identify and engage potential brand advocates from the scraped data",
        "- Impact: Organic amplification through authentic voices",
        "- Risk: Low-Medium — requires vetting for brand alignment",
    ])

    return "\n".join(lines)


def generate_markdown(rows: list[dict], sentiment_counts: dict[str, int], strategies: str = "") -> str:
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
        dist = row.get("sentiment_distribution", {})
        dist_total = sum(dist.values()) or 1

        lines.extend(
            [
                f"### Cluster {row['cluster_id']}: {row['topic_label']}",
                f"- Sentiment: **{row['sentiment']}**",
                f"- Volume: **{row['size']}** posts",
            ]
        )

        # Per-post sentiment breakdown
        if dist:
            pos_pct = (dist.get("positive", 0) / dist_total) * 100
            neg_pct = (dist.get("negative", 0) / dist_total) * 100
            neu_pct = (dist.get("neutral", 0) / dist_total) * 100
            lines.append(
                f"- Post-level sentiment: {pos_pct:.0f}% positive, "
                f"{neg_pct:.0f}% negative, {neu_pct:.0f}% neutral"
            )

        lines.append(f"- Interpretation: {row['rationale']}")
        lines.append("- Representative posts:")

        if row["representative_posts"]:
            for post in row["representative_posts"]:
                lines.append(f"  - {post}")
        else:
            lines.append("  - (No representative text available)")
        lines.append("")

    # Improvement #5: Marketing strategies section
    if strategies:
        lines.extend(["## Marketing Strategy Recommendations", "", strategies, ""])

    return "\n".join(lines).strip() + "\n"


def markdown_to_html(markdown: str) -> str:
    """Markdown-to-HTML conversion with proper bold rendering and list handling."""
    html_lines: list[str] = [
        "<html><head><meta charset='utf-8'>",
        "<title>Social Listening Report</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }",
        "  h1 { color: #1a1a2e; } h2 { color: #16213e; border-bottom: 1px solid #ddd; padding-bottom: 5px; }",
        "  h3 { color: #0f3460; } .pos { color: #27ae60; } .neg { color: #e74c3c; } .neu { color: #95a5a6; }",
        "  ul { margin: 5px 0; } li { margin: 3px 0; }",
        "</style>",
        "</head><body>",
    ]

    in_list = False

    for line in markdown.splitlines():
        stripped = line.strip()

        # Close list if we're no longer in one
        if in_list and not stripped.startswith("- ") and not stripped.startswith("  - "):
            html_lines.append("</ul>")
            in_list = False

        if stripped.startswith("### "):
            content = _bold_to_html(stripped[4:])
            html_lines.append(f"<h3>{content}</h3>")
        elif stripped.startswith("## "):
            content = _bold_to_html(stripped[3:])
            html_lines.append(f"<h2>{content}</h2>")
        elif stripped.startswith("# "):
            content = _bold_to_html(stripped[2:])
            html_lines.append(f"<h1>{content}</h1>")
        elif stripped.startswith("  - "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = _bold_to_html(stripped[4:])
            html_lines.append(f"<li style='margin-left:20px;'>{content}</li>")
        elif stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = _bold_to_html(stripped[2:])
            html_lines.append(f"<li>{content}</li>")
        elif stripped == "":
            if in_list:
                html_lines.append("</ul>")
                in_list = False
        else:
            content = _bold_to_html(stripped)
            html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def _bold_to_html(text: str) -> str:
    """Convert **bold** markdown to <strong> HTML tags."""
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


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

    # Improvement #5: Generate marketing strategies via LLM
    LOGGER.info("Generating marketing strategy recommendations...")
    strategies = generate_marketing_strategies(rows, sentiment_counts)

    report_md = generate_markdown(rows, sentiment_counts, strategies=strategies)
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
