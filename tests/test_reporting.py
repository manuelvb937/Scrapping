import csv
import json
from pathlib import Path

from reporting.generator import generate_reports


def test_generate_reports_creates_markdown_html_and_csv(tmp_path: Path) -> None:
    clusters_path = tmp_path / "clusters.json"
    analysis_path = tmp_path / "analysis.json"

    clusters_payload = {
        "clusters": [
            {
                "cluster_id": 0,
                "size": 2,
                "post_indices": [0, 1],
                "posts": [
                    {"content": "Love the new feature"},
                    {"content": "Great campaign launch"},
                ],
                "sentiment_distribution": {"positive": 2, "negative": 0, "neutral": 0},
            },
            {
                "cluster_id": 1,
                "size": 1,
                "post_indices": [2],
                "posts": [{"content": "Terrible user experience"}],
                "sentiment_distribution": {"positive": 0, "negative": 1, "neutral": 0},
            },
        ]
    }
    analysis_payload = {
        "clusters": [
            {
                "cluster_id": 0,
                "topic_label": "feature_launch",
                "sentiment": "positive",
                "rationale": "Users celebrate launch and value.",
                "size": 2,
                "sentiment_distribution": {"positive": 2, "negative": 0, "neutral": 0},
            },
            {
                "cluster_id": 1,
                "topic_label": "ux_issue",
                "sentiment": "negative",
                "rationale": "Users mention experience issues.",
                "size": 1,
                "sentiment_distribution": {"positive": 0, "negative": 1, "neutral": 0},
            },
        ]
    }

    clusters_path.write_text(json.dumps(clusters_payload), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis_payload), encoding="utf-8")

    md_path, html_path, csv_path = generate_reports(clusters_path, analysis_path, tmp_path)

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    assert "Social Listening Marketing Report" in md_content
    assert "Sentiment Distribution" in md_content
    assert "Positive Topics" in md_content
    assert "Negative Topics" in md_content
    assert "Representative posts" in md_content
    # Improvement #5: Marketing strategies section should be present
    assert "Marketing Strategy Recommendations" in md_content
    # Per-post sentiment distribution should appear in cluster details
    assert "Post-level sentiment" in md_content

    assert "<html>" in html_content
    assert "feature_launch" in html_content
    # HTML should properly render bold text
    assert "<strong>" in html_content

    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert {row["topic_label"] for row in rows} == {"feature_launch", "ux_issue"}
    # CSV should include sentiment percentages
    assert "positive_pct" in rows[0]
