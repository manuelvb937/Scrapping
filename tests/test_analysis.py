import json
from pathlib import Path

from analysis.pipeline import run_analysis


def test_run_analysis_outputs_valid_json_files(tmp_path: Path) -> None:
    input_file = tmp_path / "processed_sample.jsonl"
    records = [
        {"content": "I love this product", "cleaned_content": "I love this product"},
        {"content": "This is terrible", "cleaned_content": "This is terrible"},
        {"content": "I love this", "cleaned_content": "I love this"},
    ]

    with input_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    clusters_path, analysis_path = run_analysis(input_file, tmp_path)

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert "clusters" in clusters
    assert isinstance(clusters["clusters"], list)

    assert "clusters" in analysis
    assert isinstance(analysis["clusters"], list)

    for cluster in analysis["clusters"]:
        assert cluster["sentiment"] in {"positive", "negative", "neutral"}
        assert isinstance(cluster["topic_label"], str)
