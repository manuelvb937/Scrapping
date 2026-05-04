import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from analysis.pipeline import run_analysis


def test_run_analysis_outputs_valid_json_files(tmp_path: Path) -> None:
    """Test the full analysis pipeline with mocked embedding model."""
    input_file = tmp_path / "processed_sample.jsonl"
    records = [
        {"content": "I love this product", "cleaned_content": "I love this product"},
        {"content": "This is terrible", "cleaned_content": "This is terrible"},
        {"content": "I love this", "cleaned_content": "I love this"},
        {"content": "Great experience overall", "cleaned_content": "Great experience overall"},
        {"content": "Not bad at all", "cleaned_content": "Not bad at all"},
        {"content": "Amazing quality", "cleaned_content": "Amazing quality"},
    ]

    with input_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    # Mock the TextEmbedder to avoid downloading the model in tests
    import numpy as np

    mock_embeddings = np.random.rand(len(records), 384).tolist()

    with patch("analysis.pipeline.TextEmbedder") as MockEmbedder:
        instance = MockEmbedder.return_value
        instance.embed_texts.return_value = mock_embeddings

        clusters_path, analysis_path = run_analysis(input_file, tmp_path)

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert "clusters" in clusters
    assert isinstance(clusters["clusters"], list)

    assert "clusters" in analysis
    assert isinstance(analysis["clusters"], list)
    assert "total_posts" in analysis

    for cluster in analysis["clusters"]:
        assert cluster["sentiment"] in {"positive", "negative", "neutral"}
        assert isinstance(cluster["topic_label"], str)
        # Per-post sentiment distribution should be present
        assert "sentiment_distribution" in cluster
