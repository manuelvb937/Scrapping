from analysis.llm_labeling import LLMTopicSentimentLabeler


def test_detect_provider_gemini_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    labeler = LLMTopicSentimentLabeler()
    assert labeler.provider == "gemini"


def test_heuristic_mentions_missing_gemini_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    labeler = LLMTopicSentimentLabeler()
    result = labeler.analyze_cluster(1, ["最高だった"])
    assert "GEMINI_API_KEY" in result.rationale


def test_extract_gemini_structured_json() -> None:
    labeler = LLMTopicSentimentLabeler()
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"topic_label":"topic","sentiment":"positive","rationale":"good"}'
                        }
                    ]
                }
            }
        ]
    }

    parsed = labeler._extract_gemini_structured_json(response)
    assert parsed["topic_label"] == "topic"
    assert parsed["sentiment"] == "positive"
