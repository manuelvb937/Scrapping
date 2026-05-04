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


def test_heuristic_sentiment_expanded_japanese(monkeypatch) -> None:
    """Improvement #7: Expanded Japanese sentiment keywords."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    labeler = LLMTopicSentimentLabeler()

    # Test with Japanese positive words
    result = labeler.analyze_cluster(1, ["素晴らしい製品です", "最高の体験"])
    assert result.sentiment == "positive"

    # Test with Japanese negative words
    result = labeler.analyze_cluster(2, ["最悪のサービス", "ひどい対応"])
    assert result.sentiment == "negative"


def test_retry_logic_retries_on_failure() -> None:
    """Improvement #9: Retry logic with exponential backoff."""
    labeler = LLMTopicSentimentLabeler()

    call_count = 0

    def failing_then_succeeding():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("Connection timed out")
        return {"topic_label": "success", "sentiment": "positive", "rationale": "retry worked"}

    result = labeler._call_with_retry(failing_then_succeeding, max_retries=3)
    assert call_count == 3
    assert result["topic_label"] == "success"


def test_per_post_sentiment_heuristic(monkeypatch) -> None:
    """Improvement #4: Per-post sentiment via heuristic fallback."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    labeler = LLMTopicSentimentLabeler()
    texts = ["最高の製品です", "最悪のサービス", "普通のテキスト"]
    results = labeler.analyze_posts_batch(texts)

    assert len(results) == 3
    assert results[0].sentiment == "positive"
    assert results[1].sentiment == "negative"
    assert results[2].sentiment == "neutral"
