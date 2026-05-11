from analysis.topic_labeling import TopicLabeler, TopicAnalysis


def test_topic_labeler_heuristic_has_no_cluster_sentiment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    labeler = TopicLabeler()
    result = labeler.analyze_cluster(
        0,
        ["ドラマSeason2をずっと待ってる", "続編のキャスト続投が嬉しい"],
        top_keywords=[("Season2", 3), ("続編", 2)],
    )

    assert isinstance(result, TopicAnalysis)
    assert result.cluster_id == 0
    assert result.topic_label == "Season2"
    assert not hasattr(result, "sentiment")


def test_topic_prompt_tells_llm_not_to_classify_sentiment() -> None:
    labeler = TopicLabeler(provider="openai", model="dummy")
    prompt = labeler._build_topic_prompt(
        ["公式の続編発表を待っている"],
        top_keywords=[("続編", 4)],
        search_terms={"ドラマ名"},
    )

    assert "Do not classify sentiment" in prompt
    assert "ドラマ名" in prompt
