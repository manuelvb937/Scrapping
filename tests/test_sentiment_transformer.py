from analysis.sentiment_transformer import (
    HybridSentimentAnalyzer,
    normalize_raw_scores,
    normalize_sentiment_label,
)


def test_cardiff_label_mapping_normalizes_to_three_sentiments() -> None:
    scores = normalize_raw_scores(
        [
            {"label": "LABEL_0", "score": 0.05},
            {"label": "LABEL_1", "score": 0.13},
            {"label": "LABEL_2", "score": 0.82},
        ]
    )

    assert scores["negative"] == 0.05
    assert scores["neutral"] == 0.13
    assert scores["positive"] == 0.82


def test_multiclass_labels_fold_into_positive_neutral_negative() -> None:
    scores = normalize_raw_scores(
        [
            {"label": "Very Negative", "score": 0.20},
            {"label": "Neutral", "score": 0.30},
            {"label": "Very Positive", "score": 0.50},
        ]
    )

    assert scores["negative"] == 0.20
    assert scores["neutral"] == 0.30
    assert scores["positive"] == 0.50


def test_unknown_label_defaults_to_neutral() -> None:
    assert normalize_sentiment_label("unclear") == "neutral"


def test_hybrid_reviews_ambiguous_fandom_slang() -> None:
    class FakeTransformer:
        model_name = "fake-model"

        def analyze(self, texts):
            return [
                {
                    "text": texts[0],
                    "label": "negative",
                    "confidence": 0.76,
                    "raw_scores": {"negative": 0.76, "neutral": 0.14, "positive": 0.10},
                    "model": self.model_name,
                }
            ]

    class FakeReviewer:
        has_api_key = True

        def review_text(self, text):
            return {
                "label": "positive",
                "rationale": "Fan slang uses this as strong positive excitement.",
            }

    analyzer = HybridSentimentAnalyzer(
        transformer=FakeTransformer(),
        llm_reviewer=FakeReviewer(),
        confidence_threshold=0.70,
        review_delay_seconds=0,
    )

    result = analyzer.analyze(["尊すぎて死んだ"])[0]

    assert result["ambiguity_triggered"] is True
    assert result["final_sentiment"] == "positive"
    assert result["sentiment_source"] == "llm_fallback"


def test_hybrid_keeps_transformer_result_when_llm_review_fails() -> None:
    class FakeTransformer:
        model_name = "fake-model"

        def analyze(self, texts):
            return [
                {
                    "text": texts[0],
                    "label": "positive",
                    "confidence": 0.50,
                    "raw_scores": {"negative": 0.20, "neutral": 0.30, "positive": 0.50},
                    "model": self.model_name,
                }
            ]

    class FailingReviewer:
        has_api_key = True

        def review_text(self, text):
            raise RuntimeError("rate limit")

    analyzer = HybridSentimentAnalyzer(
        transformer=FakeTransformer(),
        llm_reviewer=FailingReviewer(),
        confidence_threshold=0.70,
        review_delay_seconds=0,
    )

    result = analyzer.analyze(["最高だけど少し不安"])[0]

    assert result["final_sentiment"] == "positive"
    assert result["sentiment_source"] == "transformer_llm_review_failed"


def test_hybrid_batches_llm_reviews() -> None:
    class FakeTransformer:
        model_name = "fake-model"

        def analyze(self, texts):
            return [
                {
                    "text": text,
                    "label": "neutral",
                    "confidence": 0.40,
                    "raw_scores": {"negative": 0.30, "neutral": 0.40, "positive": 0.30},
                    "model": self.model_name,
                }
                for text in texts
            ]

    class BatchReviewer:
        has_api_key = True

        def __init__(self):
            self.batch_sizes = []

        def review_texts(self, texts):
            self.batch_sizes.append(len(texts))
            return [
                {"label": "positive", "rationale": "batch reviewed"}
                for _ in texts
            ]

    reviewer = BatchReviewer()
    analyzer = HybridSentimentAnalyzer(
        transformer=FakeTransformer(),
        llm_reviewer=reviewer,
        confidence_threshold=0.70,
        fallback_batch_size=2,
        review_delay_seconds=0,
    )

    results = analyzer.analyze(["a", "b", "c", "d", "e"])

    assert reviewer.batch_sizes == [2, 2, 1]
    assert {item["final_sentiment"] for item in results} == {"positive"}
