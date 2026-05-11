"""Post-level sentiment analysis using transformer models.

The default path uses a Hugging Face model that is already fine-tuned for
sentiment analysis. An LLM can still review ambiguous or low-confidence posts in
hybrid mode, but it is no longer the only sentiment analyzer.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
SENTIMENT_VALUES = ("positive", "neutral", "negative")

# These Japanese fandom/SNS phrases often carry sentiment that literal models
# can misread. For example, "死んだ" can be positive in fan excitement.
AMBIGUITY_MARKERS = [
    "しんどい",
    "死んだ",
    "やばい",
    "草",
    "沼",
    "尊い",
    "供給",
    "無理",
    "泣いた",
    "えぐい",
    "最高だけど",
    "微妙",
]

# Common mappings for sentiment classifiers. Cardiff's Twitter XLM-R model uses
# LABEL_0/LABEL_1/LABEL_2 for negative/neutral/positive. Some multilingual
# models expose star ratings or "very positive" style labels; we fold those into
# the required three labels.
LABEL_MAPPING = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "0": "negative",
    "1": "neutral",
    "2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very negative": "negative",
    "very_negative": "negative",
    "very positive": "positive",
    "very_positive": "positive",
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
    "one star": "negative",
    "two stars": "negative",
    "three stars": "neutral",
    "four stars": "positive",
    "five stars": "positive",
}


@dataclass
class TransformerSentimentResult:
    text: str
    label: str
    confidence: float
    raw_scores: dict[str, float]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": self.confidence,
            "raw_scores": self.raw_scores,
            "model": self.model,
        }


class TransformerSentimentAnalyzer:
    """Run post-level sentiment classification with a Hugging Face model."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int = 32,
        device: int | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("SENTIMENT_MODEL_NAME", DEFAULT_SENTIMENT_MODEL)
        self.batch_size = batch_size
        self.device = device
        self._classifier = None

    def analyze(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Classify a list of posts and return normalized sentiment dicts."""
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return []

        classifier = self._load_classifier()
        results: list[dict[str, Any]] = []

        for start in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[start : start + self.batch_size]
            raw_batch = classifier(batch, truncation=True)
            normalized_batch = self._ensure_batch_results(raw_batch, expected_size=len(batch))

            for text, raw_scores in zip(batch, normalized_batch):
                results.append(self._build_result(text, raw_scores).to_dict())

        return results

    def _load_classifier(self):
        """Load the transformer pipeline lazily.

        Keeping imports here lets tests import this module without installing or
        downloading transformer weights.
        """
        if self._classifier is not None:
            return self._classifier

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Transformer sentiment requires the 'transformers' package. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        kwargs: dict[str, Any] = {
            "task": "text-classification",
            "model": self.model_name,
            "tokenizer": self.model_name,
            "top_k": None,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        try:
            self._classifier = pipeline(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load sentiment model '{self.model_name}'. "
                "Check internet/model cache access or set SENTIMENT_MODEL_NAME."
            ) from exc

        return self._classifier

    def _ensure_batch_results(self, raw_batch: Any, *, expected_size: int) -> list[list[dict[str, Any]]]:
        """Normalize the shapes returned by different transformers versions."""
        if expected_size == 1 and raw_batch and isinstance(raw_batch[0], Mapping):
            return [list(raw_batch)]
        if raw_batch and isinstance(raw_batch[0], list):
            return raw_batch
        if raw_batch and isinstance(raw_batch[0], Mapping):
            return [[item] for item in raw_batch]
        return [[] for _ in range(expected_size)]

    def _build_result(self, text: str, raw_model_scores: Sequence[Mapping[str, Any]]) -> TransformerSentimentResult:
        raw_scores = normalize_raw_scores(raw_model_scores)
        label = max(raw_scores, key=raw_scores.get)
        confidence = raw_scores[label]
        return TransformerSentimentResult(
            text=text,
            label=label,
            confidence=round(confidence, 4),
            raw_scores={key: round(value, 4) for key, value in raw_scores.items()},
            model=self.model_name,
        )


class LLMSentimentReviewer:
    """LLM fallback used only for low-confidence or ambiguous posts."""

    def __init__(self, model: str | None = None) -> None:
        from .llm_labeling import LLMTopicSentimentLabeler

        self.legacy_labeler = LLMTopicSentimentLabeler(model=model)

    @property
    def provider(self) -> str:
        return self.legacy_labeler.provider

    @property
    def has_api_key(self) -> bool:
        return (
            (self.provider == "gemini" and bool(self.legacy_labeler.gemini_api_key))
            or (self.provider == "openai" and bool(self.legacy_labeler.openai_api_key))
        )

    def review_text(self, text: str) -> dict[str, str]:
        """Ask the LLM for sentiment and rationale for one ambiguous post."""
        if not self.has_api_key:
            label = self.legacy_labeler._heuristic_sentiment(text)
            return {
                "label": label,
                "rationale": "LLM sentiment fallback unavailable; keyword heuristic was used.",
            }

        if self.provider == "gemini":
            parsed = self.legacy_labeler._call_with_retry(lambda: self._review_with_gemini(text))
        else:
            parsed = self.legacy_labeler._call_with_retry(lambda: self._review_with_openai(text))

        label = normalize_sentiment_label(parsed.get("label") or parsed.get("sentiment"))
        return {
            "label": label,
            "rationale": str(parsed.get("rationale") or "").strip(),
        }

    def _review_with_openai(self, text: str) -> dict[str, Any]:
        payload = {
            "model": self.legacy_labeler.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _llm_review_prompt()}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sentiment_review",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["label", "rationale"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        }

        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.legacy_labeler.openai_api_key}",
            },
            method="POST",
        )

        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return self.legacy_labeler._extract_openai_structured_json(body)

    def _review_with_gemini(self, text: str) -> dict[str, Any]:
        payload = {
            "contents": [{"parts": [{"text": _llm_review_prompt() + "\n\nPost:\n" + text}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {
                            "type": "STRING",
                            "enum": ["positive", "negative", "neutral"],
                        },
                        "rationale": {"type": "STRING"},
                    },
                    "required": ["label", "rationale"],
                },
            },
        }

        params = urlencode({"key": self.legacy_labeler.gemini_api_key})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.legacy_labeler.model}:generateContent?{params}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return self.legacy_labeler._extract_gemini_structured_json(body)


class HybridSentimentAnalyzer:
    """Transformer-first sentiment with optional LLM review."""

    def __init__(
        self,
        transformer: TransformerSentimentAnalyzer | None = None,
        llm_reviewer: LLMSentimentReviewer | None = None,
        *,
        confidence_threshold: float | None = None,
        enable_llm_fallback: bool = True,
    ) -> None:
        self.transformer = transformer or TransformerSentimentAnalyzer()
        self.llm_reviewer = llm_reviewer or LLMSentimentReviewer()
        self.confidence_threshold = (
            DEFAULT_CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self.enable_llm_fallback = enable_llm_fallback

    def analyze(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        transformer_results = self.transformer.analyze(texts)
        hybrid_results: list[dict[str, Any]] = []

        for result in transformer_results:
            text = str(result.get("text", ""))
            ambiguity_triggered = contains_ambiguity_marker(text)
            confidence = float(result.get("confidence", 0.0))
            low_confidence = confidence < self.confidence_threshold
            should_review = self.enable_llm_fallback and (low_confidence or ambiguity_triggered)

            llm_review = None
            final_sentiment = normalize_sentiment_label(result.get("label"))
            source = "transformer"

            if should_review:
                llm_review = self.llm_reviewer.review_text(text)
                final_sentiment = normalize_sentiment_label(llm_review.get("label"))
                source = "llm_fallback" if self.llm_reviewer.has_api_key else "heuristic_fallback"

            hybrid_results.append(
                {
                    "text": text,
                    "transformer_sentiment": {
                        "label": normalize_sentiment_label(result.get("label")),
                        "confidence": confidence,
                        "raw_scores": result.get("raw_scores", {}),
                        "model": result.get("model", self.transformer.model_name),
                    },
                    "llm_review": llm_review,
                    "final_sentiment": final_sentiment,
                    "sentiment_source": source,
                    "ambiguity_triggered": ambiguity_triggered,
                }
            )

        return hybrid_results


def analyze_sentiment_transformer(
    texts: list[str],
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Recommended convenience function for transformer-only sentiment."""
    return TransformerSentimentAnalyzer(model_name=model_name).analyze(texts)


def analyze_sentiment_hybrid(
    texts: list[str],
    *,
    model_name: str | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    enable_llm_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Convenience function for transformer sentiment plus LLM review."""
    transformer = TransformerSentimentAnalyzer(model_name=model_name)
    analyzer = HybridSentimentAnalyzer(
        transformer=transformer,
        confidence_threshold=confidence_threshold,
        enable_llm_fallback=enable_llm_fallback,
    )
    return analyzer.analyze(texts)


def contains_ambiguity_marker(text: str) -> bool:
    return any(marker in text for marker in AMBIGUITY_MARKERS)


def normalize_sentiment_label(label: Any) -> str:
    label_text = str(label or "").strip().lower().replace("-", " ").replace("_", " ")
    compact = label_text.replace(" ", "_")

    if label_text in LABEL_MAPPING:
        return LABEL_MAPPING[label_text]
    if compact in LABEL_MAPPING:
        return LABEL_MAPPING[compact]
    if "negative" in label_text:
        return "negative"
    if "positive" in label_text:
        return "positive"
    if "neutral" in label_text:
        return "neutral"
    return "neutral"


def normalize_raw_scores(raw_model_scores: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    for item in raw_model_scores:
        label = normalize_sentiment_label(item.get("label"))
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores[label] += score

    total = sum(scores.values())
    if total <= 0:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    return {key: value / total for key, value in scores.items()}


def _llm_review_prompt() -> str:
    markers = ", ".join(AMBIGUITY_MARKERS)
    return (
        "You are a Japanese social media sentiment reviewer. Classify exactly one "
        "post as positive, negative, or neutral and explain briefly. Pay special "
        "attention to fandom and SNS slang, where phrases such as '死んだ', "
        "'しんどい', or '無理' may express strong positive emotion depending on "
        "context. Return JSON with keys: label and rationale. Ambiguity markers: "
        f"{markers}"
    )
