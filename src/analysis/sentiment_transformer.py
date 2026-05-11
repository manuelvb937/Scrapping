"""Post-level sentiment analysis using transformer models.

The default path uses a Hugging Face model that is already fine-tuned for
sentiment analysis. An LLM can still review ambiguous or low-confidence posts in
hybrid mode, but it is no longer the only sentiment analyzer.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .llm_rate_limit import limited_llm_call

LOGGER = logging.getLogger(__name__)

DEFAULT_SENTIMENT_MODEL = "LoneWolfgang/bert-for-japanese-twitter-sentiment"
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

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # XLM-R/SentencePiece models can trigger a fast-tokenizer conversion
            # path that needs extra packages such as protobuf or tiktoken. The
            # slow tokenizer is reliable here and fast enough for this pipeline.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            kwargs: dict[str, Any] = {
                "task": "text-classification",
                "model": model,
                "tokenizer": tokenizer,
                "top_k": None,
            }
            if self.device is not None:
                kwargs["device"] = self.device

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
        """Ask the LLM for only the sentiment label of one ambiguous post."""
        return self.review_texts([text])[0]

    def review_texts(self, texts: Sequence[str]) -> list[dict[str, str]]:
        """Ask the LLM for sentiment labels for several posts at once."""
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return []

        if not self.has_api_key:
            return [
                {
                    "label": self.legacy_labeler._heuristic_sentiment(text),
                }
                for text in clean_texts
            ]

        if self.provider == "gemini":
            parsed = self.legacy_labeler._call_with_retry(lambda: self._review_batch_with_gemini(clean_texts))
        else:
            parsed = self.legacy_labeler._call_with_retry(lambda: self._review_batch_with_openai(clean_texts))

        return self._normalize_batch_reviews(clean_texts, parsed)

    def _review_batch_with_openai(self, texts: Sequence[str]) -> dict[str, Any]:
        numbered_posts = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(texts))
        payload = {
            "model": self.legacy_labeler.model,
            "temperature": _env_float("LLM_TEMPERATURE", 0.0),
            "top_p": _env_float("LLM_TOP_P", 1.0),
            "max_output_tokens": _env_int("LLM_SENTIMENT_MAX_OUTPUT_TOKENS", 1024),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _llm_review_prompt()}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": numbered_posts}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sentiment_review",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reviews": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "label": {
                                            "type": "string",
                                            "enum": ["positive", "negative", "neutral"],
                                        },
                                    },
                                    "required": ["index", "label"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["reviews"],
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

        def send_request() -> dict[str, Any]:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.legacy_labeler.model,
            "hybrid LLM sentiment review",
            _llm_review_prompt() + "\n\n" + numbered_posts,
            send_request,
        )
        return self.legacy_labeler._extract_openai_structured_json(body)

    def _review_batch_with_gemini(self, texts: Sequence[str]) -> dict[str, Any]:
        numbered_posts = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(texts))
        payload = {
            "contents": [{"parts": [{"text": _llm_review_prompt() + "\n\nPosts:\n" + numbered_posts}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": _env_float("LLM_TEMPERATURE", 0.0),
                "topP": _env_float("LLM_TOP_P", 1.0),
                "candidateCount": 1,
                "maxOutputTokens": _env_int("LLM_SENTIMENT_MAX_OUTPUT_TOKENS", 1024),
                "thinkingConfig": {
                    "thinkingBudget": _env_int("GEMINI_THINKING_BUDGET", 0),
                },
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "reviews": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "index": {"type": "INTEGER"},
                                    "label": {
                                        "type": "STRING",
                                        "enum": ["positive", "negative", "neutral"],
                                    },
                                },
                                "required": ["index", "label"],
                            },
                        },
                    },
                    "required": ["reviews"],
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

        def send_request() -> dict[str, Any]:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.legacy_labeler.model,
            "hybrid LLM sentiment review",
            _llm_review_prompt() + "\n\nPosts:\n" + numbered_posts,
            send_request,
        )
        return self.legacy_labeler._extract_gemini_structured_json(body)

    def _normalize_batch_reviews(
        self,
        texts: Sequence[str],
        parsed: Mapping[str, Any],
    ) -> list[dict[str, str]]:
        reviews_by_index: dict[int, dict[str, str]] = {}
        for item in parsed.get("reviews", []):
            try:
                index = int(item.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if index < 0:
                continue
            reviews_by_index[index] = {
                "label": normalize_sentiment_label(item.get("label") or item.get("sentiment")),
            }

        normalized: list[dict[str, str]] = []
        for index, text in enumerate(texts):
            normalized.append(
                reviews_by_index.get(
                    index,
                    {
                        "label": self.legacy_labeler._heuristic_sentiment(text),
                    },
                )
            )
        return normalized


class HybridSentimentAnalyzer:
    """Transformer-first sentiment with optional LLM review."""

    def __init__(
        self,
        transformer: TransformerSentimentAnalyzer | None = None,
        llm_reviewer: LLMSentimentReviewer | None = None,
        *,
        confidence_threshold: float | None = None,
        enable_llm_fallback: bool = True,
        review_delay_seconds: float | None = None,
        fallback_batch_size: int | None = None,
        max_reviews: int | None = None,
    ) -> None:
        self.transformer = transformer or TransformerSentimentAnalyzer()
        self.llm_reviewer = llm_reviewer or LLMSentimentReviewer()
        self.confidence_threshold = (
            DEFAULT_CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self.enable_llm_fallback = enable_llm_fallback
        self.review_delay_seconds = (
            _env_float("SENTIMENT_LLM_REVIEW_DELAY_SECONDS", 0.0)
            if review_delay_seconds is None
            else review_delay_seconds
        )
        self.fallback_batch_size = (
            max(1, _env_int("SENTIMENT_LLM_FALLBACK_BATCH_SIZE", 25))
            if fallback_batch_size is None
            else max(1, fallback_batch_size)
        )
        self.max_reviews = (
            _env_int("SENTIMENT_LLM_MAX_REVIEWS", 0)
            if max_reviews is None
            else max_reviews
        )

    def analyze(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        transformer_results = self.transformer.analyze(texts)
        hybrid_results: list[dict[str, Any]] = []
        review_candidate_positions: list[int] = []

        for result in transformer_results:
            text = str(result.get("text", ""))
            ambiguity_triggered = contains_ambiguity_marker(text)
            confidence = float(result.get("confidence", 0.0))
            low_confidence = confidence < self.confidence_threshold
            should_review = self.enable_llm_fallback and (low_confidence or ambiguity_triggered)

            llm_review = None
            final_sentiment = normalize_sentiment_label(result.get("label"))
            source = "transformer"

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
            if should_review:
                review_candidate_positions.append(len(hybrid_results) - 1)

        if self.max_reviews > 0:
            skipped = max(0, len(review_candidate_positions) - self.max_reviews)
            review_candidate_positions = review_candidate_positions[: self.max_reviews]
            if skipped:
                LOGGER.info("Skipped %d LLM sentiment reviews due to SENTIMENT_LLM_MAX_REVIEWS", skipped)

        for start in range(0, len(review_candidate_positions), self.fallback_batch_size):
            batch_positions = review_candidate_positions[start : start + self.fallback_batch_size]
            batch_texts = [hybrid_results[position]["text"] for position in batch_positions]
            try:
                reviews = self._review_texts(batch_texts)
                for position, review in zip(batch_positions, reviews):
                    final_sentiment = normalize_sentiment_label(review.get("label"))
                    hybrid_results[position]["llm_review"] = review
                    hybrid_results[position]["final_sentiment"] = final_sentiment
                    hybrid_results[position]["sentiment_source"] = (
                        "llm_fallback" if self.llm_reviewer.has_api_key else "heuristic_fallback"
                    )
                if self.llm_reviewer.has_api_key and self.review_delay_seconds > 0:
                    time.sleep(self.review_delay_seconds)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "LLM sentiment review batch failed; keeping transformer results: %s",
                    exc,
                )
                for position in batch_positions:
                    final_sentiment = hybrid_results[position]["final_sentiment"]
                    hybrid_results[position]["llm_review"] = {
                        "label": final_sentiment,
                    }
                    hybrid_results[position]["sentiment_source"] = "transformer_llm_review_failed"

        return hybrid_results

    def _review_texts(self, texts: Sequence[str]) -> list[dict[str, str]]:
        if hasattr(self.llm_reviewer, "review_texts"):
            return self.llm_reviewer.review_texts(texts)
        return [self.llm_reviewer.review_text(text) for text in texts]


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


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _llm_review_prompt() -> str:
    markers = ", ".join(AMBIGUITY_MARKERS)
    return (
        "You are a Japanese social media sentiment classifier. Classify each numbered "
        "post independently as positive, negative, or neutral. Return JSON only with "
        "attention to fandom and SNS slang, where phrases such as '死んだ', "
        "'しんどい', or '無理' may express strong positive emotion depending on "
        "context. Each review must contain only index and label. Do not include "
        "rationales or explanations. Ambiguity markers: "
        f"{markers}"
    )
