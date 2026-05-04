"""LLM-based topic and sentiment labeling with structured JSON output.

Improvements:
  #4 — Per-post sentiment analysis via batch LLM calls
  #7 — Japanese-aware bilingual prompts
  #9 — Retry logic with exponential backoff for API calls
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)
SENTIMENT_VALUES = {"positive", "negative", "neutral"}

# Improvement #7: Expanded Japanese sentiment keywords for heuristic fallback
POSITIVE_WORDS = {
    "good", "great", "love", "excellent", "happy", "success", "amazing", "awesome", "wonderful",
    "最高", "尊い", "神", "素晴らしい", "嬉しい", "楽しい", "面白い", "良い", "いい",
    "好き", "推し", "感動", "すごい", "やばい", "最強", "優秀", "便利", "期待",
}
NEGATIVE_WORDS = {
    "bad", "hate", "terrible", "awful", "fail", "problem", "worst", "horrible", "disappointing",
    "最悪", "微妙", "つまらない", "嫌い", "ひどい", "クソ", "だめ", "ダメ", "残念",
    "不満", "不安", "怖い", "危険", "問題", "炎上", "批判", "失望", "酷い",
}

# Improvement #7: Bilingual system prompts
CLUSTER_SYSTEM_PROMPT = (
    "You are a Japanese social media analyst specializing in social listening for marketing. "
    "Analyze the following cluster of social media posts (primarily Japanese). "
    "Return a JSON object with keys: topic_label (concise topic name in Japanese if posts are Japanese, "
    "otherwise in the posts' language), sentiment (one of: positive, negative, neutral), "
    "and rationale (brief explanation in Japanese if posts are Japanese). "
    "日本語のソーシャルメディア投稿を分析し、トピックとセンチメントを判定してください。"
)

POST_SENTIMENT_SYSTEM_PROMPT = (
    "You are a Japanese social media sentiment classifier. "
    "For each numbered post below, classify its sentiment as positive, negative, or neutral. "
    "Return a JSON array of objects with keys: index (the post number), sentiment. "
    "Handle Japanese text with cultural awareness — indirect criticism, sarcasm, and "
    "understatement are common in Japanese online discourse. "
    "日本語の投稿のセンチメントを分析してください。間接的な批判や皮肉にも注意してください。"
)


@dataclass
class ClusterAnalysis:
    cluster_id: int
    topic_label: str
    sentiment: str
    rationale: str


@dataclass
class PostSentiment:
    index: int
    sentiment: str


class LLMTopicSentimentLabeler:
    def __init__(self, model: str | None = None) -> None:
        self.provider = self._detect_provider()
        self.model = model or self._default_model(self.provider)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def analyze_cluster(self, cluster_id: int, texts: list[str]) -> ClusterAnalysis:
        """Analyze one cluster with structured JSON output from an LLM.

        Falls back to a deterministic heuristic when required API key is unavailable.
        """
        if not texts:
            return ClusterAnalysis(cluster_id, "empty", "neutral", "No texts in cluster")

        if self.provider == "gemini" and self.gemini_api_key:
            parsed = self._call_with_retry(lambda: self._analyze_with_gemini(texts))
            return self._build_cluster_analysis(cluster_id, parsed)

        if self.provider == "openai" and self.openai_api_key:
            parsed = self._call_with_retry(lambda: self._analyze_with_openai(texts))
            return self._build_cluster_analysis(cluster_id, parsed)

        missing_key = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
        return self._heuristic_analysis(cluster_id, texts, missing_key=missing_key)

    def analyze_posts_batch(self, texts: list[str], batch_size: int = 50) -> list[PostSentiment]:
        """Analyze sentiment for individual posts in batches.

        Improvement #4: Per-post sentiment analysis via LLM batch calls.
        Sends up to batch_size posts at once for individual sentiment labels.

        Falls back to heuristic when no API key is available.
        """
        if not texts:
            return []

        has_api = (
            (self.provider == "gemini" and self.gemini_api_key)
            or (self.provider == "openai" and self.openai_api_key)
        )

        all_sentiments: list[PostSentiment] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_indices = list(range(batch_start, batch_start + len(batch)))

            if has_api:
                try:
                    results = self._call_with_retry(
                        lambda b=batch: self._batch_sentiment_llm(b)
                    )
                    for result in results:
                        idx = result.get("index", 0)
                        sentiment = str(result.get("sentiment", "neutral")).lower()
                        if sentiment not in SENTIMENT_VALUES:
                            sentiment = "neutral"
                        # Map batch-local index to global index
                        global_idx = batch_start + (idx - 1) if idx >= 1 else batch_start
                        all_sentiments.append(PostSentiment(index=global_idx, sentiment=sentiment))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Batch sentiment LLM call failed, using heuristic: %s", exc)
                    for i, text in zip(batch_indices, batch):
                        sentiment = self._heuristic_sentiment(text)
                        all_sentiments.append(PostSentiment(index=i, sentiment=sentiment))
            else:
                for i, text in zip(batch_indices, batch):
                    sentiment = self._heuristic_sentiment(text)
                    all_sentiments.append(PostSentiment(index=i, sentiment=sentiment))

        return all_sentiments

    def _batch_sentiment_llm(self, texts: list[str]) -> list[dict[str, Any]]:
        """Call LLM for per-post sentiment on a batch of texts."""
        numbered_posts = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))

        if self.provider == "gemini" and self.gemini_api_key:
            return self._batch_sentiment_gemini(numbered_posts)
        elif self.provider == "openai" and self.openai_api_key:
            return self._batch_sentiment_openai(numbered_posts)
        return []

    def _batch_sentiment_openai(self, numbered_posts: str) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": POST_SENTIMENT_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": numbered_posts}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "post_sentiments",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "sentiment": {
                                            "type": "string",
                                            "enum": ["positive", "negative", "neutral"],
                                        },
                                    },
                                    "required": ["index", "sentiment"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["sentiments"],
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
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))

        parsed = self._extract_openai_structured_json(body)
        return parsed.get("sentiments", [])

    def _batch_sentiment_gemini(self, numbered_posts: str) -> list[dict[str, Any]]:
        prompt = POST_SENTIMENT_SYSTEM_PROMPT + "\n\n" + numbered_posts

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "sentiments": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "index": {"type": "INTEGER"},
                                    "sentiment": {
                                        "type": "STRING",
                                        "enum": ["positive", "negative", "neutral"],
                                    },
                                },
                                "required": ["index", "sentiment"],
                            },
                        }
                    },
                    "required": ["sentiments"],
                },
            },
        }

        params = urlencode({"key": self.gemini_api_key})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?{params}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))

        parsed = self._extract_gemini_structured_json(body)
        return parsed.get("sentiments", [])

    def _analyze_with_openai(self, texts: list[str]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": CLUSTER_SYSTEM_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Cluster posts:\n" + "\n".join(f"- {text}" for text in texts[:20]),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "cluster_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic_label": {"type": "string"},
                            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["topic_label", "sentiment", "rationale"],
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
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            method="POST",
        )

        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        return self._extract_openai_structured_json(body)

    def _analyze_with_gemini(self, texts: list[str]) -> dict[str, Any]:
        prompt = (
            CLUSTER_SYSTEM_PROMPT + "\n\n"
            "Cluster posts:\n" + "\n".join(f"- {text}" for text in texts[:20])
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "topic_label": {"type": "STRING"},
                        "sentiment": {"type": "STRING", "enum": ["positive", "negative", "neutral"]},
                        "rationale": {"type": "STRING"},
                    },
                    "required": ["topic_label", "sentiment", "rationale"],
                },
            },
        }

        params = urlencode({"key": self.gemini_api_key})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?{params}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        return self._extract_gemini_structured_json(body)

    def _extract_openai_structured_json(self, response_body: dict[str, Any]) -> dict[str, Any]:
        output = response_body.get("output", [])
        for entry in output:
            for content in entry.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text", "{}")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        continue
        return {}

    def _extract_gemini_structured_json(self, response_body: dict[str, Any]) -> dict[str, Any]:
        candidates = response_body.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if not text:
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        return {}

    def _build_cluster_analysis(self, cluster_id: int, parsed: dict[str, Any]) -> ClusterAnalysis:
        sentiment = str(parsed.get("sentiment", "neutral")).lower()
        if sentiment not in SENTIMENT_VALUES:
            sentiment = "neutral"

        return ClusterAnalysis(
            cluster_id=cluster_id,
            topic_label=str(parsed.get("topic_label", "unknown-topic")),
            sentiment=sentiment,
            rationale=str(parsed.get("rationale", "")),
        )

    def _heuristic_sentiment(self, text: str) -> str:
        """Classify a single text's sentiment using keyword matching.

        Uses substring matching to handle Japanese compound words that
        don't split on word boundaries with ``\\w+`` regex.
        """
        text_lower = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"

    def _heuristic_analysis(self, cluster_id: int, texts: list[str], *, missing_key: str) -> ClusterAnalysis:
        joined = " ".join(texts).lower()
        words = re.findall(r"\w+", joined)
        topic = words[0] if words else "general"

        # Use substring matching for Japanese compound word support
        pos = sum(1 for w in POSITIVE_WORDS if w in joined)
        neg = sum(1 for w in NEGATIVE_WORDS if w in joined)

        if pos > neg:
            sentiment = "positive"
        elif neg > pos:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return ClusterAnalysis(
            cluster_id=cluster_id,
            topic_label=topic,
            sentiment=sentiment,
            rationale=f"Heuristic fallback used because {missing_key} is not set.",
        )

    def _call_with_retry(
        self,
        request_fn,
        max_retries: int = 3,
    ) -> Any:
        """Call a function with exponential backoff retry on transient failures.

        Improvement #9: Handles HTTP 429 (rate limit), 500/502/503 (server errors),
        and TimeoutError with up to max_retries attempts.
        """
        for attempt in range(max_retries):
            try:
                return request_fn()
            except HTTPError as exc:
                if exc.code in {429, 500, 502, 503} and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.random()
                    LOGGER.warning(
                        "LLM API HTTP %d, retry %d/%d after %.1fs",
                        exc.code, attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
            except (TimeoutError, OSError) as exc:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.random()
                    LOGGER.warning(
                        "LLM API timeout/error, retry %d/%d after %.1fs: %s",
                        attempt + 1, max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise

        return {}  # Should not reach here, but safe fallback

    def _detect_provider(self) -> str:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if provider in {"openai", "gemini"}:
            return provider

        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "openai"

    def _default_model(self, provider: str) -> str:
        if provider == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
