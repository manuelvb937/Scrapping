"""LLM-based topic and sentiment labeling with structured JSON output."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SENTIMENT_VALUES = {"positive", "negative", "neutral"}


@dataclass
class ClusterAnalysis:
    cluster_id: int
    topic_label: str
    sentiment: str
    rationale: str


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
            parsed = self._analyze_with_gemini(texts)
            return self._build_cluster_analysis(cluster_id, parsed)

        if self.provider == "openai" and self.openai_api_key:
            parsed = self._analyze_with_openai(texts)
            return self._build_cluster_analysis(cluster_id, parsed)

        missing_key = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
        return self._heuristic_analysis(cluster_id, texts, missing_key=missing_key)

    def _analyze_with_openai(self, texts: list[str]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You label social-media clusters. Return strict JSON with keys: "
                                "topic_label, sentiment, rationale. sentiment must be one of "
                                "positive, negative, neutral."
                            ),
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
            "You label social-media clusters. Return JSON with keys: topic_label, sentiment, rationale. "
            "sentiment must be one of positive, negative, neutral.\n\n"
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

    def _heuristic_analysis(self, cluster_id: int, texts: list[str], *, missing_key: str) -> ClusterAnalysis:
        joined = " ".join(texts).lower()
        words = re.findall(r"\w+", joined)
        topic = words[0] if words else "general"

        positive_words = {"good", "great", "love", "excellent", "happy", "success", "最高", "尊い", "神"}
        negative_words = {"bad", "hate", "terrible", "awful", "fail", "problem", "最悪", "微妙", "つまらない"}

        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)

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
