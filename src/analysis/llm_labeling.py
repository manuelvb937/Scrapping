"""LLM-based topic and sentiment labeling with structured JSON output."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

SENTIMENT_VALUES = {"positive", "negative", "neutral"}


@dataclass
class ClusterAnalysis:
    cluster_id: int
    topic_label: str
    sentiment: str
    rationale: str


class LLMTopicSentimentLabeler:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    def analyze_cluster(self, cluster_id: int, texts: list[str]) -> ClusterAnalysis:
        """Analyze one cluster with structured JSON output from an LLM.

        Falls back to a deterministic heuristic when API key is unavailable.
        """
        if not texts:
            return ClusterAnalysis(cluster_id, "empty", "neutral", "No texts in cluster")

        if not self.api_key:
            return self._heuristic_analysis(cluster_id, texts)

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
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))

        parsed = self._extract_structured_json(body)
        sentiment = parsed.get("sentiment", "neutral")
        if sentiment not in SENTIMENT_VALUES:
            sentiment = "neutral"

        return ClusterAnalysis(
            cluster_id=cluster_id,
            topic_label=str(parsed.get("topic_label", "unknown-topic")),
            sentiment=sentiment,
            rationale=str(parsed.get("rationale", "")),
        )

    def _extract_structured_json(self, response_body: dict[str, Any]) -> dict[str, Any]:
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

    def _heuristic_analysis(self, cluster_id: int, texts: list[str]) -> ClusterAnalysis:
        joined = " ".join(texts).lower()
        words = re.findall(r"\w+", joined)
        topic = words[0] if words else "general"

        positive_words = {"good", "great", "love", "excellent", "happy", "success"}
        negative_words = {"bad", "hate", "terrible", "awful", "fail", "problem"}

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
            rationale="Heuristic fallback used because OPENAI_API_KEY is not set.",
        )
