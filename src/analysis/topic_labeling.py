"""Cluster-level topic labeling with an LLM.

This module intentionally labels topics only. Sentiment is handled per post in
``sentiment_transformer.py`` so cluster sentiment can be aggregated from the
actual posts instead of guessed once by an LLM.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .llm_rate_limit import limited_llm_call

LOGGER = logging.getLogger(__name__)


TOPIC_SYSTEM_PROMPT = (
    "You are a Japanese social media analyst specializing in cluster topic labeling. "
    "The posts have already been grouped by semantic similarity. Your job is ONLY to "
    "identify the specific subtopic that distinguishes this cluster from other "
    "clusters. Do not assign post sentiment or cluster sentiment. If emotional tone "
    "helps explain the topic, mention it only as qualitative context inside "
    "marketing_interpretation. Do not use the original search keyword as the topic "
    "label because it appears across clusters. Return structured JSON with these "
    "keys: topic_label, topic_summary, rationale, marketing_interpretation. "
    "Write Japanese labels and explanations when the posts are Japanese."
)


@dataclass
class TopicAnalysis:
    """Structured result for one cluster's topic label."""

    cluster_id: int
    topic_label: str
    topic_summary: str
    rationale: str
    marketing_interpretation: str = ""


class TopicLabeler:
    """LLM-backed cluster topic labeler.

    The public ``analyze_cluster`` method mirrors the old labeler shape so the
    pipeline can switch over with minimal disruption. ``label_cluster`` accepts a
    loose cluster object or dict for callers that already have cluster metadata.
    """

    def __init__(self, model: str | None = None, provider: str | None = None) -> None:
        self.provider = provider or self._detect_provider()
        self.model = model or self._default_model(self.provider)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def label_cluster(
        self,
        cluster: Any,
        *,
        texts: Sequence[str] | None = None,
        top_keywords: Sequence[tuple[str, int]] | None = None,
        representative_texts: Sequence[str] | None = None,
        search_terms: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TopicAnalysis:
        """Label a cluster object or cluster dict.

        ``cluster`` may be the local ``Cluster`` dataclass, a dict containing
        ``cluster_id`` and optional text fields, or a plain integer cluster id.
        The optional keyword arguments take precedence because the pipeline
        already prepares cleaner representative posts and keywords.
        """
        cluster_id = self._extract_cluster_id(cluster)
        cluster_metadata = dict(metadata or {})

        if isinstance(cluster, Mapping):
            texts = texts or cluster.get("texts") or cluster.get("representative_posts")
            top_keywords = top_keywords or cluster.get("top_keywords")
            representative_texts = representative_texts or cluster.get("representative_posts")
            cluster_metadata.update(cluster.get("metadata") or {})

        return self.analyze_cluster(
            cluster_id,
            list(texts or []),
            top_keywords=self._coerce_keywords(top_keywords),
            representative_texts=list(representative_texts or []) or None,
            search_terms=search_terms,
            metadata=cluster_metadata,
        )

    def analyze_cluster(
        self,
        cluster_id: int,
        texts: Sequence[str],
        *,
        top_keywords: Sequence[tuple[str, int]] | None = None,
        representative_texts: Sequence[str] | None = None,
        search_terms: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TopicAnalysis:
        """Generate a topic label and explanation for one cluster.

        The LLM is not asked for sentiment here. If no API key is configured,
        the method returns a deterministic keyword-based fallback so the rest of
        the pipeline can still run.
        """
        clean_texts = [str(text).strip() for text in texts if str(text).strip()]
        if not clean_texts:
            return TopicAnalysis(
                cluster_id=cluster_id,
                topic_label="empty",
                topic_summary="No usable posts were available for this cluster.",
                rationale="No texts in cluster.",
            )

        sample_texts = list(representative_texts or clean_texts[:20])

        if self.provider == "gemini" and self.gemini_api_key:
            try:
                parsed = self._call_with_retry(
                    lambda: self._analyze_with_gemini(
                        sample_texts,
                        top_keywords=top_keywords,
                        search_terms=search_terms,
                        metadata=metadata,
                    )
                )
                return self._build_topic_analysis(cluster_id, parsed)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Topic labeling LLM call failed, using heuristic: %s", exc)
                return self._heuristic_topic_analysis(
                    cluster_id,
                    clean_texts,
                    top_keywords=top_keywords,
                    missing_key=f"Gemini LLM call failed ({exc.__class__.__name__})",
                )

        if self.provider == "openai" and self.openai_api_key:
            try:
                parsed = self._call_with_retry(
                    lambda: self._analyze_with_openai(
                        sample_texts,
                        top_keywords=top_keywords,
                        search_terms=search_terms,
                        metadata=metadata,
                    )
                )
                return self._build_topic_analysis(cluster_id, parsed)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Topic labeling LLM call failed, using heuristic: %s", exc)
                return self._heuristic_topic_analysis(
                    cluster_id,
                    clean_texts,
                    top_keywords=top_keywords,
                    missing_key=f"OpenAI LLM call failed ({exc.__class__.__name__})",
                )

        missing_key = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
        return self._heuristic_topic_analysis(
            cluster_id,
            clean_texts,
            top_keywords=top_keywords,
            missing_key=missing_key,
        )

    def _analyze_with_openai(
        self,
        texts: Sequence[str],
        *,
        top_keywords: Sequence[tuple[str, int]] | None = None,
        search_terms: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_content = self._build_topic_prompt(
            texts,
            top_keywords=top_keywords,
            search_terms=search_terms,
            metadata=metadata,
        )
        payload = {
            "model": self.model,
            "temperature": _env_float("LLM_TEMPERATURE", 0.0),
            "top_p": _env_float("LLM_TOP_P", 1.0),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": TOPIC_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_content}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "cluster_topic_label",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic_label": {"type": "string"},
                            "topic_summary": {"type": "string"},
                            "rationale": {"type": "string"},
                            "marketing_interpretation": {"type": "string"},
                        },
                        "required": [
                            "topic_label",
                            "topic_summary",
                            "rationale",
                            "marketing_interpretation",
                        ],
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

        def send_request() -> dict[str, Any]:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "topic labeling",
            TOPIC_SYSTEM_PROMPT + "\n\n" + user_content,
            send_request,
        )

        return self._extract_openai_structured_json(body)

    def _analyze_with_gemini(
        self,
        texts: Sequence[str],
        *,
        top_keywords: Sequence[tuple[str, int]] | None = None,
        search_terms: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_content = self._build_topic_prompt(
            texts,
            top_keywords=top_keywords,
            search_terms=search_terms,
            metadata=metadata,
        )
        prompt = TOPIC_SYSTEM_PROMPT + "\n\n" + user_content

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": _env_float("LLM_TEMPERATURE", 0.0),
                "topP": _env_float("LLM_TOP_P", 1.0),
                "candidateCount": 1,
                "maxOutputTokens": _env_int("LLM_TOPIC_MAX_OUTPUT_TOKENS", 1024),
                "thinkingConfig": {
                    "thinkingBudget": _env_int("GEMINI_THINKING_BUDGET", 0),
                },
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "topic_label": {"type": "STRING"},
                        "topic_summary": {"type": "STRING"},
                        "rationale": {"type": "STRING"},
                        "marketing_interpretation": {"type": "STRING"},
                    },
                    "required": [
                        "topic_label",
                        "topic_summary",
                        "rationale",
                    ],
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

        def send_request() -> dict[str, Any]:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "topic labeling",
            prompt,
            send_request,
        )

        return self._extract_gemini_structured_json(body)

    def _build_topic_prompt(
        self,
        texts: Sequence[str],
        *,
        top_keywords: Sequence[tuple[str, int]] | None = None,
        search_terms: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Build the cluster topic prompt.

        The prompt gives the LLM enough context to name the subtopic, while
        explicitly reminding it not to do sentiment analysis.
        """
        parts: list[str] = [
            "Task: identify the distinctive subtopic for this one cluster.",
            "Do not classify sentiment. Return JSON only.",
        ]

        if search_terms:
            terms_str = ", ".join(sorted(search_terms))
            parts.extend(
                [
                    "",
                    "Original search/query terms to avoid as the label:",
                    terms_str,
                ]
            )

        if top_keywords:
            kw_str = ", ".join(f"{kw} ({count})" for kw, count in list(top_keywords)[:15])
            parts.extend(["", f"Top keywords: {kw_str}"])

        if metadata:
            safe_metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            if safe_metadata:
                parts.extend(["", f"Cluster metadata: {json.dumps(safe_metadata, ensure_ascii=False)}"])

        parts.extend(["", f"Cluster sample size in prompt: {len(texts)}", "Representative posts:"])
        for text in list(texts)[:15]:
            parts.append(f"- {text}")

        return "\n".join(parts)

    def _build_topic_analysis(self, cluster_id: int, parsed: Mapping[str, Any]) -> TopicAnalysis:
        label = str(parsed.get("topic_label") or "unknown-topic").strip()
        return TopicAnalysis(
            cluster_id=cluster_id,
            topic_label=label or "unknown-topic",
            topic_summary=str(parsed.get("topic_summary") or "").strip(),
            rationale=str(parsed.get("rationale") or "").strip(),
            marketing_interpretation=str(parsed.get("marketing_interpretation") or "").strip(),
        )

    def _heuristic_topic_analysis(
        self,
        cluster_id: int,
        texts: Sequence[str],
        *,
        top_keywords: Sequence[tuple[str, int]] | None = None,
        missing_key: str,
    ) -> TopicAnalysis:
        if top_keywords:
            topic = str(top_keywords[0][0])
        else:
            joined = " ".join(texts).lower()
            words = re.findall(r"\w+", joined)
            topic = words[0] if words else "general"

        return TopicAnalysis(
            cluster_id=cluster_id,
            topic_label=topic,
            topic_summary="Heuristic topic label generated without an LLM.",
            rationale=f"Heuristic fallback used because {missing_key} is not set.",
            marketing_interpretation="Set an LLM API key for richer marketing interpretation.",
        )

    def _extract_openai_structured_json(self, response_body: Mapping[str, Any]) -> dict[str, Any]:
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

    def _extract_gemini_structured_json(self, response_body: Mapping[str, Any]) -> dict[str, Any]:
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

    def _call_with_retry(self, request_fn, max_retries: int | None = None) -> Any:
        """Call an API function with exponential backoff for transient errors."""
        max_retries = max_retries or _env_int("TOPIC_LABELING_MAX_RETRIES", 3)
        rate_limit_cooldown = _env_float("TOPIC_LABELING_429_COOLDOWN_SECONDS", 60.0)
        retry_base = _env_float("TOPIC_LABELING_RETRY_BASE_SECONDS", 2.0)

        for attempt in range(max_retries):
            try:
                return request_fn()
            except HTTPError as exc:
                if exc.code in {429, 500, 502, 503} and attempt < max_retries - 1:
                    if exc.code == 429:
                        wait = rate_limit_cooldown + random.random()
                    else:
                        wait = (retry_base ** attempt) + random.random()
                    LOGGER.warning(
                        "Topic labeling API HTTP %d, retry %d/%d after %.1fs",
                        exc.code,
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise
            except (TimeoutError, OSError) as exc:
                if attempt < max_retries - 1:
                    wait = (retry_base ** attempt) + random.random()
                    LOGGER.warning(
                        "Topic labeling API timeout/error, retry %d/%d after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                else:
                    raise
        return {}

    def _detect_provider(self) -> str:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if provider in {"openai", "gemini"}:
            return provider
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "openai"

    def _default_model(self, provider: str) -> str:
        explicit_topic_model = os.getenv("TOPIC_LABELING_MODEL")
        if explicit_topic_model:
            return explicit_topic_model
        if provider == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def _extract_cluster_id(self, cluster: Any) -> int:
        if isinstance(cluster, int):
            return cluster
        if isinstance(cluster, Mapping):
            return int(cluster.get("cluster_id", 0))
        return int(getattr(cluster, "cluster_id", 0))

    def _coerce_keywords(self, raw_keywords: Any) -> list[tuple[str, int]] | None:
        if not raw_keywords:
            return None

        coerced: list[tuple[str, int]] = []
        for item in raw_keywords:
            if isinstance(item, Mapping):
                coerced.append((str(item.get("keyword", "")), int(item.get("count", 0))))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                coerced.append((str(item[0]), int(item[1])))
        return coerced or None


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
