"""LLM batch annotation for post engagement sentiment and raw post topics."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .llm_rate_limit import limited_llm_call
from .sentiment_transformer import normalize_sentiment_label

LOGGER = logging.getLogger(__name__)
SENTIMENT_VALUES = ("positive", "neutral", "negative")

POST_BATCH_SYSTEM_PROMPT = (
    "You are a Japanese social listening analyst for BL drama and streaming "
    "platform marketing. Analyze each post independently.\n\n"
    "Sentiment_LLM definition:\n"
    "- positive: the user is engaged, favorable, interested, watching, "
    "rewatching, recommending, anticipating, purchasing, or expressing affection "
    "toward the drama, cast, fandom, manga, platform, or viewing experience.\n"
    "- negative: the user clearly criticizes or rejects the drama, platform, "
    "cast, story, availability, price, or viewing experience.\n"
    "- neutral: informational, unclear, mixed without clear favorability, "
    "unrelated, or only mentions the title/platform without evaluation.\n\n"
    "For each post return source_row_index, sentiment_llm, and up to 3 topics. "
    "Each topic must include topic_label and topic_description. Keep labels as "
    "short Japanese phrases when the post is Japanese. Keep descriptions concise. "
    "Do not output rationale. Do not use the search keyword/title alone as a "
    "topic. Return JSON only."
)

MARKETING_SYSTEM_PROMPT = (
    "You are a concise Japanese BL drama and streaming-platform marketing "
    "strategist. Use only the cluster summaries provided. Sentiment_LLM means "
    "engagement/favorability: positive is engaged/favorable, negative is clear "
    "criticism or rejection, neutral is informational or unclear. For each "
    "cluster, write a short marketing_interpretation, risk_level, "
    "opportunity_level, and up to 3 recommended_actions. Then write one concise "
    "mini_report for the whole run. Return JSON only."
)

TOPIC_CONSOLIDATION_SYSTEM_PROMPT = (
    "You are consolidating social-listening topic labels inside one semantic "
    "cluster. The input topics are preliminary labels created from individual "
    "posts, so they may be too granular or duplicated. Merge synonyms and "
    "near-duplicates into a small taxonomy that is useful for marketing "
    "analysis. Keep labels specific: avoid using only the drama/search title, "
    "and avoid generic labels like '感想', '話題', or '投稿'. Return at most the "
    "requested number of topics. For each final topic, include the source_topic_ids "
    "that should be merged into it. Return JSON only."
)


@dataclass(frozen=True)
class RawTopic:
    topic_label: str
    topic_description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "topic_label": self.topic_label,
            "topic_description": self.topic_description,
        }


@dataclass(frozen=True)
class PostLLMAnnotation:
    source_row_index: int
    sentiment_llm: str
    topics: list[RawTopic]
    used_llm: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_row_index": self.source_row_index,
            "Sentiment_LLM": self.sentiment_llm,
            "sentiment_llm": self.sentiment_llm,
            "raw_topics_llm": [topic.to_dict() for topic in self.topics],
            "used_llm": self.used_llm,
        }


class PostBatchAnnotator:
    """Annotate clustered posts in rate-limited LLM batches."""

    def __init__(self, model: str | None = None, provider: str | None = None) -> None:
        self.provider = provider or _detect_provider()
        self.model = model or _default_model(self.provider)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.batch_size = max(1, _env_int("POST_TOPIC_LLM_BATCH_SIZE", _env_int("LLM_POST_BATCH_SIZE", 15)))

    @property
    def has_api_key(self) -> bool:
        return (
            (self.provider == "gemini" and bool(self.gemini_api_key))
            or (self.provider == "openai" and bool(self.openai_api_key))
        )

    def annotate_cluster_posts(
        self,
        *,
        cluster_id: int,
        posts: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None = None,
        search_terms: set[str] | None = None,
        fallback_sentiments_by_index: Mapping[int, str] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Return index-keyed LLM annotations for one cluster."""
        clean_posts = [
            {
                "source_row_index": int(post.get("source_row_index", post.get("index", 0))),
                "text": str(post.get("text") or ""),
            }
            for post in posts
            if str(post.get("text") or "").strip()
        ]
        if not clean_posts:
            return {}

        if not self.has_api_key:
            return {
                int(post["source_row_index"]): self._fallback_annotation(
                    int(post["source_row_index"]),
                    top_keywords=top_keywords,
                    fallback_sentiments_by_index=fallback_sentiments_by_index,
                ).to_dict()
                for post in clean_posts
            }

        annotations: dict[int, dict[str, Any]] = {}
        for start in range(0, len(clean_posts), self.batch_size):
            batch = clean_posts[start : start + self.batch_size]
            try:
                parsed = self._call_with_retry(
                    lambda batch=batch: self._annotate_batch(
                        cluster_id=cluster_id,
                        posts=batch,
                        top_keywords=top_keywords,
                        search_terms=search_terms,
                    )
                )
                batch_annotations = self._normalize_batch_annotations(
                    batch,
                    parsed,
                    fallback_sentiments_by_index=fallback_sentiments_by_index,
                    top_keywords=top_keywords,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Post LLM annotation failed for cluster %s batch %s-%s, using fallback: %s",
                    cluster_id,
                    start,
                    start + len(batch),
                    exc,
                )
                batch_annotations = [
                    self._fallback_annotation(
                        int(post["source_row_index"]),
                        top_keywords=top_keywords,
                        fallback_sentiments_by_index=fallback_sentiments_by_index,
                    )
                    for post in batch
                ]

            for annotation in batch_annotations:
                annotations[annotation.source_row_index] = annotation.to_dict()

        return annotations

    def _annotate_batch(
        self,
        *,
        cluster_id: int,
        posts: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None,
        search_terms: set[str] | None,
    ) -> dict[str, Any]:
        if self.provider == "gemini":
            return self._annotate_batch_with_gemini(
                cluster_id=cluster_id,
                posts=posts,
                top_keywords=top_keywords,
                search_terms=search_terms,
            )
        return self._annotate_batch_with_openai(
            cluster_id=cluster_id,
            posts=posts,
            top_keywords=top_keywords,
            search_terms=search_terms,
        )

    def _annotate_batch_with_openai(
        self,
        *,
        cluster_id: int,
        posts: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None,
        search_terms: set[str] | None,
    ) -> dict[str, Any]:
        user_content = _build_post_batch_prompt(
            cluster_id=cluster_id,
            posts=posts,
            top_keywords=top_keywords,
            search_terms=search_terms,
        )
        payload = {
            "model": self.model,
            "temperature": _env_float("LLM_TEMPERATURE", 0.0),
            "top_p": _env_float("LLM_TOP_P", 1.0),
            "max_output_tokens": _env_int("LLM_POST_TOPIC_MAX_OUTPUT_TOKENS", 4096),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": POST_BATCH_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_content}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "post_engagement_topics",
                    "schema": _openai_post_schema(),
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
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "post LLM sentiment/topics batch",
            POST_BATCH_SYSTEM_PROMPT + "\n\n" + user_content,
            send_request,
        )
        return _extract_openai_structured_json(body)

    def _annotate_batch_with_gemini(
        self,
        *,
        cluster_id: int,
        posts: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None,
        search_terms: set[str] | None,
    ) -> dict[str, Any]:
        user_content = _build_post_batch_prompt(
            cluster_id=cluster_id,
            posts=posts,
            top_keywords=top_keywords,
            search_terms=search_terms,
        )
        prompt = POST_BATCH_SYSTEM_PROMPT + "\n\n" + user_content
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": _env_float("LLM_TEMPERATURE", 0.0),
                "topP": _env_float("LLM_TOP_P", 1.0),
                "candidateCount": 1,
                "maxOutputTokens": _env_int("LLM_POST_TOPIC_MAX_OUTPUT_TOKENS", 4096),
                "thinkingConfig": {
                    "thinkingBudget": _env_int("GEMINI_THINKING_BUDGET", 0),
                },
                "responseSchema": _gemini_post_schema(),
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
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "post LLM sentiment/topics batch",
            prompt,
            send_request,
        )
        return _extract_gemini_structured_json(body)

    def _normalize_batch_annotations(
        self,
        batch_posts: Sequence[Mapping[str, Any]],
        parsed: Mapping[str, Any],
        *,
        fallback_sentiments_by_index: Mapping[int, str] | None,
        top_keywords: Sequence[tuple[str, int | float]] | None,
    ) -> list[PostLLMAnnotation]:
        annotations_by_index: dict[int, PostLLMAnnotation] = {}
        for item in parsed.get("posts", []):
            try:
                source_row_index = int(item.get("source_row_index"))
            except (TypeError, ValueError):
                continue
            sentiment = normalize_sentiment_label(item.get("sentiment_llm"))
            topics = _normalize_topics(item.get("topics", []), top_keywords=top_keywords)
            annotations_by_index[source_row_index] = PostLLMAnnotation(
                source_row_index=source_row_index,
                sentiment_llm=sentiment,
                topics=topics,
                used_llm=True,
            )

        normalized: list[PostLLMAnnotation] = []
        for post in batch_posts:
            source_row_index = int(post["source_row_index"])
            normalized.append(
                annotations_by_index.get(source_row_index)
                or self._fallback_annotation(
                    source_row_index,
                    top_keywords=top_keywords,
                    fallback_sentiments_by_index=fallback_sentiments_by_index,
                )
            )
        return normalized

    def _fallback_annotation(
        self,
        source_row_index: int,
        *,
        top_keywords: Sequence[tuple[str, int | float]] | None,
        fallback_sentiments_by_index: Mapping[int, str] | None,
    ) -> PostLLMAnnotation:
        fallback_sentiment = "neutral"
        if fallback_sentiments_by_index:
            fallback_sentiment = normalize_sentiment_label(fallback_sentiments_by_index.get(source_row_index))
        topics = _fallback_topics(top_keywords)
        return PostLLMAnnotation(
            source_row_index=source_row_index,
            sentiment_llm=fallback_sentiment,
            topics=topics,
            used_llm=False,
        )

    def _call_with_retry(self, callback):
        max_attempts = _env_int("LLM_MAX_RETRIES", 3)
        cooldown = _env_float("TOPIC_LABELING_429_COOLDOWN_SECONDS", 120.0)
        for attempt in range(1, max_attempts + 1):
            try:
                return callback()
            except HTTPError as exc:
                if exc.code != 429 or attempt >= max_attempts:
                    raise
                wait_seconds = cooldown * attempt
                LOGGER.warning("LLM rate limited; waiting %.1fs before retry", wait_seconds)
                time.sleep(wait_seconds)
            except Exception:
                if attempt >= max_attempts:
                    raise
                time.sleep((2**attempt) + random.random())


class TopicConsolidationRefiner:
    """Use an LLM to turn preliminary per-cluster topics into a compact taxonomy."""

    def __init__(self, model: str | None = None, provider: str | None = None) -> None:
        self.provider = provider or _detect_provider()
        self.model = model or _default_model(self.provider)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.max_topics = max(1, _env_int("LLM_TOPIC_CONSOLIDATION_MAX_TOPICS", 6))
        self.max_input_topics = max(
            self.max_topics,
            _env_int("LLM_TOPIC_CONSOLIDATION_MAX_INPUT_TOPICS", 40),
        )

    @property
    def has_api_key(self) -> bool:
        return (
            (self.provider == "gemini" and bool(self.gemini_api_key))
            or (self.provider == "openai" and bool(self.openai_api_key))
        )

    def refine_cluster_topics(
        self,
        *,
        cluster_id: int,
        preliminary_topics: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None = None,
        representative_texts: Sequence[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        clean_topics = [
            dict(topic)
            for topic in preliminary_topics
            if topic.get("topic_id") and topic.get("topic_label")
        ]
        if not clean_topics:
            return [], {}

        if not self.has_api_key or len(clean_topics) == 1:
            return _fallback_topic_refinement(
                cluster_id=cluster_id,
                preliminary_topics=clean_topics,
                max_topics=self.max_topics,
            )

        try:
            if self.provider == "gemini":
                parsed = self._call_with_retry(
                    lambda: self._refine_with_gemini(
                        cluster_id=cluster_id,
                        preliminary_topics=clean_topics,
                        top_keywords=top_keywords,
                        representative_texts=representative_texts,
                    )
                )
            else:
                parsed = self._call_with_retry(
                    lambda: self._refine_with_openai(
                        cluster_id=cluster_id,
                        preliminary_topics=clean_topics,
                        top_keywords=top_keywords,
                        representative_texts=representative_texts,
                    )
                )
            return _normalize_topic_refinement(
                cluster_id=cluster_id,
                preliminary_topics=clean_topics,
                parsed=parsed,
                max_topics=self.max_topics,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM topic consolidation failed for cluster %s: %s", cluster_id, exc)
            return _fallback_topic_refinement(
                cluster_id=cluster_id,
                preliminary_topics=clean_topics,
                max_topics=self.max_topics,
            )

    def _refine_with_openai(
        self,
        *,
        cluster_id: int,
        preliminary_topics: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None,
        representative_texts: Sequence[str] | None,
    ) -> dict[str, Any]:
        user_content = _build_topic_consolidation_prompt(
            cluster_id=cluster_id,
            preliminary_topics=preliminary_topics,
            top_keywords=top_keywords,
            representative_texts=representative_texts,
            max_topics=self.max_topics,
            max_input_topics=self.max_input_topics,
        )
        payload = {
            "model": self.model,
            "temperature": _env_float("LLM_TEMPERATURE", 0.0),
            "top_p": _env_float("LLM_TOP_P", 1.0),
            "max_output_tokens": _env_int("LLM_TOPIC_CONSOLIDATION_MAX_OUTPUT_TOKENS", 2048),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": TOPIC_CONSOLIDATION_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_content}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "topic_taxonomy_refinement",
                    "schema": _openai_topic_consolidation_schema(),
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
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "LLM topic taxonomy refinement",
            TOPIC_CONSOLIDATION_SYSTEM_PROMPT + "\n\n" + user_content,
            send_request,
        )
        return _extract_openai_structured_json(body)

    def _refine_with_gemini(
        self,
        *,
        cluster_id: int,
        preliminary_topics: Sequence[Mapping[str, Any]],
        top_keywords: Sequence[tuple[str, int | float]] | None,
        representative_texts: Sequence[str] | None,
    ) -> dict[str, Any]:
        user_content = _build_topic_consolidation_prompt(
            cluster_id=cluster_id,
            preliminary_topics=preliminary_topics,
            top_keywords=top_keywords,
            representative_texts=representative_texts,
            max_topics=self.max_topics,
            max_input_topics=self.max_input_topics,
        )
        prompt = TOPIC_CONSOLIDATION_SYSTEM_PROMPT + "\n\n" + user_content
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": _env_float("LLM_TEMPERATURE", 0.0),
                "topP": _env_float("LLM_TOP_P", 1.0),
                "candidateCount": 1,
                "maxOutputTokens": _env_int("LLM_TOPIC_CONSOLIDATION_MAX_OUTPUT_TOKENS", 2048),
                "thinkingConfig": {
                    "thinkingBudget": _env_int("GEMINI_THINKING_BUDGET", 0),
                },
                "responseSchema": _gemini_topic_consolidation_schema(),
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
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "LLM topic taxonomy refinement",
            prompt,
            send_request,
        )
        return _extract_gemini_structured_json(body)

    def _call_with_retry(self, callback):
        max_attempts = _env_int("LLM_MAX_RETRIES", 3)
        cooldown = _env_float("TOPIC_LABELING_429_COOLDOWN_SECONDS", 120.0)
        for attempt in range(1, max_attempts + 1):
            try:
                return callback()
            except HTTPError as exc:
                if exc.code != 429 or attempt >= max_attempts:
                    raise
                time.sleep(cooldown * attempt)
            except Exception:
                if attempt >= max_attempts:
                    raise
                time.sleep((2**attempt) + random.random())


class MarketingSummaryGenerator:
    """Generate final cluster marketing interpretations and mini report."""

    def __init__(self, model: str | None = None, provider: str | None = None) -> None:
        self.provider = provider or _detect_provider()
        self.model = model or _default_model(self.provider)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    @property
    def has_api_key(self) -> bool:
        return (
            (self.provider == "gemini" and bool(self.gemini_api_key))
            or (self.provider == "openai" and bool(self.openai_api_key))
        )

    def generate(self, cluster_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not cluster_summaries:
            return {"clusters": [], "mini_report": "No clusters were available."}

        if not self.has_api_key:
            return _fallback_marketing_summary(cluster_summaries)

        try:
            if self.provider == "gemini":
                return self._call_with_retry(lambda: self._generate_with_gemini(cluster_summaries))
            return self._call_with_retry(lambda: self._generate_with_openai(cluster_summaries))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Final marketing LLM call failed, using fallback: %s", exc)
            return _fallback_marketing_summary(cluster_summaries)

    def _generate_with_openai(self, cluster_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        user_content = _build_marketing_prompt(cluster_summaries)
        payload = {
            "model": self.model,
            "temperature": _env_float("LLM_TEMPERATURE", 0.0),
            "top_p": _env_float("LLM_TOP_P", 1.0),
            "max_output_tokens": _env_int("LLM_MARKETING_MAX_OUTPUT_TOKENS", 4096),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": MARKETING_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_content}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "marketing_summary",
                    "schema": _openai_marketing_schema(),
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
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "final marketing summary",
            MARKETING_SYSTEM_PROMPT + "\n\n" + user_content,
            send_request,
        )
        return _normalize_marketing_summary(_extract_openai_structured_json(body), cluster_summaries)

    def _generate_with_gemini(self, cluster_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        user_content = _build_marketing_prompt(cluster_summaries)
        prompt = MARKETING_SYSTEM_PROMPT + "\n\n" + user_content
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": _env_float("LLM_TEMPERATURE", 0.0),
                "topP": _env_float("LLM_TOP_P", 1.0),
                "candidateCount": 1,
                "maxOutputTokens": _env_int("LLM_MARKETING_MAX_OUTPUT_TOKENS", 4096),
                "thinkingConfig": {
                    "thinkingBudget": _env_int("GEMINI_THINKING_BUDGET", 0),
                },
                "responseSchema": _gemini_marketing_schema(),
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
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        body = limited_llm_call(
            self.provider,
            self.model,
            "final marketing summary",
            prompt,
            send_request,
        )
        return _normalize_marketing_summary(_extract_gemini_structured_json(body), cluster_summaries)

    def _call_with_retry(self, callback):
        max_attempts = _env_int("LLM_MAX_RETRIES", 3)
        cooldown = _env_float("TOPIC_LABELING_429_COOLDOWN_SECONDS", 120.0)
        for attempt in range(1, max_attempts + 1):
            try:
                return callback()
            except HTTPError as exc:
                if exc.code != 429 or attempt >= max_attempts:
                    raise
                time.sleep(cooldown * attempt)
            except Exception:
                if attempt >= max_attempts:
                    raise
                time.sleep((2**attempt) + random.random())


def _build_post_batch_prompt(
    *,
    cluster_id: int,
    posts: Sequence[Mapping[str, Any]],
    top_keywords: Sequence[tuple[str, int | float]] | None,
    search_terms: set[str] | None,
) -> str:
    payload = {
        "cluster_id": cluster_id,
        "avoid_as_topic_labels": sorted(search_terms or [])[:20],
        "cluster_keywords": [
            {"keyword": str(keyword), "score": float(score)}
            for keyword, score in list(top_keywords or [])[:10]
        ],
        "posts": [
            {
                "source_row_index": int(post["source_row_index"]),
                "text": _truncate(str(post.get("text") or ""), 900),
            }
            for post in posts
        ],
    }
    return (
        "Analyze these posts. Use the JSON source_row_index exactly. "
        "Topic labels should describe the concrete conversation angle, not only "
        "the drama/search title.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _build_topic_consolidation_prompt(
    *,
    cluster_id: int,
    preliminary_topics: Sequence[Mapping[str, Any]],
    top_keywords: Sequence[tuple[str, int | float]] | None,
    representative_texts: Sequence[str] | None,
    max_topics: int,
    max_input_topics: int,
) -> str:
    sorted_topics = sorted(
        preliminary_topics,
        key=lambda topic: int(topic.get("post_count", 0)),
        reverse=True,
    )[:max_input_topics]
    payload = {
        "cluster_id": cluster_id,
        "max_final_topics": max_topics,
        "cluster_keywords": [
            {"keyword": str(keyword), "score": float(score)}
            for keyword, score in list(top_keywords or [])[:10]
        ],
        "representative_texts": [
            _truncate(str(text), 500)
            for text in list(representative_texts or [])[:4]
        ],
        "preliminary_topics": [
            {
                "topic_id": topic.get("topic_id"),
                "topic_label": topic.get("topic_label"),
                "topic_description": topic.get("topic_description"),
                "post_count": topic.get("post_count", 0),
                "sentiment_llm_counts": topic.get("sentiment_llm_counts", {}),
                "aliases": topic.get("aliases", [])[:6],
                "example_texts": topic.get("example_texts", [])[:2],
            }
            for topic in sorted_topics
        ],
    }
    return (
        "Merge the preliminary topics into a compact final taxonomy for this "
        "single cluster. Use source_topic_ids to preserve traceability. If a "
        "topic is rare but important for marketing risk/opportunity, you may keep "
        "it. Otherwise merge rare variants into the nearest broader topic.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _build_marketing_prompt(cluster_summaries: Sequence[Mapping[str, Any]]) -> str:
    trimmed = []
    max_clusters = _env_int("LLM_MARKETING_MAX_CLUSTERS", 60)
    for cluster in list(cluster_summaries)[:max_clusters]:
        trimmed.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "cluster_size": cluster.get("cluster_size"),
                "top_keywords": cluster.get("top_keywords", [])[:8],
                "sentiment_llm_counts": cluster.get("sentiment_llm_counts", {}),
                "topics": [
                    {
                        "topic_id": topic.get("topic_id"),
                        "topic_label": topic.get("topic_label"),
                        "topic_description": topic.get("topic_description"),
                        "post_count": topic.get("post_count"),
                        "sentiment_llm_counts": topic.get("sentiment_llm_counts"),
                        "example_texts": topic.get("example_texts", [])[:2],
                    }
                    for topic in cluster.get("topics", [])[:8]
                ],
                "representative_texts": cluster.get("representative_texts", [])[:3],
            }
        )
    return (
        "Create concise marketing guidance from these cluster summaries. "
        "Keep Japanese outputs short and practical.\n\n"
        + json.dumps({"clusters": trimmed}, ensure_ascii=False)
    )


def _normalize_topics(
    topics: Any,
    *,
    top_keywords: Sequence[tuple[str, int | float]] | None,
) -> list[RawTopic]:
    normalized: list[RawTopic] = []
    if isinstance(topics, Sequence) and not isinstance(topics, (str, bytes)):
        for item in topics:
            if not isinstance(item, Mapping):
                continue
            label = _clean_short_text(item.get("topic_label"), limit=40)
            description = _clean_short_text(item.get("topic_description"), limit=90)
            if not label:
                continue
            normalized.append(RawTopic(topic_label=label, topic_description=description or label))
            if len(normalized) >= 3:
                break
    return normalized or _fallback_topics(top_keywords)


def _fallback_topics(top_keywords: Sequence[tuple[str, int | float]] | None) -> list[RawTopic]:
    keyword = None
    for item in top_keywords or []:
        if not item:
            continue
        keyword = str(item[0]).strip()
        if keyword:
            break
    if not keyword:
        keyword = "general conversation"
    return [RawTopic(topic_label=_truncate(keyword, 40), topic_description=f"Conversation around {keyword}")]


def _normalize_topic_refinement(
    *,
    cluster_id: int,
    preliminary_topics: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, Any],
    max_topics: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    preliminary_by_id = {
        str(topic.get("topic_id")): dict(topic)
        for topic in preliminary_topics
        if topic.get("topic_id")
    }
    assignments: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()

    for item in parsed.get("topics", []):
        if not isinstance(item, Mapping):
            continue
        label = _clean_short_text(item.get("topic_label"), limit=48)
        description = _clean_short_text(item.get("topic_description"), limit=120)
        source_ids = [
            str(topic_id)
            for topic_id in item.get("source_topic_ids", [])
            if str(topic_id) in preliminary_by_id
        ]
        source_ids = [topic_id for topic_id in source_ids if topic_id not in assigned_ids]
        if not label or not source_ids:
            continue
        assignments.append(
            {
                "topic_label": label,
                "topic_description": description or label,
                "source_topic_ids": source_ids,
            }
        )
        assigned_ids.update(source_ids)
        if len(assignments) >= max_topics:
            break

    if not assignments:
        return _fallback_topic_refinement(
            cluster_id=cluster_id,
            preliminary_topics=preliminary_topics,
            max_topics=max_topics,
        )

    for missing_id in preliminary_by_id:
        if missing_id in assigned_ids:
            continue
        best_assignment = max(
            assignments,
            key=lambda assignment: _topic_label_similarity(
                str(preliminary_by_id[missing_id].get("topic_label") or ""),
                str(assignment.get("topic_label") or ""),
            ),
        )
        best_assignment["source_topic_ids"].append(missing_id)

    return _build_refined_topics_from_assignments(
        cluster_id=cluster_id,
        preliminary_by_id=preliminary_by_id,
        assignments=assignments,
        method="llm",
    )


def _fallback_topic_refinement(
    *,
    cluster_id: int,
    preliminary_topics: Sequence[Mapping[str, Any]],
    max_topics: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sorted_topics = sorted(
        [dict(topic) for topic in preliminary_topics if topic.get("topic_id")],
        key=lambda topic: int(topic.get("post_count", 0)),
        reverse=True,
    )
    if not sorted_topics:
        return [], {}

    seed_topics = sorted_topics[:max_topics]
    assignments = [
        {
            "topic_label": str(topic.get("topic_label") or ""),
            "topic_description": str(topic.get("topic_description") or topic.get("topic_label") or ""),
            "source_topic_ids": [str(topic.get("topic_id"))],
        }
        for topic in seed_topics
    ]

    for topic in sorted_topics[max_topics:]:
        best_assignment = max(
            assignments,
            key=lambda assignment: _topic_label_similarity(
                str(topic.get("topic_label") or ""),
                str(assignment.get("topic_label") or ""),
            ),
        )
        best_assignment["source_topic_ids"].append(str(topic.get("topic_id")))

    preliminary_by_id = {
        str(topic.get("topic_id")): dict(topic)
        for topic in sorted_topics
    }
    return _build_refined_topics_from_assignments(
        cluster_id=cluster_id,
        preliminary_by_id=preliminary_by_id,
        assignments=assignments,
        method="fallback",
    )


def _build_refined_topics_from_assignments(
    *,
    cluster_id: int,
    preliminary_by_id: Mapping[str, Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    refined_topics: list[dict[str, Any]] = []
    topic_id_map: dict[str, dict[str, Any]] = {}

    sorted_assignments = sorted(
        assignments,
        key=lambda assignment: _assignment_post_count(assignment, preliminary_by_id),
        reverse=True,
    )
    for position, assignment in enumerate(sorted_assignments, start=1):
        source_ids = [
            str(topic_id)
            for topic_id in assignment.get("source_topic_ids", [])
            if str(topic_id) in preliminary_by_id
        ]
        if not source_ids:
            continue

        source_topics = [preliminary_by_id[topic_id] for topic_id in source_ids]
        source_row_indices = sorted(
            {
                int(index)
                for topic in source_topics
                for index in topic.get("source_row_indices", [])
            }
        )
        counts = _merge_sentiment_counts(source_topics)
        aliases = sorted(
            {
                str(value)
                for topic in source_topics
                for value in [topic.get("topic_label"), *list(topic.get("aliases") or [])]
                if value and str(value) != str(assignment.get("topic_label"))
            }
        )[:12]
        examples = []
        example_indices = []
        for topic in source_topics:
            for index in topic.get("example_source_row_indices", []):
                if index not in example_indices:
                    example_indices.append(index)
            for text in topic.get("example_texts", []):
                if text and text not in examples:
                    examples.append(str(text))

        topic_id = _refined_topic_id(cluster_id, len(refined_topics) + 1)
        refined_topic = {
            "topic_id": topic_id,
            "topic_label": _clean_short_text(assignment.get("topic_label"), limit=48),
            "topic_description": _clean_short_text(
                assignment.get("topic_description") or assignment.get("topic_label"),
                limit=120,
            ),
            "post_count": len(source_row_indices),
            "sentiment_llm_counts": counts,
            "source_row_indices": source_row_indices,
            "source_topic_ids": source_ids,
            "aliases": aliases,
            "example_source_row_indices": example_indices[:3],
            "example_texts": examples[:3],
            "consolidation_method": method,
        }
        refined_topics.append(refined_topic)
        for source_id in source_ids:
            topic_id_map[source_id] = refined_topic

    return refined_topics, topic_id_map


def _assignment_post_count(
    assignment: Mapping[str, Any],
    preliminary_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    source_indices = {
        int(index)
        for topic_id in assignment.get("source_topic_ids", [])
        for index in preliminary_by_id.get(str(topic_id), {}).get("source_row_indices", [])
    }
    return len(source_indices)


def _merge_sentiment_counts(topics: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {sentiment: 0 for sentiment in SENTIMENT_VALUES}
    for topic in topics:
        topic_counts = topic.get("sentiment_llm_counts") or {}
        for sentiment in SENTIMENT_VALUES:
            counts[sentiment] += int(topic_counts.get(sentiment, 0))
    return counts


def _topic_label_similarity(left: str, right: str) -> float:
    left_key = _normalize_label_key(left)
    right_key = _normalize_label_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    if left_key in right_key or right_key in left_key:
        return min(len(left_key), len(right_key)) / max(len(left_key), len(right_key))
    left_chars = set(left_key)
    right_chars = set(right_key)
    return len(left_chars & right_chars) / (len(left_chars | right_chars) or 1)


def _normalize_label_key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _refined_topic_id(cluster_id: int, position: int) -> str:
    base = f"t{max(cluster_id, 0) + 1:03d}"
    if position == 1:
        return base
    return f"{base}_{position:02d}"


def _normalize_marketing_summary(
    parsed: Mapping[str, Any],
    cluster_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    known_ids = {int(cluster.get("cluster_id", -1)) for cluster in cluster_summaries}
    clusters = []
    for item in parsed.get("clusters", []):
        if not isinstance(item, Mapping):
            continue
        try:
            cluster_id = int(item.get("cluster_id"))
        except (TypeError, ValueError):
            continue
        if cluster_id not in known_ids:
            continue
        clusters.append(
            {
                "cluster_id": cluster_id,
                "marketing_interpretation": _clean_short_text(item.get("marketing_interpretation"), limit=180),
                "risk_level": _normalize_level(item.get("risk_level")),
                "opportunity_level": _normalize_level(item.get("opportunity_level")),
                "recommended_actions": _normalize_actions(item.get("recommended_actions")),
            }
        )
    fallback = _fallback_marketing_summary(cluster_summaries)
    by_id = {item["cluster_id"]: item for item in clusters}
    for item in fallback["clusters"]:
        by_id.setdefault(item["cluster_id"], item)
    return {
        "clusters": [by_id[int(cluster.get("cluster_id", -1))] for cluster in cluster_summaries if int(cluster.get("cluster_id", -1)) in by_id],
        "mini_report": _clean_short_text(parsed.get("mini_report"), limit=900) or fallback["mini_report"],
    }


def _fallback_marketing_summary(cluster_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clusters = []
    total_positive = 0
    total_negative = 0
    total_posts = 0
    for cluster in cluster_summaries:
        cluster_id = int(cluster.get("cluster_id", -1))
        size = int(cluster.get("cluster_size", 0))
        counts = cluster.get("sentiment_llm_counts") or {}
        positive = int(counts.get("positive", 0))
        negative = int(counts.get("negative", 0))
        total_positive += positive
        total_negative += negative
        total_posts += size
        positive_ratio = positive / (size or 1)
        negative_ratio = negative / (size or 1)
        top_label = " / ".join(
            str(topic.get("topic_label"))
            for topic in cluster.get("topics", [])[:2]
            if topic.get("topic_label")
        ) or "cluster conversation"
        clusters.append(
            {
                "cluster_id": cluster_id,
                "marketing_interpretation": f"Conversation centers on {top_label}.",
                "risk_level": "high" if negative_ratio >= 0.35 else "medium" if negative_ratio >= 0.15 else "low",
                "opportunity_level": "high" if positive_ratio >= 0.5 else "medium" if positive_ratio >= 0.25 else "low",
                "recommended_actions": [
                    "Review representative posts before campaign use.",
                    "Amplify high-engagement angles with platform-specific copy.",
                ],
            }
        )
    overall_positive = total_positive / (total_posts or 1)
    overall_negative = total_negative / (total_posts or 1)
    mini_report = (
        f"Processed {total_posts} posts. Sentiment_LLM engagement is "
        f"{overall_positive:.1%} positive and {overall_negative:.1%} negative. "
        "Prioritize clusters with high positive ratios and monitor clusters with visible criticism."
    )
    return {"clusters": clusters, "mini_report": mini_report}


def _extract_openai_structured_json(response_body: Mapping[str, Any]) -> dict[str, Any]:
    output = response_body.get("output", [])
    for entry in output:
        for content in entry.get("content", []):
            if content.get("type") == "output_text":
                return json.loads(content.get("text") or "{}")
    if "output_text" in response_body:
        return json.loads(str(response_body.get("output_text") or "{}"))
    return {}


def _extract_gemini_structured_json(response_body: Mapping[str, Any]) -> dict[str, Any]:
    candidates = response_body.get("candidates") or []
    if not candidates:
        return {}
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return {}
    text = parts[0].get("text") or "{}"
    return json.loads(text)


def _openai_post_schema() -> dict[str, Any]:
    topic_schema = {
        "type": "object",
        "properties": {
            "topic_label": {"type": "string"},
            "topic_description": {"type": "string"},
        },
        "required": ["topic_label", "topic_description"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "posts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_row_index": {"type": "integer"},
                        "sentiment_llm": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"],
                        },
                        "topics": {"type": "array", "items": topic_schema},
                    },
                    "required": ["source_row_index", "sentiment_llm", "topics"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["posts"],
        "additionalProperties": False,
    }


def _gemini_post_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "posts": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "source_row_index": {"type": "INTEGER"},
                        "sentiment_llm": {
                            "type": "STRING",
                            "enum": ["positive", "neutral", "negative"],
                        },
                        "topics": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "topic_label": {"type": "STRING"},
                                    "topic_description": {"type": "STRING"},
                                },
                                "required": ["topic_label", "topic_description"],
                            },
                        },
                    },
                    "required": ["source_row_index", "sentiment_llm", "topics"],
                },
            }
        },
        "required": ["posts"],
    }


def _openai_topic_consolidation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic_label": {"type": "string"},
                        "topic_description": {"type": "string"},
                        "source_topic_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["topic_label", "topic_description", "source_topic_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["topics"],
        "additionalProperties": False,
    }


def _gemini_topic_consolidation_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "topics": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "topic_label": {"type": "STRING"},
                        "topic_description": {"type": "STRING"},
                        "source_topic_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["topic_label", "topic_description", "source_topic_ids"],
                },
            }
        },
        "required": ["topics"],
    }


def _openai_marketing_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "integer"},
                        "marketing_interpretation": {"type": "string"},
                        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                        "opportunity_level": {"type": "string", "enum": ["low", "medium", "high"]},
                        "recommended_actions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "cluster_id",
                        "marketing_interpretation",
                        "risk_level",
                        "opportunity_level",
                        "recommended_actions",
                    ],
                    "additionalProperties": False,
                },
            },
            "mini_report": {"type": "string"},
        },
        "required": ["clusters", "mini_report"],
        "additionalProperties": False,
    }


def _gemini_marketing_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "clusters": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "cluster_id": {"type": "INTEGER"},
                        "marketing_interpretation": {"type": "STRING"},
                        "risk_level": {"type": "STRING", "enum": ["low", "medium", "high"]},
                        "opportunity_level": {"type": "STRING", "enum": ["low", "medium", "high"]},
                        "recommended_actions": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": [
                        "cluster_id",
                        "marketing_interpretation",
                        "risk_level",
                        "opportunity_level",
                        "recommended_actions",
                    ],
                },
            },
            "mini_report": {"type": "STRING"},
        },
        "required": ["clusters", "mini_report"],
    }


def _detect_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"openai", "gemini"}:
        return explicit
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "openai"


def _default_model(provider: str) -> str:
    explicit = os.getenv("POST_TOPIC_MODEL") or os.getenv("TOPIC_LABELING_MODEL")
    if explicit:
        return explicit
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _clean_short_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return _truncate(text, limit)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _normalize_actions(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    actions = [_clean_short_text(item, limit=120) for item in value]
    return [item for item in actions if item][:3]


def _normalize_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    return "medium"
