"""Build Shiny-friendly structured analysis output."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .sentiment_transformer import normalize_sentiment_label

SENTIMENT_VALUES = ("positive", "neutral", "negative")
TWEET_ID_PATTERNS = [
    re.compile(r"/tweet/(\d+)"),
    re.compile(r"/status/(\d+)"),
]


def build_structured_output(
    *,
    posts: Sequence[dict],
    input_path: Path,
    clusters_payload: Sequence[dict],
    analysis_payload: Mapping[str, Any],
    sentiment_results_by_index: Mapping[int, Mapping[str, Any]],
    coordinates_by_index: Mapping[int, tuple[float, float]],
    representative_indices_by_cluster: Mapping[int, set[int]],
) -> dict[str, Any]:
    """Return one JSON payload tailored for downstream Shiny dashboards."""
    analysis_clusters = {
        int(cluster.get("cluster_id", -1)): cluster
        for cluster in analysis_payload.get("clusters", [])
    }
    post_cluster = _build_post_cluster_lookup(clusters_payload)
    post_records = []

    for index, post in enumerate(posts):
        cluster_id = post_cluster.get(index)
        cluster_lookup_id = cluster_id if cluster_id is not None else -1
        analysis_cluster = analysis_clusters.get(cluster_lookup_id, {})
        topics = _post_topics(post, analysis_cluster)
        primary_topic = topics[0] if topics else {}
        sentiment_result = sentiment_results_by_index.get(index, {})
        transformer_sentiment = _transformer_sentiment(post, sentiment_result)
        sentiment_llm = normalize_sentiment_label(post.get("Sentiment_LLM") or post.get("post_sentiment"))
        coords = coordinates_by_index.get(index)

        post_records.append(
            {
                "post_id": _extract_post_id(post, index),
                "source_row_index": index,
                "username": _clean_username(post.get("username")),
                "text": post.get("content") or "",
                "cleaned_text": post.get("cleaned_content") or "",
                "created_at": post.get("timestamp") or post.get("created_at"),
                "source_keyword": _source_keyword(post),
                "cluster_id": cluster_id,
                "topic_id": primary_topic.get("topic_id"),
                "topic_label": primary_topic.get("topic_label"),
                "topic_description": primary_topic.get("topic_description"),
                "topics": topics,
                "raw_topics_llm": post.get("raw_topics_llm") or [],
                "Sentiment_Transformer": transformer_sentiment,
                "sentiment_transformer": transformer_sentiment,
                "Sentiment_LLM": sentiment_llm,
                "sentiment_llm": sentiment_llm,
                "sentiment": sentiment_llm,
                "sentiment_rationale": None,
                "sentiment_score": _sentiment_score(sentiment_result, transformer_sentiment),
                "engagement_like_count": None,
                "engagement_repost_count": None,
                "engagement_reply_count": None,
                "engagement_total": None,
                "is_representative_post": index in representative_indices_by_cluster.get(cluster_lookup_id, set()),
                "url": post.get("url"),
                "language": post.get("language"),
                "fetched_at": post.get("fetched_at"),
                "processed_at": post.get("processed_at"),
                "umap_x": coords[0] if coords else None,
                "umap_y": coords[1] if coords else None,
            }
        )

    cluster_records = [
        _build_cluster_record(cluster, analysis_clusters.get(int(cluster.get("cluster_id", -1)), {}), post_records)
        for cluster in clusters_payload
    ]

    return {
        "metadata": _build_metadata(posts, input_path, analysis_payload),
        "posts": post_records,
        "clusters": cluster_records,
        "daily_topic_metrics": _build_daily_topic_metrics(post_records),
        "report_summary": _build_report_summary(analysis_payload, cluster_records, post_records),
    }


def _build_post_cluster_lookup(clusters_payload: Sequence[dict]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for cluster in clusters_payload:
        cluster_id = int(cluster.get("cluster_id", -1))
        for index in cluster.get("post_indices", []):
            lookup[int(index)] = cluster_id
    return lookup


def _build_cluster_record(cluster: Mapping[str, Any], analysis: Mapping[str, Any], posts: Sequence[dict]) -> dict:
    cluster_id = int(cluster.get("cluster_id", -1))
    topics = list(cluster.get("topics") or analysis.get("topics") or [])
    primary_topic = topics[0] if topics else {
        "topic_id": cluster.get("topic_id") or analysis.get("topic_id") or _topic_id(cluster_id),
        "topic_label": cluster.get("topic_label") or analysis.get("topic_label"),
        "topic_description": cluster.get("topic_description") or analysis.get("topic_description"),
    }
    sentiment_counts = cluster.get("sentiment_llm_counts") or cluster.get("sentiment_counts") or {}
    cluster_size = int(cluster.get("size", 0))
    total = cluster_size or sum(int(sentiment_counts.get(value, 0)) for value in SENTIMENT_VALUES) or 1

    positive_count = int(sentiment_counts.get("positive", 0))
    neutral_count = int(sentiment_counts.get("neutral", 0))
    negative_count = int(sentiment_counts.get("negative", 0))
    cluster_posts = [post for post in posts if post.get("cluster_id") == cluster_id]
    sentiment_scores = [
        float(post["sentiment_score"])
        for post in cluster_posts
        if isinstance(post.get("sentiment_score"), (int, float))
    ]
    representative_posts = [post for post in cluster_posts if post.get("is_representative_post")]
    x_values = [post["umap_x"] for post in cluster_posts if isinstance(post.get("umap_x"), (int, float))]
    y_values = [post["umap_y"] for post in cluster_posts if isinstance(post.get("umap_y"), (int, float))]

    return {
        "cluster_id": cluster_id,
        "topic_id": primary_topic.get("topic_id"),
        "topic_label": primary_topic.get("topic_label"),
        "topic_description": primary_topic.get("topic_description"),
        "topics": topics,
        "cluster_size": cluster_size,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "positive_ratio": round(positive_count / total, 3),
        "neutral_ratio": round(neutral_count / total, 3),
        "negative_ratio": round(negative_count / total, 3),
        "avg_sentiment_score": round(sum(sentiment_scores) / len(sentiment_scores), 4) if sentiment_scores else None,
        "sentiment_llm_counts": _ordered_counts(sentiment_counts),
        "sentiment_transformer_counts": _ordered_counts(cluster.get("sentiment_transformer_counts") or {}),
        "total_engagement": None,
        "avg_engagement": None,
        "top_keywords": [
            str(item.get("keyword"))
            for item in cluster.get("top_keywords", [])
            if item.get("keyword")
        ],
        "representative_post_ids": [post.get("post_id") for post in representative_posts],
        "representative_post_texts": [post.get("cleaned_text") or post.get("text") for post in representative_posts],
        "marketing_interpretation": cluster.get("marketing_interpretation") or analysis.get("marketing_interpretation"),
        "risk_level": cluster.get("risk_level") or analysis.get("risk_level") or _risk_level(negative_count / total),
        "opportunity_level": (
            cluster.get("opportunity_level")
            or analysis.get("opportunity_level")
            or _opportunity_level(positive_count / total, cluster_size)
        ),
        "recommended_actions": cluster.get("recommended_actions") or analysis.get("recommended_actions") or [],
        "umap_centroid_x": round(sum(x_values) / len(x_values), 6) if x_values else None,
        "umap_centroid_y": round(sum(y_values) / len(y_values), 6) if y_values else None,
    }


def _build_metadata(posts: Sequence[dict], input_path: Path, analysis_payload: Mapping[str, Any]) -> dict:
    generated_at = str(analysis_payload.get("generated_at") or datetime.now(timezone.utc).isoformat())
    return {
        "project_name": "social_listening_topic_analysis",
        "source_keyword": _joined_keywords(posts),
        "analysis_date": _date_part(generated_at),
        "language": _dominant_language(posts),
        "notes": "",
        "input_file": input_path.name,
        "input_path": str(input_path),
        "input_post_count": len(posts),
        "input_fetched_at": _first_nonempty(posts, "fetched_at"),
        "sentiment_transformer_model": analysis_payload.get("sentiment_transformer_model"),
        "llm_provider": analysis_payload.get("llm_provider"),
        "llm_model": analysis_payload.get("llm_model"),
        "llm_post_batch_size": analysis_payload.get("llm_post_batch_size"),
        "coordinate_fields": ["umap_x", "umap_y"],
        "topic_consolidation": analysis_payload.get("topic_consolidation"),
    }


def _build_daily_topic_metrics(posts: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    labels: dict[tuple[str, str], str | None] = {}

    for post in posts:
        day = _post_day(post)
        topics = post.get("topics") or [
            {
                "topic_id": post.get("topic_id") or "t000",
                "topic_label": post.get("topic_label"),
            }
        ]
        for topic in topics:
            topic_id = str(topic.get("topic_id") or "t000")
            key = (day, topic_id)
            grouped[key]["post_count"] += 1
            grouped[key][normalize_sentiment_label(post.get("Sentiment_LLM") or post.get("sentiment"))] += 1
            labels[key] = topic.get("topic_label") or post.get("topic_label")

    rows: list[dict] = []
    for (day, topic_id), counts in sorted(grouped.items()):
        total = counts["post_count"] or 1
        rows.append(
            {
                "date": day,
                "topic_id": topic_id,
                "topic_label": labels.get((day, topic_id)),
                "post_count": int(counts["post_count"]),
                "positive_count": int(counts["positive"]),
                "neutral_count": int(counts["neutral"]),
                "negative_count": int(counts["negative"]),
                "positive_ratio": round(counts["positive"] / total, 3),
                "neutral_ratio": round(counts["neutral"] / total, 3),
                "negative_ratio": round(counts["negative"] / total, 3),
            }
        )
    return rows


def _build_report_summary(
    analysis_payload: Mapping[str, Any],
    clusters: Sequence[dict],
    posts: Sequence[dict],
) -> dict:
    sentiment_llm_counts = Counter(normalize_sentiment_label(post.get("Sentiment_LLM")) for post in posts)
    sentiment_transformer_counts = Counter(
        normalize_sentiment_label(post.get("Sentiment_Transformer")) for post in posts
    )
    diagnostics = analysis_payload.get("diagnostics") or {}
    marketing_report = analysis_payload.get("marketing_report") or {}
    return {
        "total_posts": len(posts),
        "cluster_count": len(clusters),
        "positive_count": int(sentiment_llm_counts["positive"]),
        "neutral_count": int(sentiment_llm_counts["neutral"]),
        "negative_count": int(sentiment_llm_counts["negative"]),
        "sentiment_llm_counts": _ordered_counts(sentiment_llm_counts),
        "sentiment_transformer_counts": _ordered_counts(sentiment_transformer_counts),
        "sentiment_method": analysis_payload.get("sentiment_method"),
        "sentiment_model": analysis_payload.get("sentiment_model"),
        "llm_model": analysis_payload.get("llm_model"),
        "mini_report": marketing_report.get("mini_report"),
        "coordinate_method": diagnostics.get("coordinate_method"),
        "coordinate_fields": ["umap_x", "umap_y"],
        "largest_cluster_ratio": diagnostics.get("largest_cluster_ratio"),
        "noise_count": diagnostics.get("noise_count"),
    }


def _post_topics(post: Mapping[str, Any], analysis_cluster: Mapping[str, Any]) -> list[dict[str, Any]]:
    topics = post.get("topics") or []
    if topics:
        return [dict(topic) for topic in topics if isinstance(topic, Mapping)]
    topic_id = post.get("topic_id") or analysis_cluster.get("topic_id")
    topic_label = post.get("topic_label") or analysis_cluster.get("topic_label")
    topic_description = post.get("topic_description") or analysis_cluster.get("topic_description")
    if not topic_id and not topic_label:
        return []
    return [
        {
            "topic_id": topic_id,
            "topic_label": topic_label,
            "topic_description": topic_description,
        }
    ]


def _transformer_sentiment(post: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    transformer = result.get("transformer_sentiment") or {}
    return normalize_sentiment_label(
        post.get("Sentiment_Transformer")
        or transformer.get("label")
        or result.get("final_sentiment")
    )


def _sentiment_score(result: Mapping[str, Any], transformer_sentiment: str) -> float | None:
    transformer = result.get("transformer_sentiment") or {}
    raw_scores = transformer.get("raw_scores") or {}
    if transformer_sentiment in raw_scores:
        return round(float(raw_scores[transformer_sentiment]), 4)
    confidence = transformer.get("confidence")
    if isinstance(confidence, (int, float)):
        return round(float(confidence), 4)
    return None


def _extract_post_id(post: Mapping[str, Any], index: int) -> str:
    url = str(post.get("url") or "")
    for pattern in TWEET_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    base = f"{url}|{post.get('content', '')}|{index}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _ordered_counts(counts: Mapping[str, Any] | Counter) -> dict[str, int]:
    return {sentiment: int(counts.get(sentiment, 0)) for sentiment in SENTIMENT_VALUES}


def _risk_level(negative_ratio: float) -> str:
    if negative_ratio >= 0.35:
        return "high"
    if negative_ratio >= 0.15:
        return "medium"
    return "low"


def _opportunity_level(positive_ratio: float, cluster_size: int) -> str:
    if positive_ratio >= 0.5 and cluster_size >= 10:
        return "high"
    if positive_ratio >= 0.25:
        return "medium"
    return "low"


def _topic_id(cluster_id: int | None) -> str | None:
    if cluster_id is None or cluster_id < 0:
        return None
    return f"t{cluster_id + 1:03d}"


def _source_keyword(post: Mapping[str, Any]) -> str:
    return str(post.get("keyword") or post.get("query") or "")


def _joined_keywords(posts: Sequence[Mapping[str, Any]]) -> str:
    values = sorted({_source_keyword(post) for post in posts if _source_keyword(post)})
    return ", ".join(values)


def _dominant_language(posts: Sequence[Mapping[str, Any]]) -> str | None:
    languages = [str(post.get("language")) for post in posts if post.get("language")]
    if not languages:
        return None
    return Counter(languages).most_common(1)[0][0]


def _first_nonempty(posts: Iterable[Mapping[str, Any]], field: str) -> Any:
    for post in posts:
        value = post.get(field)
        if value:
            return value
    return None


def _clean_username(username: Any) -> str | None:
    if username is None:
        return None
    text = str(username).strip()
    return text.lstrip("@") or None


def _post_day(post: Mapping[str, Any]) -> str:
    for field in ("created_at", "fetched_at", "processed_at"):
        value = post.get(field)
        if value:
            return _date_part(str(value))
    return datetime.now(timezone.utc).date().isoformat()


def _date_part(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10]
