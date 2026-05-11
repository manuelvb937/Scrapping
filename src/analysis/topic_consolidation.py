"""Consolidate raw per-post LLM topics into canonical cluster topics."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .clustering import Cluster
from .sentiment_transformer import normalize_sentiment_label

LOGGER = logging.getLogger(__name__)
SENTIMENT_VALUES = ("positive", "neutral", "negative")


def consolidate_cluster_topics(
    *,
    clusters: Sequence[Cluster],
    post_annotations_by_index: Mapping[int, Mapping[str, Any]],
    post_texts: Sequence[str],
    embedder: Any | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """Consolidate raw post topics inside each cluster.

    The LLM can produce small label variations for the same idea. This function
    merges labels first by normalization, then by high semantic/lexical
    similarity. It returns:
    - cluster_id -> canonical topic records
    - source_row_index -> canonical topic records attached to that post
    """
    cluster_topics_by_id: dict[int, list[dict[str, Any]]] = {}
    post_topics_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    threshold = _env_float("TOPIC_MERGE_SIMILARITY_THRESHOLD", 0.86)

    for cluster in clusters:
        instances = _collect_cluster_instances(
            cluster=cluster,
            post_annotations_by_index=post_annotations_by_index,
            post_texts=post_texts,
        )
        if not instances:
            cluster_topics_by_id[cluster.cluster_id] = []
            continue

        normalized_groups = _group_by_normalized_label(instances)
        merged_groups = _merge_similar_groups(
            normalized_groups,
            embedder=embedder,
            threshold=threshold,
        )
        canonical_topics, raw_group_to_topic = _build_canonical_topics(
            cluster_id=cluster.cluster_id,
            groups=merged_groups,
        )
        cluster_topics_by_id[cluster.cluster_id] = canonical_topics

        for group_index, topic in raw_group_to_topic.items():
            for instance in merged_groups[group_index]["instances"]:
                source_row_index = int(instance["source_row_index"])
                if not any(item["topic_id"] == topic["topic_id"] for item in post_topics_by_index[source_row_index]):
                    post_topics_by_index[source_row_index].append(
                        {
                            "topic_id": topic["topic_id"],
                            "topic_label": topic["topic_label"],
                            "topic_description": topic["topic_description"],
                            "raw_topic_label": instance["label"],
                        }
                    )

    for source_row_index, topics in list(post_topics_by_index.items()):
        post_topics_by_index[source_row_index] = topics[:3]

    return cluster_topics_by_id, dict(post_topics_by_index)


def _collect_cluster_instances(
    *,
    cluster: Cluster,
    post_annotations_by_index: Mapping[int, Mapping[str, Any]],
    post_texts: Sequence[str],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for source_row_index in cluster.indices:
        annotation = post_annotations_by_index.get(source_row_index) or {}
        sentiment = normalize_sentiment_label(annotation.get("Sentiment_LLM") or annotation.get("sentiment_llm"))
        raw_topics = annotation.get("raw_topics_llm") or []
        if not isinstance(raw_topics, Sequence) or isinstance(raw_topics, (str, bytes)):
            raw_topics = []
        for raw_order, raw_topic in enumerate(raw_topics[:3]):
            if not isinstance(raw_topic, Mapping):
                continue
            label = _clean_label(raw_topic.get("topic_label"))
            description = _clean_description(raw_topic.get("topic_description"))
            if not label:
                continue
            instances.append(
                {
                    "source_row_index": int(source_row_index),
                    "label": label,
                    "description": description or label,
                    "normalized_label": _normalize_topic_key(label),
                    "sentiment_llm": sentiment,
                    "raw_order": raw_order,
                    "text": post_texts[source_row_index] if source_row_index < len(post_texts) else "",
                }
            )
    return instances


def _group_by_normalized_label(instances: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for instance in instances:
        key = str(instance.get("normalized_label") or instance.get("label") or "").strip()
        if key:
            grouped[key].append(instance)

    groups = []
    for key, group_instances in grouped.items():
        groups.append(
            {
                "normalized_label": key,
                "instances": list(group_instances),
                "label": _choose_canonical_text([str(item["label"]) for item in group_instances]),
                "description": _choose_canonical_text([str(item["description"]) for item in group_instances]),
            }
        )
    return groups


def _merge_similar_groups(
    groups: list[dict[str, Any]],
    *,
    embedder: Any | None,
    threshold: float,
) -> list[dict[str, Any]]:
    if len(groups) <= 1:
        return groups

    embeddings = _topic_embeddings(groups, embedder)
    order = sorted(range(len(groups)), key=lambda idx: len(groups[idx]["instances"]), reverse=True)
    merged: list[dict[str, Any]] = []
    merged_vectors: list[Any] = []

    for group_index in order:
        group = groups[group_index]
        target_index = None
        for existing_index, existing in enumerate(merged):
            lexical_score = _lexical_similarity(str(group["label"]), str(existing["label"]))
            semantic_score = 0.0
            if embeddings is not None and group_index < len(embeddings) and existing_index < len(merged_vectors):
                semantic_score = _cosine_similarity(embeddings[group_index], merged_vectors[existing_index])
            if lexical_score >= 0.72 or semantic_score >= threshold:
                target_index = existing_index
                break

        if target_index is None:
            merged.append(
                {
                    "normalized_label": group["normalized_label"],
                    "instances": list(group["instances"]),
                    "label": group["label"],
                    "description": group["description"],
                }
            )
            if embeddings is not None and group_index < len(embeddings):
                merged_vectors.append(embeddings[group_index])
            else:
                merged_vectors.append(None)
        else:
            merged[target_index]["instances"].extend(group["instances"])
            labels = [str(item["label"]) for item in merged[target_index]["instances"]]
            descriptions = [str(item["description"]) for item in merged[target_index]["instances"]]
            merged[target_index]["label"] = _choose_canonical_text(labels)
            merged[target_index]["description"] = _choose_canonical_text(descriptions)
            if embeddings is not None and group_index < len(embeddings) and merged_vectors[target_index] is not None:
                merged_vectors[target_index] = _average_vectors(
                    [merged_vectors[target_index], embeddings[group_index]]
                )

    return sorted(merged, key=lambda item: (-_unique_post_count(item["instances"]), str(item["label"])))


def _build_canonical_topics(
    *,
    cluster_id: int,
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    topics: list[dict[str, Any]] = []
    raw_group_to_topic: dict[int, dict[str, Any]] = {}
    sorted_groups = sorted(
        enumerate(groups),
        key=lambda item: (-_unique_post_count(item[1]["instances"]), str(item[1]["label"])),
    )

    for position, (group_index, group) in enumerate(sorted_groups, start=1):
        instances = list(group["instances"])
        source_indices = sorted({int(item["source_row_index"]) for item in instances})
        counts = Counter(
            normalize_sentiment_label(item.get("sentiment_llm"))
            for item in _dedupe_instances_by_post(instances)
        )
        aliases = sorted(
            {
                str(item["label"])
                for item in instances
                if str(item["label"]) != str(group["label"])
            }
        )[:8]
        example_instances = _dedupe_instances_by_post(instances)[:3]
        topic = {
            "topic_id": _topic_id(cluster_id, position),
            "topic_label": str(group["label"]),
            "topic_description": str(group["description"]),
            "post_count": len(source_indices),
            "sentiment_llm_counts": _ordered_counts(counts),
            "source_row_indices": source_indices,
            "aliases": aliases,
            "example_source_row_indices": [int(item["source_row_index"]) for item in example_instances],
            "example_texts": [str(item.get("text") or "") for item in example_instances],
        }
        topics.append(topic)
        raw_group_to_topic[group_index] = topic

    return topics, raw_group_to_topic


def _topic_embeddings(groups: Sequence[Mapping[str, Any]], embedder: Any | None):
    if embedder is None or not hasattr(embedder, "embed_texts"):
        return None
    texts = [
        f"{group.get('label', '')}. {group.get('description', '')}"
        for group in groups
    ]
    try:
        embeddings = embedder.embed_texts(texts)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Topic embedding consolidation skipped: %s", exc)
        return None
    if len(embeddings) != len(groups):
        LOGGER.warning("Topic embedding consolidation skipped: embedding count mismatch")
        return None
    return embeddings


def _dedupe_instances_by_post(instances: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[int] = set()
    deduped: list[Mapping[str, Any]] = []
    for instance in sorted(instances, key=lambda item: (int(item["source_row_index"]), int(item.get("raw_order", 0)))):
        source_row_index = int(instance["source_row_index"])
        if source_row_index in seen:
            continue
        seen.add(source_row_index)
        deduped.append(instance)
    return deduped


def _unique_post_count(instances: Sequence[Mapping[str, Any]]) -> int:
    return len({int(item["source_row_index"]) for item in instances})


def _choose_canonical_text(values: Sequence[str]) -> str:
    clean_values = [" ".join(str(value).split()) for value in values if str(value).strip()]
    if not clean_values:
        return ""
    counts = Counter(clean_values)
    return sorted(counts, key=lambda value: (-counts[value], len(value), value))[0]


def _ordered_counts(counts: Counter) -> dict[str, int]:
    return {sentiment: int(counts.get(sentiment, 0)) for sentiment in SENTIMENT_VALUES}


def _topic_id(cluster_id: int, position: int) -> str:
    base = f"t{max(cluster_id, 0) + 1:03d}"
    if position == 1:
        return base
    return f"{base}_{position:02d}"


def _clean_label(value: Any) -> str:
    return _truncate(" ".join(str(value or "").split()), 48)


def _clean_description(value: Any) -> str:
    return _truncate(" ".join(str(value or "").split()), 120)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _normalize_topic_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    text = re.sub(r"season\s*2|シーズン\s*2|s\s*2", "season2", text)
    text = re.sub(r"season\s*1|シーズン\s*1|s\s*1", "season1", text)
    text = re.sub(r"[\s\-_・、。，,.!?！？:：;；\"'“”‘’（）()\[\]{}]+", "", text)
    return text


def _lexical_similarity(left: str, right: str) -> float:
    left_key = _normalize_topic_key(left)
    right_key = _normalize_topic_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    if left_key in right_key or right_key in left_key:
        return min(len(left_key), len(right_key)) / max(len(left_key), len(right_key))
    left_chars = set(left_key)
    right_chars = set(right_key)
    return len(left_chars & right_chars) / (len(left_chars | right_chars) or 1)


def _cosine_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    left_values = _as_float_list(left)
    right_values = _as_float_list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = sum(a * a for a in left_values) ** 0.5
    right_norm = sum(b * b for b in right_values) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _average_vectors(vectors: Sequence[Any]) -> list[float] | None:
    clean_vectors = [_as_float_list(vector) for vector in vectors if vector is not None]
    clean_vectors = [vector for vector in clean_vectors if vector]
    if not clean_vectors:
        return None
    length = min(len(vector) for vector in clean_vectors)
    return [
        sum(vector[index] for vector in clean_vectors) / len(clean_vectors)
        for index in range(length)
    ]


def _as_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    try:
        return [float(value) for value in vector]
    except TypeError:
        return []


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default
