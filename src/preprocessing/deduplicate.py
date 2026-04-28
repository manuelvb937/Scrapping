"""Post deduplication utilities."""

from __future__ import annotations

import hashlib
from typing import Iterable


def _fingerprint(post: dict) -> str:
    base = "|".join(
        [
            str(post.get("query", "")),
            str(post.get("keyword", "")),
            str(post.get("url", "")),
            str(post.get("content", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def deduplicate_posts(posts: Iterable[dict]) -> list[dict]:
    """Return a deduplicated post list preserving first-seen order."""
    seen: set[str] = set()
    output: list[dict] = []

    for post in posts:
        fp = _fingerprint(post)
        if fp in seen:
            continue
        seen.add(fp)
        output.append(post)

    return output
