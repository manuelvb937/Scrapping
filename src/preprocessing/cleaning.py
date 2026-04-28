"""Text cleaning utilities for downstream processing."""

from __future__ import annotations

import re

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Clean text for preprocessing without mutating raw source records."""
    if not text:
        return ""

    cleaned = URL_PATTERN.sub(" ", text)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned
