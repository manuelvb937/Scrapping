"""Language detection helpers (Japanese vs others)."""

from __future__ import annotations

import re

JAPANESE_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")


def detect_language(text: str | None) -> str:
    """Detect whether text is Japanese or not.

    Returns:
        - "ja" for Japanese text
        - "other" for non-Japanese text or empty values
    """
    if not text:
        return "other"

    return "ja" if JAPANESE_CHAR_PATTERN.search(text) else "other"
