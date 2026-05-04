"""Post deduplication utilities with exact and near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


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


def _simhash(text: str, hashbits: int = 64) -> int:
    """Compute a SimHash fingerprint for near-duplicate detection.

    Pure-Python implementation using character-level shingling (n-grams)
    directly from the text. This approach works for all languages including
    Japanese, which doesn't have whitespace-separated words.
    """
    text = re.sub(r"\s+", "", text.lower())  # Remove whitespace for character-level comparison

    if len(text) < 3:
        return hash(text) & ((1 << hashbits) - 1)

    # Character-level 3-grams (shingling)
    shingles: list[str] = []
    for i in range(len(text) - 2):
        shingles.append(text[i : i + 3])

    if not shingles:
        return 0

    vector = [0] * hashbits
    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
        for i in range(hashbits):
            if h & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1

    fingerprint = 0
    for i in range(hashbits):
        if vector[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Count the number of differing bits between two integers."""
    return bin(a ^ b).count("1")


def is_near_duplicate(hash1: int, hash2: int, threshold: int = 6) -> bool:
    """Return True if two SimHash values are within Hamming distance threshold.

    Default threshold of 6 bits (out of 64) works well for social media posts
    in both Japanese and English.
    """
    return _hamming_distance(hash1, hash2) <= threshold


def deduplicate_posts(
    posts: Iterable[dict],
    *,
    near_dedup: bool = True,
    hamming_threshold: int = 3,
) -> list[dict]:
    """Return a deduplicated post list preserving first-seen order.

    First removes exact duplicates (SHA256 fingerprint), then optionally
    removes near-duplicates using SimHash with Hamming distance.
    """
    # Pass 1: exact dedup
    seen_exact: set[str] = set()
    exact_unique: list[dict] = []

    for post in posts:
        fp = _fingerprint(post)
        if fp in seen_exact:
            continue
        seen_exact.add(fp)
        exact_unique.append(post)

    if not near_dedup:
        return exact_unique

    # Pass 2: near-duplicate removal via SimHash
    output: list[dict] = []
    seen_hashes: list[int] = []

    for post in exact_unique:
        content = str(post.get("content", ""))
        sh = _simhash(content)

        is_dup = False
        for existing_hash in seen_hashes:
            if is_near_duplicate(sh, existing_hash, threshold=hamming_threshold):
                is_dup = True
                break

        if not is_dup:
            seen_hashes.append(sh)
            output.append(post)

    return output
