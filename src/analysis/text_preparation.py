"""Text preparation for clustering — strip search terms, noise, and boilerplate.

Creates a ``clustering_text`` field optimised for embedding/clustering.
The original ``cleaned_content`` is preserved for display and sentiment analysis.

Design choice: title stripping is **generic** — it reads each post's own
``query`` and ``keyword`` fields and strips those terms plus common variants
(fullwidth/halfwidth digits, with/without spaces around punctuation).
No hardcoded titles are needed.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Sequence

LOGGER = logging.getLogger(__name__)

# --- Patterns for noise removal ---

# Trailing "nickname @ handle timestamp" block scraped from Yahoo cards
# e.g. "ゆーみ @ great_mountain 15:23" or "かんな @ HeathKang6596 10:54"
# e.g. "ピピ @ 123__m514 18分前" or "敷島 @ sksmhaimen 9:23"
TRAILING_USER_TS = re.compile(
    r"\s+\S{1,20}\s+@\s*\w{1,30}\s+"              # nickname @ handle
    r"(?:\d{1,2}:\d{2}|\d{1,3}[分時間日]前|"        # 15:23 | 18分前
    r"\d{1,2}月\d{1,2}日\([月火水木金土日]\)\s*\d{1,2}:\d{2}|"  # 5月1日(金) 11:41
    r"昨日\s*\d{1,2}:\d{2})"                        # 昨日 7:01
    r"\s*$",
    re.UNICODE,
)

# Reply prefix
REPLY_PREFIX = re.compile(r"^返信先:\s*", re.UNICODE)

# Residual URLs that survived cleaning (partial domains, pic.x.com, etc.)
RESIDUAL_URL = re.compile(
    r"(?:pic\.x\.com|comicomi-studio\.com|bookmeter\.com|jp\.mercari\.com"
    r"|amzn\.to|t\.co)/\S+",
    re.UNICODE,
)

# Generic URL-like tokens
URL_LIKE = re.compile(r"\S+\.\S+/\S+", re.UNICODE)

# Multiple spaces / whitespace collapse
MULTISPACE = re.compile(r"\s+")

# Yahoo-style relative timestamps that appear inline
INLINE_TS = re.compile(
    r"\s*\d{1,3}[分時間日]前\s*$"
    r"|\s*\d{1,2}:\d{2}\s*$"
    r"|\s*昨日\s*\d{1,2}:\d{2}\s*$"
    r"|\s*\d{1,2}月\d{1,2}日\([月火水木金土日]\)\s*\d{1,2}:\d{2}\s*$",
    re.UNICODE,
)

# Standalone @ mentions (not already removed)
AT_MENTION = re.compile(r"@\s*\w{1,30}", re.UNICODE)

# Numeric-only noise (retweet count "1", like count, etc.)
STANDALONE_NUM = re.compile(r"(?<!\w)\d{1,3}(?!\w)")


def _normalize_fullwidth(text: str) -> str:
    """Convert fullwidth digits/ASCII to halfwidth for consistent matching."""
    return unicodedata.normalize("NFKC", text)


def _split_on_punctuation(term: str) -> list[str]:
    """Split a search term on Japanese/ASCII punctuation to get sub-fragments.

    For example, "25時、赤坂で" → ["25時", "赤坂で"]
    """
    parts = re.split(r'[、,。.・\s]+', term)
    return [p for p in parts if len(p) >= 2]


def _make_flexible_pattern(term: str) -> re.Pattern | None:
    """Build a single regex pattern that matches a term with flexible
    spacing/punctuation.  Returns None on regex compilation failure."""
    escaped = re.escape(term)
    # Allow optional whitespace and Japanese punctuation between characters
    # where the original had punctuation or whitespace
    flexible = re.sub(
        r"(\\[、,。.・\s]|\s)+",
        lambda _: "[\\s\u3001,\u3002.\u30fb]*",
        escaped,
    )
    try:
        return re.compile(flexible, re.UNICODE | re.IGNORECASE)
    except re.error:
        try:
            return re.compile(re.escape(term), re.UNICODE | re.IGNORECASE)
        except re.error:
            return None


def _build_title_patterns(query: str, keyword: str) -> list[re.Pattern]:
    """Build regex patterns from a post's query/keyword that match all spacing
    and punctuation variants of the search term **and its sub-fragments**.

    For example, "25時、赤坂で" generates patterns that match:
    - The full string with flexible punctuation/spacing
    - The full string with punctuation removed ("25時赤坂で")
    - Each sub-fragment individually: "25時", "赤坂で"
    This ensures that fragments like "25時" are stripped even when they
    appear without the rest of the query.
    """
    raw_terms: set[str] = set()
    for term in (query, keyword):
        if term:
            raw_terms.add(term.strip())
            raw_terms.add(_normalize_fullwidth(term.strip()))

    patterns: list[re.Pattern] = []
    seen_pattern_strings: set[str] = set()  # Avoid duplicate patterns

    for term in raw_terms:
        if len(term) < 2:
            continue

        # --- Full-term pattern with flexible punctuation ---
        pat = _make_flexible_pattern(term)
        if pat and pat.pattern not in seen_pattern_strings:
            patterns.append(pat)
            seen_pattern_strings.add(pat.pattern)

        # --- Full-term with punctuation completely removed ---
        no_punct = re.sub(r"[、,。.・\s]+", "", term)
        if no_punct != term and len(no_punct) >= 2:
            pat = _make_flexible_pattern(no_punct)
            if pat and pat.pattern not in seen_pattern_strings:
                patterns.append(pat)
                seen_pattern_strings.add(pat.pattern)

        # --- Individual sub-fragments ---
        # Split "25時、赤坂で" → ["25時", "赤坂で"] and strip each independently
        fragments = _split_on_punctuation(term)
        if len(fragments) > 1:
            for frag in fragments:
                pat = _make_flexible_pattern(frag)
                if pat and pat.pattern not in seen_pattern_strings:
                    patterns.append(pat)
                    seen_pattern_strings.add(pat.pattern)

    return patterns


def prepare_clustering_text(post: dict) -> str:
    """Create a ``clustering_text`` for one post by stripping the search term,
    trailing user/timestamp blocks, and other noise.

    The post's ``query`` and ``keyword`` fields are used to auto-detect what
    title text to strip — no hardcoded titles needed.
    """
    text = str(post.get("cleaned_content") or post.get("content") or "")
    if not text.strip():
        return ""

    # Normalize fullwidth characters for consistent processing
    text = _normalize_fullwidth(text)

    # Build title patterns from this post's own query/keyword
    query = _normalize_fullwidth(str(post.get("query") or ""))
    keyword = _normalize_fullwidth(str(post.get("keyword") or ""))
    title_patterns = _build_title_patterns(query, keyword)

    # Strip title variants
    for pat in title_patterns:
        text = pat.sub(" ", text)

    # Strip reply prefix
    text = REPLY_PREFIX.sub("", text)

    # Strip trailing "nickname @ handle timestamp"
    text = TRAILING_USER_TS.sub("", text)

    # Strip residual URLs
    text = RESIDUAL_URL.sub(" ", text)
    text = URL_LIKE.sub(" ", text)

    # Strip inline timestamps at end of text
    text = INLINE_TS.sub("", text)

    # Strip remaining @ mentions
    text = AT_MENTION.sub(" ", text)

    # Strip standalone small numbers (retweet counts, etc.)
    text = STANDALONE_NUM.sub(" ", text)

    # Strip surrounding punctuation brackets that held the title
    text = re.sub(r"[『』「」＼／()（）]+", " ", text)

    # Collapse whitespace
    text = MULTISPACE.sub(" ", text).strip()

    return text


def prepare_clustering_texts(
    posts: Sequence[dict],
    *,
    min_length: int = 5,
) -> tuple[list[str], list[int]]:
    """Prepare clustering texts for all posts.

    Args:
        posts: List of post dicts (must have cleaned_content and query/keyword).
        min_length: Minimum character length for clustering_text to be usable.

    Returns:
        Tuple of (clustering_texts, valid_indices).
        ``valid_indices`` maps each position in clustering_texts back to the
        original index in ``posts``.  Posts with too-short clustering_text
        are excluded from clustering but will be placed in a catch-all cluster.
    """
    clustering_texts: list[str] = []
    valid_indices: list[int] = []
    skipped = 0

    for idx, post in enumerate(posts):
        ct = prepare_clustering_text(post)
        if len(ct) >= min_length:
            clustering_texts.append(ct)
            valid_indices.append(idx)
        else:
            skipped += 1

    if skipped:
        LOGGER.info(
            "Text preparation: %d/%d posts skipped (clustering_text too short after title removal)",
            skipped, len(posts),
        )
    LOGGER.info(
        "Text preparation: %d posts ready for clustering",
        len(clustering_texts),
    )

    return clustering_texts, valid_indices


def extract_search_term_fragments(posts: Sequence[dict]) -> set[str]:
    """Collect all search-term fragments from posts' query/keyword fields.

    Returns a set of normalized, lowercased fragments suitable for use as
    dynamic stopwords in keyword extraction.  Includes the full query,
    the no-punctuation version, and each sub-fragment.

    Example: query="25時、赤坂で" → {"25時、赤坂で", "25時赤坂で", "25時", "赤坂で"}
    """
    fragments: set[str] = set()
    seen_raw: set[str] = set()

    for post in posts:
        for field in ("query", "keyword"):
            val = str(post.get(field) or "").strip()
            if not val or val in seen_raw:
                continue
            seen_raw.add(val)

            norm = _normalize_fullwidth(val).lower()
            fragments.add(norm)

            # No-punctuation version
            no_punct = re.sub(r'[、,。.・\s]+', '', norm)
            if no_punct != norm and len(no_punct) >= 2:
                fragments.add(no_punct)

            # Individual sub-fragments
            for part in _split_on_punctuation(norm):
                fragments.add(part)

    return fragments


def extract_top_keywords(
    texts: Sequence[str],
    *,
    top_n: int = 10,
    min_length: int = 2,
    search_terms: set[str] | None = None,
) -> list[tuple[str, int]]:
    """Extract top keywords from a list of texts using simple token frequency.

    Filters out very short tokens, common Japanese particles/stopwords,
    and optionally any search-term fragments (to prevent the original
    scraping keywords from dominating cluster keywords).

    Args:
        texts: Texts to extract keywords from.
        top_n: Maximum number of keywords to return.
        min_length: Minimum token length.
        search_terms: Optional set of lowercased search-term fragments to
            exclude.  Use ``extract_search_term_fragments(posts)`` to build
            this set from the post data.

    Returns list of (keyword, count) tuples sorted by frequency.
    """
    STOPWORDS = {
        # Japanese particles and common words
        "の", "に", "は", "を", "が", "で", "と", "も", "た", "て", "し", "な",
        "い", "る", "から", "まで", "より", "へ", "だ", "です", "ます", "ない",
        "ある", "いる", "する", "なる", "この", "その", "あの", "これ", "それ",
        "あれ", "ここ", "そこ", "あそこ", "こと", "もの", "ため", "ところ",
        "さん", "くん", "ちゃん", "って", "けど", "から", "のに", "ので",
        "という", "ように", "として", "について", "みたい", "らしい",
        # Common scraping noise
        "amp", "utm", "source", "medium", "search", "yjrealtime", "pic",
        "com", "co", "jp", "https", "http", "www",
    }

    # Merge dynamic search-term stopwords
    effective_stopwords = STOPWORDS | (search_terms or set())

    counter: Counter[str] = Counter()
    token_pat = re.compile(r"\w+", re.UNICODE)

    for text in texts:
        tokens = token_pat.findall(text.lower())
        for token in tokens:
            if len(token) >= min_length and token not in effective_stopwords:
                # Also check if the token is a substring of or contains
                # any search term fragment (handles compound tokens)
                is_search_term = any(
                    token == st or token in st or st in token
                    for st in (search_terms or set())
                )
                if not is_search_term:
                    counter[token] += 1

    return counter.most_common(top_n)


def select_representative_posts(
    texts: Sequence[str],
    *,
    limit: int = 5,
) -> list[str]:
    """Select diverse representative posts from a cluster.

    Picks posts that are different from each other by checking simple
    text overlap. Prefers medium-length posts (not too short, not too long).
    """
    if not texts:
        return []

    if len(texts) <= limit:
        return [t[:300] for t in texts]

    # Score posts: prefer medium length, penalize very short/long
    scored: list[tuple[int, float, str]] = []
    lengths = [len(t) for t in texts]
    median_len = sorted(lengths)[len(lengths) // 2] if lengths else 100

    for i, text in enumerate(texts):
        length_score = 1.0 - abs(len(text) - median_len) / max(median_len, 1)
        length_score = max(0.1, length_score)
        scored.append((i, length_score, text))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    selected_texts_lower: list[str] = []

    for _, _, text in scored:
        if len(selected) >= limit:
            break
        text_lower = text.lower()
        # Check overlap with already selected posts
        is_duplicate = False
        for prev in selected_texts_lower:
            # Simple Jaccard-like overlap check on character trigrams
            trigrams_a = {text_lower[i:i+3] for i in range(len(text_lower) - 2)}
            trigrams_b = {prev[i:i+3] for i in range(len(prev) - 2)}
            if trigrams_a and trigrams_b:
                overlap = len(trigrams_a & trigrams_b) / len(trigrams_a | trigrams_b)
                if overlap > 0.6:
                    is_duplicate = True
                    break
        if not is_duplicate:
            selected.append(text[:300])
            selected_texts_lower.append(text_lower)

    return selected


def clean_for_llm(text: str) -> str:
    """Clean text for LLM input — strip URLs, usernames, timestamps.

    Unlike ``prepare_clustering_text``, this does NOT strip the search title
    (the LLM needs it for context). It only removes noise that wastes tokens.
    """
    if not text:
        return ""

    # Normalize fullwidth
    text = _normalize_fullwidth(text)

    # Strip reply prefix
    text = REPLY_PREFIX.sub("", text)

    # Strip trailing "nickname @ handle timestamp"
    text = TRAILING_USER_TS.sub("", text)

    # Strip URLs (both full and partial)
    text = re.sub(r"https?://\S+", " ", text)
    text = RESIDUAL_URL.sub(" ", text)
    text = URL_LIKE.sub(" ", text)

    # Strip remaining @ mentions
    text = AT_MENTION.sub(" ", text)

    # Strip inline timestamps at end
    text = INLINE_TS.sub("", text)

    # Strip surrounding brackets
    text = re.sub(r"[『』「」＼／]+", " ", text)

    # Collapse whitespace
    text = MULTISPACE.sub(" ", text).strip()

    return text
