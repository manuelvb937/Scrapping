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
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Mapping, Sequence

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
REPLY_CONTEXT = re.compile(
    r"(?:返信先|replying\s+to)[:：]?\s*(?:@?\w{2,30}\s*){1,4}",
    re.IGNORECASE | re.UNICODE,
)

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

LATIN_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-]{1,}")
JAPANESE_SEQUENCE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9fー]{2,}")

TOPIC_STOPWORDS = {
    # Japanese particles and common words
    "の", "に", "は", "を", "が", "で", "と", "も", "た", "て", "し", "な", "い",
    "いる", "ある", "する", "なる", "です", "ます", "ない", "こと", "もの", "これ",
    "それ", "あれ", "ここ", "そこ", "さん", "くん", "ちゃん", "って", "けど", "ので",
    "から", "まで", "より", "よう", "みたい", "そして", "でも", "また", "もう", "まだ",
    # Common scraped/web noise
    "amp", "utm", "source", "medium", "search", "yjrealtime", "pic", "https", "http",
    "www", "com", "co", "jp",
    "返信", "返信先", "reply", "replies", "replying", "to", "rt", "via",
    "twitter", "tweet", "post", "user", "account", "follow", "フォロー",
    "こちら", "これ", "それ", "あれ", "ここ", "そこ", "ため", "もの",
    "する", "なる", "いる", "ある", "ない", "さん", "ちゃん", "くん",
}


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
    text = REPLY_CONTEXT.sub(" ", text)

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


def extract_cluster_keywords_ctfidf(
    cluster_texts: Mapping[int, Sequence[str]],
    *,
    top_n: int = 10,
    search_terms: set[str] | None = None,
    reduce_frequent_words: bool = True,
) -> dict[int, list[tuple[str, float]]]:
    """Extract BERTopic-style c-TF-IDF keywords for each cluster.

    Unlike simple per-cluster frequency, c-TF-IDF scores terms by how important
    they are inside one cluster compared with all other clusters. This makes the
    keywords better topic descriptors after HDBSCAN has already grouped posts.

    SudachiPy is used for Japanese tokenization when installed. If it is not
    available, a built-in fallback uses Latin tokens plus Japanese character
    n-grams, so the pipeline still runs without extra dependencies.
    """
    if not cluster_texts:
        return {}

    search_terms = search_terms or set()
    class_counts: dict[int, Counter[str]] = {}
    term_total_frequency: Counter[str] = Counter()
    class_lengths: dict[int, int] = {}

    for cluster_id, texts in cluster_texts.items():
        tokens: list[str] = []
        for text in texts:
            tokens.extend(tokenize_topic_text(text, search_terms=search_terms))
        counts = Counter(tokens)
        class_counts[cluster_id] = counts
        class_lengths[cluster_id] = sum(counts.values())
        term_total_frequency.update(counts)

    non_empty_lengths = [length for length in class_lengths.values() if length > 0]
    avg_class_length = sum(non_empty_lengths) / len(non_empty_lengths) if non_empty_lengths else 1.0

    output: dict[int, list[tuple[str, float]]] = {}
    for cluster_id, counts in class_counts.items():
        class_length = class_lengths.get(cluster_id, 0)
        if class_length <= 0:
            output[cluster_id] = []
            continue

        scored_terms: list[tuple[str, float]] = []
        for term, count in counts.items():
            tf = count / class_length
            if reduce_frequent_words:
                tf = math.sqrt(tf)

            # BERTopic-style class-based IDF. Terms that appear in many classes
            # get less weight, cluster-specific terms get more weight.
            total_frequency = term_total_frequency.get(term, 1)
            idf = math.log(1 + (avg_class_length / total_frequency))
            score = tf * idf
            scored_terms.append((term, score))

        scored_terms.sort(key=lambda item: item[1], reverse=True)
        output[cluster_id] = _diversify_keyword_list(scored_terms, top_n=top_n)

    return output


def tokenize_topic_text(text: str, *, search_terms: set[str] | None = None) -> list[str]:
    """Tokenize text for topic representation with Japanese-aware fallbacks."""
    normalized = _normalize_fullwidth(str(text or "")).lower()
    if not normalized.strip():
        return []

    search_terms = search_terms or set()
    sudachi_tokens = _tokenize_with_sudachi(normalized)
    if sudachi_tokens is not None:
        return [
            token
            for token in sudachi_tokens
            if _keep_topic_token(token, search_terms=search_terms)
        ]

    tokens: list[str] = []
    tokens.extend(LATIN_TOKEN.findall(normalized))
    for sequence in JAPANESE_SEQUENCE.findall(normalized):
        tokens.extend(_japanese_character_ngrams(sequence))

    return [
        token
        for token in tokens
        if _keep_topic_token(token, search_terms=search_terms)
    ]


def _keep_topic_token(token: str, *, search_terms: set[str]) -> bool:
    token = token.strip().lower()
    if len(token) < 2:
        return False
    if token in TOPIC_STOPWORDS:
        return False
    if _looks_like_topic_noise(token):
        return False
    if token.isdigit():
        return False
    return not _is_search_term_token(token, search_terms)


def _looks_like_topic_noise(token: str) -> bool:
    useful_latin = {
        "netflix", "hulu", "unext", "u-next", "telasa", "fod", "amazon",
        "rakuten", "youtube", "tver", "bl", "blcd", "dvd", "blu", "ray",
        "bluray", "sns", "x",
    }
    if token in useful_latin:
        return False
    if re.fullmatch(r"[_\W]+", token, flags=re.UNICODE):
        return True
    if re.fullmatch(r"[━─✨❤♥♡❣️‼!?！？・、。,.\\/\-_=]+", token, flags=re.UNICODE):
        return True
    if re.fullmatch(r"[a-z]{8,}\d*", token) and _has_repeated_character_run(token):
        return True
    if re.fullmatch(r"[a-z0-9_]{10,}", token) and not any(marker in token for marker in useful_latin):
        return True
    return False


def _has_repeated_character_run(token: str) -> bool:
    return any(
        token[index] == token[index - 1] == token[index - 2]
        for index in range(2, len(token))
    )


def _is_search_term_token(token: str, search_terms: set[str]) -> bool:
    if not search_terms:
        return False
    return any(token == term or token in term or term in token for term in search_terms)


def _japanese_character_ngrams(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= 6:
        return [compact]

    ngrams: list[str] = []
    for n in (2, 3, 4):
        for i in range(0, len(compact) - n + 1):
            ngrams.append(compact[i : i + n])
    return ngrams


def _diversify_keyword_list(scored_terms: Sequence[tuple[str, float]], *, top_n: int) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    for term, score in scored_terms:
        if any(term in chosen or chosen in term for chosen, _ in selected):
            continue
        selected.append((term, round(float(score), 6)))
        if len(selected) >= top_n:
            break
    return selected


@lru_cache(maxsize=1)
def _sudachi_resources():
    try:
        from sudachipy import dictionary
        from sudachipy import tokenizer as sudachi_tokenizer
    except ImportError:
        return None

    tokenizer_obj = dictionary.Dictionary().create()
    return tokenizer_obj, sudachi_tokenizer.Tokenizer.SplitMode.C


def _tokenize_with_sudachi(text: str) -> list[str] | None:
    resources = _sudachi_resources()
    if resources is None:
        return None

    tokenizer_obj, split_mode = resources
    tokens: list[str] = []
    for morpheme in tokenizer_obj.tokenize(text, split_mode):
        pos = morpheme.part_of_speech()
        if not pos:
            continue
        if pos[0] not in {"名詞", "動詞", "形容詞"}:
            continue
        token = morpheme.normalized_form() or morpheme.dictionary_form() or morpheme.surface()
        token = token.strip().lower()
        if token:
            tokens.append(token)
    return tokens


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
    text = REPLY_CONTEXT.sub(" ", text)

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
