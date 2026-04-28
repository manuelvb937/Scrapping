"""Yahoo real-time search scraper for collecting X post search results.

This module intentionally focuses only on raw collection.
No clustering, topic extraction, or other analysis is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)
YAHOO_SEARCH_URL = "https://search.yahoo.com/search"
X_URL_PATTERN = re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/[^\s\"'<>]+", re.IGNORECASE)
RESULT_BLOCK_PATTERN = re.compile(r"<li\b[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class ScrapeRecord:
    query: str
    keyword: str
    username: str | None
    timestamp: str | None
    content: str | None
    url: str
    media_urls: list[str]
    fetched_at: str
    scroll_index: int


def request_with_retry(url: str, *, attempts: int = 4, timeout_seconds: int = 30) -> str:
    """Fetch URL with basic retry logic and exponential backoff."""
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )

    for attempt in range(1, attempts + 1):
        req = Request(url, headers={"User-Agent": user_agent})
        try:
            with urlopen(req, timeout=timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            wait_seconds = (2 ** (attempt - 1)) + random.random()  # nosec B311
            LOGGER.warning(
                "Request failed (attempt %s/%s): %s. Retrying in %.2fs",
                attempt,
                attempts,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected retry loop completion.")


def extract_username(url: str) -> str | None:
    match = re.search(r"(?:x|twitter)\.com/([^/?#]+)/?", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def clean_text(raw_html: str) -> str | None:
    text = TAG_PATTERN.sub(" ", raw_html)
    text = unescape(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def extract_records_from_html(
    html: str,
    *,
    keyword: str,
    query: str,
    scroll_index: int,
    fetched_at: str,
) -> list[ScrapeRecord]:
    """Parse Yahoo search HTML and produce records for X/Twitter result links."""
    records: list[ScrapeRecord] = []

    for block in RESULT_BLOCK_PATTERN.findall(html):
        hrefs = HREF_PATTERN.findall(block)
        candidate_urls = [url for url in hrefs if X_URL_PATTERN.search(url)]
        media_urls = [url for url in hrefs if url.lower().startswith("http") and not X_URL_PATTERN.search(url)]

        if not candidate_urls:
            candidate_urls.extend(X_URL_PATTERN.findall(block))

        if not candidate_urls:
            continue

        content = clean_text(block)

        for url in candidate_urls:
            records.append(
                ScrapeRecord(
                    query=query,
                    keyword=keyword,
                    username=extract_username(url),
                    timestamp=None,
                    content=content,
                    url=url,
                    media_urls=media_urls,
                    fetched_at=fetched_at,
                    scroll_index=scroll_index,
                )
            )

    return records


def deduplicate_records(records: Iterable[ScrapeRecord]) -> list[ScrapeRecord]:
    """Deduplicate records by stable hash over query + keyword + url + content."""
    deduped: list[ScrapeRecord] = []
    seen: set[str] = set()

    for record in records:
        fingerprint = hashlib.sha256(
            f"{record.query}|{record.keyword}|{record.url}|{record.content}".encode("utf-8")
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)

    return deduped


def scrape_yahoo_realtime(
    *,
    keywords: list[str],
    max_posts: int,
    max_scrolls: int,
    output_dir: str | Path = "data/raw",
    timeout_seconds: int = 30,
) -> Path:
    """Scrape Yahoo search results for each keyword and save raw records to JSONL."""
    if max_posts <= 0:
        raise ValueError("max_posts must be > 0")
    if max_scrolls <= 0:
        raise ValueError("max_scrolls must be > 0")
    if not keywords:
        raise ValueError("keywords cannot be empty")

    fetched_at = datetime.now(timezone.utc).isoformat()
    all_records: list[ScrapeRecord] = []

    for keyword in keywords:
        query = f'site:x.com "{keyword}"'
        LOGGER.info("Collecting keyword='%s' query='%s'", keyword, query)

        for scroll_index in range(max_scrolls):
            if len(all_records) >= max_posts:
                break

            start = 1 + (scroll_index * 10)
            params = urlencode({"p": query, "b": str(start)})
            url = f"{YAHOO_SEARCH_URL}?{params}"

            html = request_with_retry(url, timeout_seconds=timeout_seconds)
            records = extract_records_from_html(
                html,
                keyword=keyword,
                query=query,
                scroll_index=scroll_index,
                fetched_at=fetched_at,
            )

            if not records:
                LOGGER.info("No records found at scroll_index=%s for keyword='%s'", scroll_index, keyword)

            all_records.extend(records)

            if len(all_records) >= max_posts:
                break

    deduped_records = deduplicate_records(all_records)[:max_posts]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"yahoo_realtime_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    with file_path.open("w", encoding="utf-8") as f:
        for record in deduped_records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    LOGGER.info("Saved %s records to %s", len(deduped_records), file_path)
    return file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yahoo realtime search scraper (raw collection only).")
    parser.add_argument("--keywords", nargs="+", required=True, help="Keyword list to search for.")
    parser.add_argument("--max-posts", type=int, default=100, help="Maximum number of records to keep.")
    parser.add_argument("--max-scrolls", type=int, default=10, help="Maximum number of Yahoo pagination steps.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory where JSONL output is saved.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, etc.).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    scrape_yahoo_realtime(
        keywords=args.keywords,
        max_posts=args.max_posts,
        max_scrolls=args.max_scrolls,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
