"""Yahoo realtime search scraper for collecting X post search results.

This implementation uses Yahoo Japan realtime pages with Selenium-based dynamic loading,
which is more robust for Yahoo realtime timelines than static HTML scraping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)
YAHOO_REALTIME_URL = "https://search.yahoo.co.jp/realtime/search"

AUTHOR_ID_PATTERN = re.compile(r'<a[^>]*class="[^"]*Tweet_authorID__JKhEb[^"]*"[^>]*>(.*?)</a>', re.DOTALL)
TIME_PATTERN = re.compile(r'<time[^>]*datetime="([^"]+)"', re.IGNORECASE)
DATA_CL_PATTERN = re.compile(r'data-cl-params="([^"]+)"')
TAG_PATTERN = re.compile(r"<[^>]+>")
BROWSER_CANDIDATES = ["google-chrome", "chromium", "chromium-browser", "chrome"]


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



def find_browser_binary() -> str | None:
    for candidate in BROWSER_CANDIDATES:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def validate_chrome_setup() -> tuple[str, str | None]:
    """Return detected browser and chromedriver paths, or raise helpful error."""
    browser_path = find_browser_binary()
    chromedriver_path = shutil.which("chromedriver")

    if not browser_path:
        raise RuntimeError(
            "Chrome/Chromium browser not found. Install one of: google-chrome, chromium, chromium-browser."
        )

    return browser_path, chromedriver_path


def smoke_test_chrome_driver() -> tuple[bool, str]:
    """Try opening and closing a headless browser to validate runtime wiring."""
    try:
        driver = set_chromedriver()
        driver.get("about:blank")
        driver.quit()
        return True, "Chrome + Selenium check passed"
    except ModuleNotFoundError as exc:
        return False, (
            "Chrome + Selenium check failed: selenium is not installed. "
            "Run: source .venv/bin/activate && pip install -e . "
            f"(detail: {exc})"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Chrome + Selenium check failed: {exc}"


def set_chromedriver():
    """Create a headless Chrome webdriver instance."""
    from selenium import webdriver

    browser_path, chromedriver_path = validate_chrome_setup()

    options = webdriver.ChromeOptions()
    options.binary_location = browser_path
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    if chromedriver_path:
        from selenium.webdriver.chrome.service import Service

        service = Service(executable_path=chromedriver_path)
        return webdriver.Chrome(service=service, options=options)

    return webdriver.Chrome(options=options)


def parse_tweet_id_from_data_cl(data_cl: str | None) -> str | None:
    if not data_cl:
        return None
    for part in data_cl.split(";"):
        if part.startswith("twid:"):
            return part.replace("twid:", "").strip()
    return None


def _strip_tags(html: str) -> str:
    text = TAG_PATTERN.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_username_from_html(card_html: str) -> str | None:
    m = AUTHOR_ID_PATTERN.search(card_html)
    if m:
        username = _strip_tags(m.group(1)).lstrip("@")
        return username or None

    text = _strip_tags(card_html)
    m2 = re.search(r"@([A-Za-z0-9_]{1,15})", text)
    return m2.group(1) if m2 else None


def extract_timestamp_from_html(card_html: str) -> str | None:
    m = TIME_PATTERN.search(card_html)
    return m.group(1) if m else None


def extract_card_record(
    *,
    card_html: str,
    keyword: str,
    query: str,
    fetched_at: str,
    scroll_index: int,
) -> tuple[ScrapeRecord | None, str | None]:
    """Extract one ScrapeRecord and tweet_id from a Yahoo realtime card HTML."""
    content = _strip_tags(card_html) or None
    username = extract_username_from_html(card_html)
    timestamp = extract_timestamp_from_html(card_html)

    tweet_id: str | None = None
    media_urls: list[str] = []

    for data_cl in DATA_CL_PATTERN.findall(card_html):
        parsed_twid = parse_tweet_id_from_data_cl(data_cl)
        if parsed_twid:
            tweet_id = parsed_twid

    for href in re.findall(r'href="([^"]+)"', card_html):
        if href.startswith("http") and "search.yahoo.co.jp/realtime/search/tweet/" not in href:
            media_urls.append(href)

    if not tweet_id:
        return None, None

    yahoo_link = (
        f"https://search.yahoo.co.jp/realtime/search/tweet/{tweet_id}"
        "?detail=1&ifr=tl_twdtl&rkf=1"
    )

    record = ScrapeRecord(
        query=query,
        keyword=keyword,
        username=username,
        timestamp=timestamp,
        content=content,
        url=yahoo_link,
        media_urls=media_urls,
        fetched_at=fetched_at,
        scroll_index=scroll_index,
    )
    return record, tweet_id


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


def scrape_keyword_with_retry(
    *,
    keyword: str,
    max_scrolls: int,
    max_posts_per_keyword: int,
    fetched_at: str,
    attempts: int = 3,
) -> list[ScrapeRecord]:
    """Scrape one keyword with retry around driver startup/navigation failures."""
    from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException, WebDriverException
    from selenium.webdriver.common.by import By

    query = keyword

    for attempt in range(1, attempts + 1):
        driver = None
        try:
            driver = set_chromedriver()
            driver.get(f"{YAHOO_REALTIME_URL}?p={keyword}&ei=UTF-8")
            time.sleep(random.uniform(1.0, 2.0))

            seen_ids: set[str] = set()
            records: list[ScrapeRecord] = []

            for scroll_index in range(max_scrolls):
                cards = driver.find_elements(By.CLASS_NAME, "Tweet_Tweet__sna2i")
                LOGGER.info("keyword='%s' scroll=%s cards=%s", keyword, scroll_index, len(cards))

                for card in cards:
                    if len(records) >= max_posts_per_keyword:
                        break
                    try:
                        card_html = card.get_attribute("outerHTML")
                        record, tweet_id = extract_card_record(
                            card_html=card_html,
                            keyword=keyword,
                            query=query,
                            fetched_at=fetched_at,
                            scroll_index=scroll_index,
                        )
                        if not record or not tweet_id or tweet_id in seen_ids:
                            continue
                        seen_ids.add(tweet_id)
                        records.append(record)
                    except Exception as card_exc:  # noqa: BLE001
                        LOGGER.debug("Skipping card due to parse error: %s", card_exc)
                        continue

                if len(records) >= max_posts_per_keyword:
                    break

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(1.0, 1.5))

                try:
                    more_link = driver.find_element(By.CSS_SELECTOR, 'a[data-cl-params*="_cl_link:more"]')
                    driver.execute_script("arguments[0].scrollIntoView(true);", more_link)
                    time.sleep(0.5)
                    try:
                        more_link.click()
                    except ElementClickInterceptedException:
                        driver.execute_script("arguments[0].click();", more_link)
                    time.sleep(random.uniform(1.5, 2.5))
                except NoSuchElementException:
                    LOGGER.info("No 'more' link for keyword='%s' at scroll=%s", keyword, scroll_index)
                    break

            return records

        except WebDriverException as exc:
            LOGGER.warning("WebDriver failed for keyword='%s' (attempt %s/%s): %s", keyword, attempt, attempts, exc)
            if attempt == attempts:
                return []
            time.sleep((2 ** (attempt - 1)) + random.random())
        finally:
            if driver is not None:
                driver.quit()

    return []


def scrape_yahoo_realtime(
    *,
    keywords: list[str],
    max_posts: int,
    max_scrolls: int,
    output_dir: str | Path = "data/raw",
    timeout_seconds: int = 30,
) -> Path:
    """Scrape Yahoo realtime results for each keyword and save raw records to JSONL."""
    del timeout_seconds  # intentionally unused in Selenium-based implementation

    if max_posts <= 0:
        raise ValueError("max_posts must be > 0")
    if max_scrolls <= 0:
        raise ValueError("max_scrolls must be > 0")
    if not keywords:
        raise ValueError("keywords cannot be empty")

    validate_chrome_setup()

    fetched_at = datetime.now(timezone.utc).isoformat()
    all_records: list[ScrapeRecord] = []

    max_posts_per_keyword = max(1, max_posts // max(1, len(keywords)))

    for keyword in keywords:
        LOGGER.info("Collecting keyword='%s'", keyword)
        keyword_records = scrape_keyword_with_retry(
            keyword=keyword,
            max_scrolls=max_scrolls,
            max_posts_per_keyword=max_posts_per_keyword,
            fetched_at=fetched_at,
        )
        all_records.extend(keyword_records)
        if len(all_records) >= max_posts:
            break

    deduped_records = deduplicate_records(all_records)[:max_posts]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"yahoo_realtime_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        for record in deduped_records:
            file.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    LOGGER.info("Saved %s records to %s", len(deduped_records), file_path)
    return file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yahoo realtime search scraper (raw collection only).")
    parser.add_argument("--keywords", nargs="+", default=[], help="Keyword list to search for.")
    parser.add_argument("--max-posts", type=int, default=100, help="Maximum number of records to keep.")
    parser.add_argument("--max-scrolls", type=int, default=10, help="Maximum number of Yahoo pagination steps.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory where JSONL output is saved.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, etc.).")
    parser.add_argument("--check-chrome", action="store_true", help="Only validate Selenium/Chrome setup and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.check_chrome:
        ok, message = smoke_test_chrome_driver()
        if ok:
            print(message)
            return
        raise RuntimeError(message)

    scrape_yahoo_realtime(
        keywords=args.keywords,
        max_posts=args.max_posts,
        max_scrolls=args.max_scrolls,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
