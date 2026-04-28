import json
from pathlib import Path

from preprocessing.cleaning import clean_text
from preprocessing.deduplicate import deduplicate_posts
from preprocessing.language import detect_language
from preprocessing.pipeline import preprocess_raw_file


def test_deduplicate_posts_preserves_first_seen_order() -> None:
    posts = [
        {"query": "q", "keyword": "k", "url": "u1", "content": "same"},
        {"query": "q", "keyword": "k", "url": "u1", "content": "same"},
        {"query": "q", "keyword": "k", "url": "u2", "content": "different"},
    ]
    deduped = deduplicate_posts(posts)
    assert len(deduped) == 2
    assert deduped[0]["url"] == "u1"
    assert deduped[1]["url"] == "u2"


def test_detect_language_japanese_vs_other() -> None:
    assert detect_language("これは日本語です") == "ja"
    assert detect_language("This is English") == "other"


def test_clean_text_removes_url_mentions_and_hashtags() -> None:
    raw = "Hello @alice see https://example.com #news"
    assert clean_text(raw) == "Hello see"


def test_preprocess_raw_file_creates_processed_output_without_overwriting_raw(tmp_path: Path) -> None:
    raw_file = tmp_path / "sample.jsonl"
    posts = [
        {
            "query": 'site:x.com "openai"',
            "keyword": "openai",
            "username": "user1",
            "timestamp": None,
            "content": "こんにちは #AI",
            "url": "https://x.com/user1/status/1",
            "media_urls": [],
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "scroll_index": 0,
        },
        {
            "query": 'site:x.com "openai"',
            "keyword": "openai",
            "username": "user1",
            "timestamp": None,
            "content": "こんにちは #AI",
            "url": "https://x.com/user1/status/1",
            "media_urls": [],
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "scroll_index": 1,
        },
    ]

    with raw_file.open("w", encoding="utf-8") as file:
        for post in posts:
            file.write(json.dumps(post, ensure_ascii=False) + "\n")

    output_path = preprocess_raw_file(raw_file, tmp_path / "processed")

    raw_lines = raw_file.read_text(encoding="utf-8").strip().splitlines()
    processed_lines = output_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(raw_lines) == 2
    assert len(processed_lines) == 1

    processed_record = json.loads(processed_lines[0])
    assert processed_record["language"] == "ja"
    assert processed_record["cleaned_content"] == "こんにちは"
    assert "processed_at" in processed_record
