"""Preprocessing pipeline for raw JSONL posts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .cleaning import clean_text
from .deduplicate import deduplicate_posts
from .language import detect_language


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def enrich_post(post: dict) -> dict:
    """Create processed record while preserving raw fields untouched."""
    content = post.get("content")
    processed = dict(post)
    processed["cleaned_content"] = clean_text(content)
    processed["language"] = detect_language(content)
    processed["processed_at"] = datetime.now(timezone.utc).isoformat()
    return processed


def preprocess_raw_file(input_path: str | Path, output_dir: str | Path = "data/processed") -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_posts = load_jsonl(input_path)
    unique_posts = deduplicate_posts(raw_posts)
    processed_posts = [enrich_post(post) for post in unique_posts]

    output_file = output_dir / f"{input_path.stem}_processed.jsonl"
    with output_file.open("w", encoding="utf-8") as file:
        for post in processed_posts:
            file.write(json.dumps(post, ensure_ascii=False) + "\n")

    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw JSONL posts.")
    parser.add_argument("--input", required=True, help="Input raw JSONL file path.")
    parser.add_argument("--output-dir", default="data/processed", help="Processed output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = preprocess_raw_file(args.input, args.output_dir)
    print(f"Saved processed output to: {output_path}")


if __name__ == "__main__":
    main()
