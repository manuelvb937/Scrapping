from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli import build_parser, latest_file


def test_build_parser_supports_required_commands() -> None:
    parser = build_parser()

    for command in ["scrape", "preprocess", "analyze", "report"]:
        args = parser.parse_args([command])
        assert args.command == command


def test_latest_file_returns_most_recent(tmp_path: Path) -> None:
    older = tmp_path / "a.jsonl"
    newer = tmp_path / "b.jsonl"
    older.write_text("a", encoding="utf-8")
    newer.write_text("b", encoding="utf-8")

    now = time.time()
    os.utime(older, (now - 10, now - 10))
    os.utime(newer, (now, now))

    result = latest_file(tmp_path, "*.jsonl")
    assert result is not None
    assert result.name == "b.jsonl"
