"""Unified project CLI with scrape/preprocess/analyze/report commands.

Examples:
    python cli.py scrape --keywords openai
    python cli.py preprocess
    python cli.py analyze
    python cli.py report
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.pipeline import run_analysis
from collector.yahoo_realtime import scrape_yahoo_realtime, smoke_test_chrome_driver
from preprocessing.pipeline import preprocess_raw_file
from reporting.generator import generate_reports
from reporting.exporter import export_processed_to_html


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def command_scrape(args: argparse.Namespace) -> None:
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
        output_dir=args.raw_dir,
    )


def command_preprocess(args: argparse.Namespace) -> None:
    input_path = Path(args.input) if args.input else latest_file(Path(args.raw_dir), "*.jsonl")
    if input_path is None:
        raise FileNotFoundError(f"No raw JSONL files found in {args.raw_dir}")

    output_path = preprocess_raw_file(input_path, args.processed_dir)
    print(f"Processed file: {output_path}")


def command_analyze(args: argparse.Namespace) -> None:
    input_path = Path(args.input) if args.input else latest_file(Path(args.processed_dir), "*.jsonl")
    if input_path is None:
        raise FileNotFoundError(f"No processed JSONL files found in {args.processed_dir}")

    clusters_path, analysis_path = run_analysis(input_path, args.reports_dir)
    print(f"Clusters JSON: {clusters_path}")
    print(f"Analysis JSON: {analysis_path}")


def command_report(args: argparse.Namespace) -> None:
    reports_dir = Path(args.reports_dir)
    clusters_path = Path(args.clusters) if args.clusters else reports_dir / "clusters.json"
    analysis_path = Path(args.analysis) if args.analysis else reports_dir / "analysis.json"

    if not clusters_path.exists():
        raise FileNotFoundError(f"clusters.json not found: {clusters_path}")
    if not analysis_path.exists():
        raise FileNotFoundError(f"analysis.json not found: {analysis_path}")

    md_path, html_path, csv_path = generate_reports(clusters_path, analysis_path, reports_dir)
    print(f"Markdown report: {md_path}")
    print(f"HTML report: {html_path}")
    print(f"Clusters CSV: {csv_path}")


def command_export(args: argparse.Namespace) -> None:
    input_path = Path(args.input) if args.input else latest_file(Path(args.processed_dir), "*.jsonl")
    if input_path is None:
        raise FileNotFoundError(f"No processed JSONL files found in {args.processed_dir}")

    output_path = export_processed_to_html(input_path, Path(args.reports_dir))
    print(f"Data successfully exported and opened: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Social listening unified CLI")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--reports-dir", default="data/reports", help="Reports directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape", help="Run Yahoo realtime scraper")
    scrape.add_argument("--keywords", nargs="+", default=["openai"], help="Keywords to scrape")
    scrape.add_argument("--max-posts", type=int, default=100)
    scrape.add_argument("--max-scrolls", type=int, default=10)
    scrape.add_argument("--check-chrome", action="store_true", help="Validate Selenium + Chrome setup and exit")
    scrape.set_defaults(func=command_scrape)

    preprocess = subparsers.add_parser("preprocess", help="Preprocess latest/raw JSONL")
    preprocess.add_argument("--input", default=None, help="Raw JSONL input (defaults to latest in raw dir)")
    preprocess.set_defaults(func=command_preprocess)

    analyze = subparsers.add_parser("analyze", help="Analyze latest/processed JSONL")
    analyze.add_argument("--input", default=None, help="Processed JSONL input (defaults to latest in processed dir)")
    analyze.set_defaults(func=command_analyze)

    report = subparsers.add_parser("report", help="Generate report artifacts")
    report.add_argument("--clusters", default=None, help="Path to clusters.json (defaults to reports dir)")
    report.add_argument("--analysis", default=None, help="Path to analysis.json (defaults to reports dir)")
    report.set_defaults(func=command_report)

    export = subparsers.add_parser("export", help="Export processed data to an HTML table for manual inspection")
    export.add_argument("--input", default=None, help="Processed JSONL input (defaults to latest in processed dir)")
    export.set_defaults(func=command_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
