# Social Listening Pipeline

Clean Python project scaffold for building a social listening pipeline to analyze X posts for clustering and topic extraction.

## Current scope

- Project structure and packaging
- CLI entry point
- Centralized logging setup
- Config handling via environment variables
- Yahoo realtime raw collector
- Modular preprocessing (deduplication, language detection, text cleaning)
- Analysis pipeline (embeddings, clustering, topic/sentiment labeling)
- Reporting pipeline (markdown/html/csv outputs for marketing)

> Note: scraper and preprocessing are implemented; topic/sentiment uses LLM structured JSON output when `OPENAI_API_KEY` is set, otherwise a deterministic fallback is used.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
social-listening --help
```


### Yahoo realtime scraper runtime requirements

The Yahoo realtime scraper uses Selenium + headless Chrome. Install browser dependencies before scraping:

```bash
sudo apt update
sudo apt install -y chromium chromium-driver
```

Validate browser wiring before scraping:

```bash
python cli.py scrape --check-chrome
```

## Run Yahoo realtime scraper

```bash
yahoo-realtime-scrape --keywords "openai" "ai safety" --max-posts 50 --max-scrolls 5
```

Output is written to `data/raw/yahoo_realtime_<timestamp>.jsonl`.

## Run preprocessing

```bash
preprocess-posts --input data/raw/yahoo_realtime_<timestamp>.jsonl --output-dir data/processed
```

Output is written to `data/processed/<input_name>_processed.jsonl`.

## Run analysis

```bash
run-analysis --input data/processed/<input_name>_processed.jsonl --output-dir data/reports
```

Outputs:
- `data/reports/clusters.json`
- `data/reports/analysis.json`

## Run reporting

```bash
generate-report --clusters data/reports/clusters.json --analysis data/reports/analysis.json --output-dir data/reports
```

Outputs:
- `data/reports/report.md`
- `data/reports/report.html`
- `data/reports/clusters.csv`

## Unified CLI

You can also run the whole pipeline from the repository root:

```bash
python cli.py scrape --keywords "openai"
python cli.py preprocess
python cli.py analyze
python cli.py report
```
