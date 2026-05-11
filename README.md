# Social Listening Pipeline

Clean Python project scaffold for building a social listening pipeline to analyze X posts for clustering and topic extraction.

## Current scope

- Project structure and packaging
- CLI entry point
- Centralized logging setup
- Config handling via environment variables
- Yahoo realtime raw collector
- Modular preprocessing (deduplication, language detection, text cleaning)
- Analysis pipeline (embeddings, clustering, LLM topic labeling, transformer sentiment)
- Reporting pipeline (markdown/html/csv outputs for marketing)

> Note: scraper and preprocessing are implemented. Cluster topic labeling uses LLM structured JSON output when an API key is set. Post-level sentiment defaults to a fine-tuned Hugging Face transformer, with optional LLM review for ambiguous Japanese fandom/SNS slang.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
social-listening --help
```

## Analysis architecture

```text
Posts
↓
Embeddings
↓
UMAP
↓
HDBSCAN
↓
LLM topic labeling per cluster
↓
Transformer sentiment per post
↓
LLM fallback for ambiguous cases
↓
Cluster-level sentiment aggregation
↓
Final report
```

Topic labeling and sentiment analysis are separate responsibilities:

- `analysis.topic_labeling.TopicLabeler` labels only cluster topics with an LLM.
- `analysis.sentiment_transformer.TransformerSentimentAnalyzer` classifies each post with a fine-tuned transformer model.
- `analysis.sentiment_transformer.LLMSentimentReviewer` is used only as a fallback/reviewer in hybrid mode.

The transformer model is already fine-tuned for sentiment analysis. You do not need to train a model at this stage.

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


## LLM provider configuration

Topic labeling and optional sentiment review support two providers:

- OpenAI (default): set `OPENAI_API_KEY`
- Google Gemini: set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`

Example (Gemini):

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your_key"
export GEMINI_MODEL="gemini-2.5-flash"
export TOPIC_LABELING_MODEL="gemini-2.5-flash"
python cli.py analyze
python cli.py report
```

## Sentiment configuration

Set `SENTIMENT_METHOD` to choose how post-level sentiment is calculated:

- `llm`: legacy LLM sentiment behavior for backward compatibility.
- `transformer`: classify every post with the Hugging Face transformer.
- `hybrid`: run the transformer first, then send low-confidence or ambiguous slang posts to the LLM reviewer.

Default settings:

```bash
export SENTIMENT_METHOD=hybrid
export SENTIMENT_MODEL_NAME=LoneWolfgang/bert-for-japanese-twitter-sentiment
export SENTIMENT_CONFIDENCE_THRESHOLD=0.70
export ENABLE_LLM_SENTIMENT_FALLBACK=true
export SENTIMENT_LLM_FALLBACK_BATCH_SIZE=25
export SENTIMENT_LLM_REVIEW_DELAY_SECONDS=0
export TOPIC_LABELING_MODEL=gpt-4.1-mini
export TOPIC_LABELING_DELAY_SECONDS=0
export TOPIC_LABELING_429_COOLDOWN_SECONDS=60
export LLM_TEMPERATURE=0
export GEMINI_THINKING_BUDGET=0
```

The default sentiment model is Japanese Twitter sentiment:

```bash
export SENTIMENT_MODEL_NAME=LoneWolfgang/bert-for-japanese-twitter-sentiment
```

The transformer output is normalized to `positive`, `neutral`, and `negative`. `LoneWolfgang/bert-for-japanese-twitter-sentiment` uses `0 -> negative`, `1 -> neutral`, and `2 -> positive`; the pipeline also maps `LABEL_0`, `LABEL_1`, and `LABEL_2` the same way. Star-rating or "very positive/negative" labels from other models are folded into the same three labels. LLM sentiment review returns only the sentiment label to reduce token usage; sentiment rationales are written as `null` in the structured dashboard output.

## Run analysis

```bash
run-analysis --input data/processed/<input_name>_processed.jsonl --output-dir data/reports
```

For free-tier API limits, use conservative batching and longer delays:

```bash
python cli.py analyze --free
```

`--free` enables Gemini free-tier protection while preserving any sentiment method you explicitly set. By default it uses hybrid sentiment, batches LLM sentiment review in groups of 25, caps fallback reviews at 30 posts, lowers the low-confidence fallback threshold to `0.55`, and uses a shared local limiter for Gemini requests.

To run full legacy LLM sentiment under the same free-tier limiter:

```bash
export SENTIMENT_METHOD=llm
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your_key"
python cli.py analyze --free
```

The Gemini limiter defaults to 10 requests/minute, 250,000 estimated tokens/minute, and 250 requests/day. The local daily counter is stored in `data/reports/llm_rate_limit_state.json`.
For Gemini 2.5 Flash, `--free` also sets `GEMINI_THINKING_BUDGET=0` unless you override it, which disables thinking tokens for cheaper/faster label-only calls.

Outputs:
- `data/reports/clusters.json`
- `data/reports/analysis.json`
- `data/reports/structured_output.json`

`analysis.json` includes cluster topic fields plus sentiment aggregated from individual posts:

```json
{
  "cluster_id": 0,
  "topic_label": "ドラマSeason2への期待",
  "topic_summary": "続編やキャスト続投への期待を表す投稿が中心。",
  "sentiment_distribution": {
    "positive": 0.72,
    "neutral": 0.21,
    "negative": 0.07
  },
  "dominant_sentiment": "positive"
}
```

## Sentiment demo

```bash
python examples/test_hybrid_sentiment.py
```

The demo prints the transformer label, confidence, whether LLM fallback was triggered, and the final sentiment for a few Japanese posts.

## Run reporting

```bash
generate-report --clusters data/reports/clusters.json --analysis data/reports/analysis.json --output-dir data/reports
```

Outputs:
- `data/reports/report.md`
- `data/reports/report.html`
- `data/reports/clusters.csv`

`structured_output.json` is designed for dashboard/shiny consumption. It contains top-level `metadata`, `posts`, `clusters`, `daily_topic_metrics`, and `report_summary` sections. Each post includes its cluster/topic assignment, sentiment, representative-post flag, and `umap_x`/`umap_y` coordinates for cluster visualization.

## Unified CLI

You can also run the whole pipeline from the repository root:

```bash
python cli.py scrape --keywords "openai"
python cli.py preprocess
python cli.py analyze
python cli.py report
```
