# Social Listening Pipeline

Python pipeline for scraping, preprocessing, clustering, topic extraction, sentiment analysis, and marketing-oriented reporting for social listening data.

## Current Pipeline

```text
Posts
|
Preprocess posts for clustering
|
Transformer sentiment per post -> Sentiment_Transformer
|
Embeddings
|
UMAP + HDBSCAN clustering
|
c-TF-IDF keywords per cluster
|
LLM batches inside each cluster -> Sentiment_LLM + raw post topics
|
Algorithmic topic consolidation inside each cluster
|
Second LLM topic consolidation -> compact canonical taxonomy
|
Attach canonical topics back to posts
|
Final LLM marketing call
|
clusters.json, analysis.json, structured_output.json, report files
```

The two sentiment fields mean different things:

- `Sentiment_Transformer`: local Hugging Face model sentiment.
- `Sentiment_LLM`: engagement/favorability label from the LLM. `positive` means the user is engaged or favorable; `negative` means clear criticism/rejection; `neutral` means unclear or informational.

## Main Modules

- `collector.yahoo_realtime`: Yahoo realtime scraper.
- `preprocessing.pipeline`: raw JSONL cleaning, deduplication, language metadata.
- `analysis.sentiment_transformer`: Hugging Face transformer sentiment.
- `analysis.embeddings`: multilingual semantic embeddings with `BAAI/bge-m3` by default.
- `analysis.clustering`: UMAP + HDBSCAN clustering and UMAP visualization coordinates. Defaults are `HDBSCAN_MIN_CLUSTER_SIZE=6`, `HDBSCAN_MIN_SAMPLES=2`, and `HDBSCAN_CLUSTER_SELECTION_METHOD=eom`.
- `analysis.text_preparation`: title/noise stripping, light reply/handle/decorative-token filtering, c-TF-IDF keyword extraction, representative posts.
- `analysis.post_llm_annotation`: LLM post batches, second LLM topic-taxonomy refinement, and final marketing summary.
- `analysis.topic_consolidation`: first-pass canonical topic IDs/labels per cluster using normalized labels plus embedding/lexical similarity.
- `analysis.structured_output`: Shiny/QueryChat-friendly JSON output.

## Topic Consolidation

Topic extraction has two stages:

1. First pass: each post can receive up to 3 raw topics from the LLM. The algorithm groups similar raw labels inside the same cluster.
2. Second pass: an LLM receives the preliminary topics for one cluster, not all posts, with topic counts, sentiment counts, aliases, keywords, and a few examples. It returns up to `LLM_TOPIC_CONSOLIDATION_MAX_TOPICS` final topics and lists which preliminary `source_topic_ids` should be merged into each final topic.

This keeps token usage lower than re-reading every post while reducing noisy micro-topics. The final `topics` in `structured_output.json` are the second-pass canonical taxonomy. Each final topic keeps `source_topic_ids` for traceability.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## LLM Provider

OpenAI:

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your_key"
$env:TOPIC_LABELING_MODEL="gpt-4.1-mini"
```

Gemini:

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_key"
$env:GEMINI_MODEL="gemini-2.5-flash"
$env:TOPIC_LABELING_MODEL="gemini-2.5-flash"
```

## Sentiment And LLM Settings

Default transformer model:

```powershell
$env:SENTIMENT_MODEL_NAME="LoneWolfgang/bert-for-japanese-twitter-sentiment"
```

LLM batch/config controls:

```powershell
$env:POST_TOPIC_LLM_BATCH_SIZE="15"
$env:LLM_TOPIC_CONSOLIDATION_MAX_TOPICS="6"
$env:LLM_TOPIC_CONSOLIDATION_MAX_INPUT_TOPICS="40"
$env:LLM_TEMPERATURE="0"
$env:LLM_TOP_P="1"
$env:GEMINI_THINKING_BUDGET="0"
```

The LLM post-batch call returns only:

- `source_row_index`
- `sentiment_llm`
- `topics`, max 3 per post, each with `topic_label` and `topic_description`

It does not return sentiment rationales.

Clustering controls:

```powershell
$env:HDBSCAN_MIN_CLUSTER_SIZE="6"
$env:HDBSCAN_MIN_SAMPLES="2"
$env:HDBSCAN_CLUSTER_SELECTION_METHOD="eom"
```

## Run The Pipeline

Scrape:

```powershell
python cli.py scrape --keywords "おっさんずラブ"
```

Preprocess latest raw file:

```powershell
python cli.py preprocess
```

Analyze latest processed file:

```powershell
python cli.py analyze
```

Analyze a specific processed JSONL file with free-tier settings:

```powershell
cd C:\Users\MANUE\Desktop\Scrapping
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_gemini_key_here"
$env:GEMINI_MODEL="gemini-2.5-flash"
$env:TOPIC_LABELING_MODEL="gemini-2.5-flash"
python cli.py analyze --free --input "data/processed/YOUR_FILE_processed.jsonl"
```

If the file is in the project root:

```powershell
python cli.py analyze --free --input ".\YOUR_FILE_processed.jsonl"
```

Generate report files:

```powershell
python cli.py report
```

## Free-Tier Mode

`python cli.py analyze --free` sets conservative defaults if they are not already set:

- `POST_TOPIC_LLM_BATCH_SIZE=15`
- `LLM_TOPIC_CONSOLIDATION_MAX_TOPICS=6`
- `LLM_TOPIC_CONSOLIDATION_MAX_INPUT_TOPICS=40`
- `HDBSCAN_MIN_CLUSTER_SIZE=6`
- `HDBSCAN_MIN_SAMPLES=2`
- `HDBSCAN_CLUSTER_SELECTION_METHOD=eom`
- `GEMINI_REQUESTS_PER_MINUTE=10`
- `GEMINI_TOKENS_PER_MINUTE=250000`
- `GEMINI_REQUESTS_PER_DAY=250`
- `GEMINI_MIN_SECONDS_BETWEEN_REQUESTS=6`
- `GEMINI_THINKING_BUDGET=0`
- `LLM_TEMPERATURE=0`
- `LLM_TOP_P=1`

The shared limiter stores its local daily counter at:

```text
data/reports/llm_rate_limit_state.json
```

## Outputs

The main dashboard file is:

```text
data/reports/structured_output.json
```

Top-level shape:

```json
{
  "metadata": {},
  "posts": [],
  "clusters": [],
  "daily_topic_metrics": [],
  "report_summary": {}
}
```

Important post fields:

```json
{
  "source_row_index": 0,
  "text": "Original post text",
  "cleaned_text": "Cleaned post text",
  "cluster_id": 2,
  "Sentiment_Transformer": "neutral",
  "Sentiment_LLM": "positive",
  "raw_topics_llm": [],
  "topics": [],
  "topic_id": "t003",
  "topic_label": "S2期待",
  "topic_description": "続編への期待",
  "umap_x": 1.23,
  "umap_y": -0.44
}
```

Important cluster fields:

```json
{
  "cluster_id": 2,
  "topic_id": "t003",
  "topic_label": "S2期待",
  "topics": [],
  "positive_count": 14,
  "neutral_count": 25,
  "negative_count": 3,
  "sentiment_llm_counts": {},
  "sentiment_transformer_counts": {},
  "top_keywords": ["Hulu", "配信", "放送"],
  "marketing_interpretation": "What this topic means",
  "risk_level": "low",
  "opportunity_level": "medium",
  "recommended_actions": [],
  "umap_centroid_x": 0.12,
  "umap_centroid_y": -0.08
}
```

Other files:

- `data/reports/clusters.json`
- `data/reports/analysis.json`
- `data/reports/report.md`
- `data/reports/report.html`
- `data/reports/clusters.csv`
