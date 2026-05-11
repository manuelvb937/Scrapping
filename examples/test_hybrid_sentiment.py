"""Small demo for transformer and hybrid sentiment analysis.

Run from the repository root after installing dependencies:

    python examples/test_hybrid_sentiment.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.sentiment_transformer import HybridSentimentAnalyzer


EXAMPLE_POSTS = [
    "このドラマ最高すぎる",
    "本日22時から放送開始",
    "公式またこれか…",
    "尊すぎて死んだ",
    "しんどいけど最高",
    "キャスト変更はちょっと微妙",
]


def main() -> None:
    analyzer = HybridSentimentAnalyzer()
    results = analyzer.analyze(EXAMPLE_POSTS)

    for item in results:
        transformer = item["transformer_sentiment"]
        print(f"text: {item['text']}")
        print(f"  transformer label: {transformer['label']}")
        print(f"  confidence: {transformer['confidence']:.2f}")
        print(f"  LLM fallback triggered: {item['llm_review'] is not None}")
        print(f"  final sentiment: {item['final_sentiment']}")


if __name__ == "__main__":
    main()
