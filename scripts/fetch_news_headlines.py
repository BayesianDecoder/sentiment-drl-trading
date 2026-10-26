"""
Fetches each ticker's recent news via yfinance (real headline text this time, not just an
aggregated sentiment score) and scores each headline individually with the trained FinBERT-LoRA
model. This feeds the dashboard's "news received" panel.

Run: .venv/bin/python scripts/fetch_news_headlines.py
Writes: data/processed/news_headlines.json
"""
import os
import json
import time

import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rl import config as C
from src.rl.data_pipeline import _load_finbert_lora, _score_headlines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data", "processed", "news_headlines.json")


def main():
    model, tokenizer, id2label = _load_finbert_lora()

    records = []
    for ticker in C.TICKERS:
        print(f"Fetching news: {ticker}")
        try:
            news = yf.Ticker(ticker).news or []
        except Exception as e:
            print(f"  failed: {e}")
            continue

        news = news[: C.MAX_NEWS_PER_TICKER]
        texts, dates, urls = [], [], []
        for item in news:
            content = item.get("content", item)
            title = content.get("title")
            pub_date = content.get("pubDate") or content.get("displayTime")
            url = (content.get("canonicalUrl") or {}).get("url", "")
            if not title or not pub_date:
                continue
            texts.append(title)
            dates.append(pd.to_datetime(pub_date).tz_localize(None).isoformat())
            urls.append(url)

        if not texts:
            continue
        scores = _score_headlines(model, tokenizer, id2label, texts)
        for text, date, url, score in zip(texts, dates, urls, scores):
            label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
            records.append({
                "ticker": ticker, "headline": text, "date": date, "url": url,
                "sentiment_score": round(score, 4), "sentiment_label": label,
            })
        time.sleep(0.3)

    records.sort(key=lambda r: r["date"], reverse=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} headlines to {OUT_PATH}")


if __name__ == "__main__":
    main()
