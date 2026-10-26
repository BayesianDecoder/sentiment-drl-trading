# data_pipeline.py — Download and preprocess price + news data
import pandas as pd
import numpy as np
import yfinance as yf
from datasets import load_dataset
from pathlib import Path
import hashlib
import re
import json
import warnings
warnings.filterwarnings("ignore")

from config import TRAIN_END, VAL_START, VAL_END, TEST_START, WARMUP_DAYS

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ── Price data ─────────────────────────────────────────────────────────────

def download_prices(ticker: str, start: str = "2010-01-01",
                    end: str = "2026-01-01") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance."""
    cache_path = CACHE_DIR / f"prices_{ticker}.parquet"
    if cache_path.exists():
        print(f"  Loading cached prices for {ticker}")
        return pd.read_parquet(cache_path)

    print(f"  Downloading prices for {ticker} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index = pd.to_datetime(df.index).normalize()
    df = df.dropna()
    df.to_parquet(cache_path)
    print(f"  ✅ {len(df)} trading days downloaded")
    return df


# ── News data ──────────────────────────────────────────────────────────────

def load_news(asset: str) -> pd.DataFrame:
    """
    Load financial news from HuggingFace dataset.
    Filters for asset-relevant articles using regex (matching paper).
    """
    cache_path = CACHE_DIR / f"news_{asset}.parquet"
    if cache_path.exists():
        print(f"  Loading cached news for {asset}")
        return pd.read_parquet(cache_path)

    print(f"  Loading news dataset for {asset} ...")

    # Load Brian Ferrell's Financial News Multi-Source dataset
    try:
        dataset = load_dataset(
            "Brianferrell787/financial-news-multisource",
            split="train", trust_remote_code=True
        )
        df = dataset.to_pandas()
    except Exception as e:
        print(f"  ⚠️  Could not load HF dataset: {e}")
        print("  Creating empty news DataFrame")
        df = pd.DataFrame(columns=["date", "text", "publisher"])

    # Standardize columns
    text_col = next((c for c in ["text", "content", "article", "body"] if c in df.columns), None)
    date_col = next((c for c in ["date", "published_at", "publishedAt", "timestamp"] if c in df.columns), None)
    pub_col  = next((c for c in ["publisher", "source", "author"] if c in df.columns), None)

    if text_col and date_col:
        df = df.rename(columns={text_col: "text", date_col: "date"})
        if pub_col:
            df = df.rename(columns={pub_col: "publisher"})
        else:
            df["publisher"] = ""
    else:
        df["text"] = ""
        df["date"] = pd.Timestamp("2020-01-01")
        df["publisher"] = ""

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "text"])
    df["text"] = df["text"].astype(str)

    # ── Asset keyword filtering (matching paper) ──────────────────────────
    keywords = _get_keywords(asset)
    pattern  = "|".join(keywords)

    def is_relevant(text: str) -> bool:
        text_lower = text.lower()
        # At least 2 mentions anywhere, OR 1 mention in first 100 words (lede)
        full_count = len(re.findall(pattern, text_lower))
        lede_count = len(re.findall(pattern, " ".join(text_lower.split()[:100])))
        return full_count >= 2 or lede_count >= 1

    df = df[df["text"].apply(is_relevant)].copy()
    print(f"  ✅ {len(df)} relevant articles after filtering")

    # ── Deduplication via MD5 hash ────────────────────────────────────────
    def md5_hash(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]", "", text.lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    df["hash"] = df["text"].apply(md5_hash)
    # Keep article with publisher metadata when duplicates exist
    df["has_publisher"] = df["publisher"].str.len() > 0
    df = df.sort_values("has_publisher", ascending=False)
    df = df.drop_duplicates(subset="hash", keep="first")
    df = df.drop(columns=["hash", "has_publisher"])

    # ── Temporal alignment ────────────────────────────────────────────────
    if asset.upper() == "TSLA":
        # Equity: after 16:00 UTC → next trading day
        df["trade_date"] = df["date"].apply(
            lambda d: (d + pd.Timedelta(days=1)).date()
            if d.hour >= 16 else d.date()
        )
    else:
        # BTC: 24/7, anchor to UTC calendar date
        df["trade_date"] = df["date"].dt.date

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # Prepend publisher info for structured context (matching paper)
    def format_article(row) -> str:
        if row["publisher"] and len(row["publisher"]) > 0:
            return f"[Publisher: {row['publisher']}] {row['text']}"
        return row["text"]

    df["formatted_text"] = df.apply(format_article, axis=1)
    df = df[["trade_date", "formatted_text"]].rename(columns={"formatted_text": "text"})

    df.to_parquet(cache_path)
    return df


def _get_keywords(asset: str) -> list:
    mapping = {
        "TSLA":    ["tesla", "tsla", "elon musk"],
        "BTC-USD": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    }
    return mapping.get(asset.upper(), [asset.lower()])


# ── Combine prices + news ──────────────────────────────────────────────────

def build_dataset(asset: str, price_start: str = "2010-01-01") -> pd.DataFrame:
    """
    Build a combined daily DataFrame with prices, volume, and article lists.
    Returns one row per trading day.
    """
    print(f"\n📊 Building dataset for {asset}")

    # Prices
    prices = download_prices(asset, start=price_start)

    # News
    news = load_news(asset)

    # Group news by trading day → list of articles
    if not news.empty:
        daily_news = (
            news.groupby("trade_date")["text"]
            .apply(list)
            .reset_index()
            .rename(columns={"trade_date": "date", "text": "articles"})
        )
        daily_news["date"] = pd.to_datetime(daily_news["date"])
    else:
        daily_news = pd.DataFrame(columns=["date", "articles"])

    # Merge prices + news on date
    prices.index.name = "date"
    prices = prices.reset_index()
    prices["date"] = pd.to_datetime(prices["date"])

    df = prices.merge(daily_news, on="date", how="left")
    df["articles"] = df["articles"].apply(lambda x: x if isinstance(x, list) else [])

    # Compute returns
    df["return"]     = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["return"])

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  ✅ Dataset: {len(df)} days | {df['articles'].apply(len).sum()} total articles")
    return df


def split_dataset(df: pd.DataFrame):
    """Split into train / val / test matching the paper."""
    train = df[df["date"] <= TRAIN_END].copy()
    val   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    test  = df[df["date"] >= TEST_START].copy()

    print(f"\n📅 Data splits:")
    print(f"  Train : {len(train)} days ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Val   : {len(val)}   days ({val['date'].min().date()} → {val['date'].max().date()})")
    print(f"  Test  : {len(test)}  days ({test['date'].min().date()} → {test['date'].max().date()})")
    return train, val, test


if __name__ == "__main__":
    for asset in ["TSLA", "BTC-USD"]:
        df = build_dataset(asset)
        train, val, test = split_dataset(df)
        print(f"\n{asset} dataset ready.")
