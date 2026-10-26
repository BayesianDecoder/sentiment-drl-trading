"""
Builds the train/validation/test panels for one asset, matching the paper's Section 4.1 splits:
  - train: before 2023-01-01
  - validation: 2023-01-01 to 2024-08-01
  - test: the CLEF Task3 evaluation window (2025-08-01 onward), using the official CLEF
    Task3 Hugging Face dataset's own price series for test-day closes (matching the paper's
    "the official CLEF Task3 evaluation set ... provides daily closing prices ... for the
    testing phase"), with volume backfilled from yfinance for the same dates.

Sentiment: test-period sentiment comes from the user's fine-tuned Qwen adapter scoring the
CLEF dataset's own pre-linked daily news (scripts/score_clef_test_sentiment.py). Train/val
sentiment requires Brianferrell787/financial-news-multisource, which is gated on Hugging Face
pending the user's access request -- until granted, train/val sentiment is filled with 0.0
(neutral) so the rest of the pipeline can be built and dry-run end to end; this is flagged
wherever it's used and must not be reported as a final result.
"""
import os

import pandas as pd

from src.paper_replication.features import add_technical_indicators

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw", "paper_replication")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed", "paper_replication")

TRAIN_END = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2024-08-01")
WARMUP_DAYS = 60


def _load_yf_prices(asset: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{'TSLA' if asset == 'TSLA' else 'BTC'}_prices.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})[["close", "volume"]]
    df["date"] = df.index
    return df.reset_index(drop=True)


def _load_clef_test(asset: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "clef_task3", "data", f"{asset}-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "prices"]].rename(columns={"prices": "close"})


def _load_test_sentiment(asset: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{asset}_test_sentiment.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "sentiment"])
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_panels(asset: str, sentiment_available_train_val: bool = False):
    yf = _load_yf_prices(asset)
    yf_train_val = yf[yf["date"] < VAL_END].copy()

    train_val_full = yf_train_val.copy()
    if sentiment_available_train_val:
        sent_path = os.path.join(PROCESSED_DIR, f"{asset}_trainval_sentiment.csv")
        sent = pd.read_csv(sent_path, parse_dates=["date"])
        train_val_full = train_val_full.merge(sent, on="date", how="left")
        train_val_full["sentiment"] = train_val_full["sentiment"].fillna(0.0)
    else:
        train_val_full["sentiment"] = 0.0

    train_val_full = add_technical_indicators(train_val_full)
    train_val_full["position"] = 0.0

    train = train_val_full[train_val_full["date"] < TRAIN_END].reset_index(drop=True)
    val = train_val_full[train_val_full["date"] >= TRAIN_END].reset_index(drop=True)
    if len(train) > WARMUP_DAYS:
        train = train.iloc[WARMUP_DAYS:].reset_index(drop=True)

    clef_test = _load_clef_test(asset)
    warmup_prices = yf_train_val.tail(WARMUP_DAYS)[["date", "close"]]
    test_price_series = pd.concat([warmup_prices, clef_test[["date", "close"]]], ignore_index=True)
    test_price_series = test_price_series.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    yf_vol = yf[["date", "volume"]]
    test_price_series = test_price_series.merge(yf_vol, on="date", how="left")
    test_price_series["volume"] = test_price_series["volume"].ffill().fillna(0.0).infer_objects(copy=False)

    test_sent = _load_test_sentiment(asset)
    test_price_series = test_price_series.merge(test_sent, on="date", how="left")
    test_price_series["sentiment"] = test_price_series["sentiment"].fillna(0.0).infer_objects(copy=False)

    test_full = add_technical_indicators(test_price_series)
    test_full["position"] = 0.0
    test = test_full[test_full["date"] >= clef_test["date"].min()].reset_index(drop=True)

    return train, val, test
