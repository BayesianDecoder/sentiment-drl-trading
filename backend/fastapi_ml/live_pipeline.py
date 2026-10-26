"""
live_pipeline.py — a real background polling loop, not a "live" label on static data.

Every POLL_INTERVAL_SECONDS, for each of the 10 tickers:
  1. Fetches recent news via yfinance, scores any headline not already seen (deduped by a
     hash of ticker+headline+date) with the trained FinBERT-LoRA model.
  2. Fetches the latest intraday price (yfinance 1-minute bars; falls back to last close
     when the market is shut).
  3. Recomputes technical indicators using real historical data + this latest price appended
     as "today so far", then feeds the resulting state through the trained A2C model + its
     saved VecNormalize stats to get a genuine current recommendation -- not the historical
     backtest replayed, an actual fresh forward pass on today's real data.

Exposes the result via `get_live_state()` / `get_live_status()`, polled by FastAPI endpoints.
"""
import asyncio
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, time as dtime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
import yfinance as yf
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

log = logging.getLogger("live_pipeline")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(ROOT, "data", "processed", "rl_panel.csv")
MODEL_PATH = os.path.join(ROOT, "notebooks", "saved_models", "A2C_sentiment.zip")
NORM_PATH = os.path.join(ROOT, "notebooks", "saved_models", "A2C_sentiment_vecnormalize.pkl")

TICKERS = ["INFY.NS", "RELIANCE.NS", "TCS.NS", "SBIN.NS", "MARUTI.NS",
           "WIPRO.NS", "TMPV.NS", "LT.NS", "ITC.NS", "SUNPHARMA.NS"]
INDICATORS = ["macd", "rsi_14", "cci_20", "adx_14"]
POLL_INTERVAL_SECONDS = 90
MAX_LIVE_NEWS = 50
IST = pytz.timezone("Asia/Kolkata")

_state = {
    "last_updated": None,
    "market_open": False,
    "prices": {},
    "sentiment": {},
    "recommended_actions": {},
    "news": [],
}
_seen_news_hashes = set()
_model = None
_vecnorm = None
_finbert_model = None
_finbert_tokenizer = None
_finbert_id2label = None


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:   # Sat/Sun
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 30)


def set_finbert(model, tokenizer, id2label):
    """Called once from main.py's startup, after it loads FinBERT-LoRA for /predict -- avoids
    loading the same ~440MB model into memory twice."""
    global _finbert_model, _finbert_tokenizer, _finbert_id2label
    _finbert_model, _finbert_tokenizer, _finbert_id2label = model, tokenizer, id2label


def _load_models():
    global _model, _vecnorm
    if _finbert_model is None:
        raise RuntimeError("set_finbert() must be called before the live pipeline starts polling")

    _model = A2C.load(MODEL_PATH)
    dummy_panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])

    def _init():
        from src.rl.trading_env import PortfolioTradingEnv
        return Monitor(PortfolioTradingEnv(dummy_panel))
    dummy_env = DummyVecEnv([_init])
    _vecnorm = VecNormalize.load(NORM_PATH, dummy_env)
    log.info("Live pipeline: FinBERT-LoRA + A2C models loaded")


@torch.no_grad()
def _score_headline(text: str) -> float:
    inputs = _finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    probs = torch.softmax(_finbert_model(**inputs).logits, dim=-1)[0]
    pos_idx = [k for k, v in _finbert_id2label.items() if v == "positive"][0]
    neg_idx = [k for k, v in _finbert_id2label.items() if v == "negative"][0]
    return float(probs[pos_idx] - probs[neg_idx])


def _fetch_live_news(ticker: str) -> list[dict]:
    new_items = []
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        log.warning(f"news fetch failed for {ticker}: {e}")
        return new_items

    for item in news[:15]:
        content = item.get("content", item)
        title = content.get("title")
        pub_date = content.get("pubDate") or content.get("displayTime")
        if not title or not pub_date:
            continue
        h = hashlib.md5(f"{ticker}|{title}".encode()).hexdigest()
        if h in _seen_news_hashes:
            continue
        _seen_news_hashes.add(h)
        score = _score_headline(title)
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        new_items.append({
            "ticker": ticker, "headline": title, "date": str(pd.to_datetime(pub_date)),
            "sentiment_score": round(score, 4), "sentiment_label": label,
            "fetched_at": datetime.now(IST).isoformat(),
        })
    return new_items


def _fetch_live_price(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if hist.empty:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"price fetch failed for {ticker}: {e}")
        return None


def _build_live_observation(latest_prices: dict, latest_sentiment: dict) -> np.ndarray | None:
    """Appends today's live price as one more row onto real history, recomputes indicators,
    and builds the same 71-dim state the trained model expects."""
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    today = pd.Timestamp.now(IST).normalize().tz_localize(None)

    rows = []
    for ticker in TICKERS:
        if ticker not in latest_prices or latest_prices[ticker] is None:
            return None
        last_row = panel[panel["ticker"] == ticker].sort_values("date").iloc[-1]
        rows.append({
            "date": today, "ticker": ticker,
            "open": latest_prices[ticker], "high": latest_prices[ticker],
            "low": latest_prices[ticker], "close": latest_prices[ticker],
            "volume": last_row["volume"],
        })
    today_df = pd.DataFrame(rows)
    panel = pd.concat([panel[["date", "ticker", "open", "high", "low", "close", "volume"]], today_df], ignore_index=True)

    out = []
    for ticker, df in panel.groupby("ticker"):
        df = df.sort_values("date").copy()
        macd = ta.macd(df["close"])
        df["macd"] = macd.iloc[:, 0] if macd is not None else np.nan
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx_14"] = adx.iloc[:, 0] if adx is not None else np.nan
        df["sentiment"] = latest_sentiment.get(ticker, 0.0)
        out.append(df.tail(1))
    today_full = pd.concat(out, ignore_index=True)

    cash = 1_000_000.0   # a nominal cash baseline purely to shape the observation for inference
    holdings = np.zeros(len(TICKERS))
    prices = np.array([latest_prices[t] for t in TICKERS], dtype=np.float32)
    parts = [np.array([cash], dtype=np.float32), holdings.astype(np.float32), prices]
    for col in INDICATORS + ["sentiment"]:
        parts.append(today_full.set_index("ticker").loc[TICKERS, col].to_numpy(dtype=np.float32))
    obs = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)
    return obs


async def poll_once():
    global _state
    if _model is None:
        _load_models()

    market_open = is_market_open()
    all_new_news = []
    latest_prices = {}
    latest_sentiment = dict(_state["sentiment"])

    for ticker in TICKERS:
        price = _fetch_live_price(ticker)
        if price is not None:
            latest_prices[ticker] = price
        new_news = _fetch_live_news(ticker)
        if new_news:
            latest_sentiment[ticker] = np.mean([n["sentiment_score"] for n in new_news])
        all_new_news.extend(new_news)

    recommended_actions = {}
    if len(latest_prices) == len(TICKERS):
        try:
            obs = _build_live_observation(latest_prices, latest_sentiment)
            if obs is not None:
                norm_obs = _vecnorm.normalize_obs(obs).reshape(1, -1)
                action, _ = _model.predict(norm_obs, deterministic=True)
                for i, ticker in enumerate(TICKERS):
                    a = float(action[0][i])
                    recommended_actions[ticker] = "buy" if a > 0.1 else "sell" if a < -0.1 else "hold"
        except Exception as e:
            log.warning(f"live recommendation failed: {e}")

    _state = {
        "last_updated": datetime.now(IST).isoformat(),
        "market_open": market_open,
        "prices": latest_prices,
        "sentiment": latest_sentiment,
        "recommended_actions": recommended_actions,
        "news": (all_new_news + _state["news"])[:MAX_LIVE_NEWS],
    }
    log.info(f"Live poll complete: {len(all_new_news)} new headlines, market_open={market_open}")


async def poll_loop():
    while True:
        try:
            await poll_once()
        except Exception as e:
            log.error(f"Live poll cycle failed: {e}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def get_live_state():
    return _state


def get_live_status():
    return {
        "last_updated": _state["last_updated"],
        "market_open": _state["market_open"],
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
    }
