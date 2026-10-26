"""
data_pipeline.py — Part B (RL trading agents)

Builds the panel dataset the trading environment reads:
  date, ticker, open, high, low, close, volume, macd, rsi_14, cci_20, adx_14, sentiment

Price history comes from yfinance (years of daily OHLCV, fully real).
Sentiment comes from yfinance's recent-news feed per ticker, scored with the
project's own trained FinBERT-LoRA adapter — real, but only covers a recent
window (yfinance doesn't serve historical dated news). Days outside that
window get a neutral (0.0) placeholder; see config.py for the reasoning.
"""
import logging
import time

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from src.rl import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def fetch_price_panel(tickers=C.TICKERS, years_back=C.PRICE_YEARS_BACK) -> pd.DataFrame:
    """Long-format daily OHLCV panel: one row per (date, ticker)."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years_back)

    frames = []
    for ticker in tickers:
        log.info(f"Fetching price history: {ticker}")
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist.empty:
            log.warning(f"  no data returned for {ticker}, skipping")
            continue
        hist = hist.reset_index()
        hist["ticker"] = ticker
        hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
        hist = hist.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                     "Close": "close", "Volume": "volume"})
        frames.append(hist[["date", "ticker", "open", "high", "low", "close", "volume"]])
        time.sleep(0.3)   # be polite to the API

    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

    # yfinance occasionally drops a single ticker's row for one date (a halt, or lagged data
    # availability for the most recent day) while other tickers still have that date -- when
    # later pivoted into a shared (date x ticker) price matrix, that gap becomes a real NaN,
    # which poisons the ENTIRE portfolio value (holdings . prices includes a NaN term even for
    # tickers holding 0 shares, since 0 * NaN = NaN, not 0). Reindex every ticker onto the full
    # union of trading dates and forward-fill (assume price unchanged on a no-data day) so the
    # panel is always rectangular.
    all_dates = panel["date"].unique()
    filled = []
    for ticker, df in panel.groupby("ticker"):
        df = df.set_index("date").reindex(all_dates).sort_index()
        df["ticker"] = ticker
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()
        filled.append(df)
    panel = pd.concat(filled).reset_index().rename(columns={"index": "date"})
    panel = panel.dropna(subset=["close"])   # a ticker's leading dates before its own history starts
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info(f"Price panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
              f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    return panel


def add_technical_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds macd, rsi_14, cci_20, adx_14 per ticker (each computed on that ticker's own series)."""
    out = []
    for ticker, df in panel.groupby("ticker"):
        df = df.sort_values("date").copy()
        macd = ta.macd(df["close"])
        df["macd"] = macd.iloc[:, 0] if macd is not None else np.nan
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        cci = ta.cci(df["high"], df["low"], df["close"], length=20)
        df["cci_20"] = cci
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx_14"] = adx.iloc[:, 0] if adx is not None else np.nan
        out.append(df)

    result = pd.concat(out, ignore_index=True)
    # Indicators need a warmup window (e.g. ADX needs ~2x its length) — drop those NaN rows.
    result = result.dropna(subset=C.INDICATORS).reset_index(drop=True)
    log.info(f"After indicators + warmup drop: {len(result)} rows")
    return result


def _load_finbert_lora():
    base = AutoModelForSequenceClassification.from_pretrained(C.FINBERT_BASE_MODEL, num_labels=3)
    model = PeftModel.from_pretrained(base, C.FINBERT_ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(C.FINBERT_TOKENIZER_DIR)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}   # e.g. {0:'positive',1:'negative',2:'neutral'}
    return model, tokenizer, id2label


@torch.no_grad()
def _score_headlines(model, tokenizer, id2label, texts: list[str]) -> list[float]:
    """Returns a signed sentiment score per headline: P(positive) - P(negative), in [-1, 1]."""
    if not texts:
        return []
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    probs = torch.softmax(model(**inputs).logits, dim=-1)
    pos_idx = [k for k, v in id2label.items() if v == "positive"][0]
    neg_idx = [k for k, v in id2label.items() if v == "negative"][0]
    scores = (probs[:, pos_idx] - probs[:, neg_idx]).tolist()
    return scores


def fetch_and_score_news_sentiment(tickers=C.TICKERS, use_qwen: bool = True) -> pd.DataFrame:
    """
    Pulls each ticker's recent news via yfinance and scores it.

    use_qwen=True (default): scores with the project's own fine-tuned Qwen2.5-3B LoRA adapter
    (vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned) instead of FinBERT-LoRA. Only the small set
    of real headlines fetched here get scored live (the RL pipeline bootstrap-resamples these
    across the rest of the training window), so the per-call cost of a larger model is fine.

    Returns (date, ticker, sentiment) aggregated by mean when a ticker has
    multiple headlines on the same day.
    """
    if use_qwen:
        from src.rl import qwen_sentiment
        qwen_sentiment.load_model()
        log.info("Scoring news sentiment with fine-tuned Qwen2.5-3B (local MLX adapter)")
        score_fn = None   # bound per-ticker below, since Qwen scoring needs the company name too
    else:
        model, tokenizer, id2label = _load_finbert_lora()
        log.info(f"FinBERT-LoRA label mapping: {id2label}")
        score_fn = lambda texts: _score_headlines(model, tokenizer, id2label, texts)

    rows = []
    for ticker in tickers:
        log.info(f"Fetching news: {ticker}")
        try:
            news = yf.Ticker(ticker).news or []
        except Exception as e:
            log.warning(f"  news fetch failed for {ticker}: {e}")
            continue

        news = news[: C.MAX_NEWS_PER_TICKER]
        texts, dates = [], []
        for item in news:
            content = item.get("content", item)   # yfinance nests fields under "content"
            title = content.get("title")
            pub_date = content.get("pubDate") or content.get("displayTime")
            if not title or not pub_date:
                continue
            texts.append(title)
            dates.append(pd.to_datetime(pub_date).tz_localize(None).normalize())

        if not texts:
            continue
        if use_qwen:
            from src.rl import qwen_sentiment
            scores = qwen_sentiment.score_headlines_batch(texts, ticker)
        else:
            scores = score_fn(texts)
        for date, score in zip(dates, scores):
            rows.append({"date": date, "ticker": ticker, "sentiment": score})
        time.sleep(0.3)

    if not rows:
        log.warning("No news sentiment collected at all — sentiment column will be all-neutral.")
        return pd.DataFrame(columns=["date", "ticker", "sentiment"])

    df = pd.DataFrame(rows)
    daily = df.groupby(["date", "ticker"], as_index=False)["sentiment"].mean()
    log.info(f"Collected sentiment for {daily['ticker'].nunique()} tickers, "
              f"{daily['date'].min().date()} -> {daily['date'].max().date()} ({len(daily)} ticker-days)")
    return daily


def build_panel(save: bool = True) -> pd.DataFrame:
    price = fetch_price_panel()
    price = add_technical_indicators(price)

    sentiment = fetch_and_score_news_sentiment()
    if save:
        sentiment.to_csv(C.RL_SENTIMENT_PATH, index=False)

    panel = price.merge(sentiment, on=["date", "ticker"], how="left")
    # has_real_sentiment must be checked BEFORE fillna, and on the merged column itself
    # (not two independent .isin() checks against date/ticker separately, which would
    # wrongly mark a row "real" whenever ITS date and ITS ticker each appear ANYWHERE in
    # the sentiment table, even for a (date, ticker) pair that was never actually scored).
    panel["has_real_sentiment"] = panel["sentiment"].notna()
    panel["sentiment"] = panel["sentiment"].fillna(C.SENTIMENT_NEUTRAL_FILL)

    # Bootstrap column: real value where we have one, else a random draw from the empirical
    # distribution of real scores (seeded, so it's reproducible). Used only for TRAIN/VAL splits
    # (see train_agents.load_splits) so the sentiment-fused agent actually sees varied, realistic
    # sentiment values while learning — with a constant 0.0 fill it never encounters a non-zero
    # sentiment input until the test period, so it has no chance to learn to use the feature at all.
    # TEST keeps the real+0-fill column untouched, since that's the actual held-out evaluation.
    rng = np.random.RandomState(C.SEED if hasattr(C, "SEED") else 42)
    real_values = panel.loc[panel["has_real_sentiment"], "sentiment"].to_numpy()
    if len(real_values) > 0:
        n_missing = (~panel["has_real_sentiment"]).sum()
        bootstrap_fill = rng.choice(real_values, size=n_missing, replace=True)
        panel["sentiment_bootstrap"] = panel["sentiment"].copy()
        panel.loc[~panel["has_real_sentiment"], "sentiment_bootstrap"] = bootstrap_fill
    else:
        panel["sentiment_bootstrap"] = panel["sentiment"]

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    n_real = panel["has_real_sentiment"].sum()
    log.info(f"Final panel: {len(panel)} rows | {n_real} rows ({n_real/len(panel)*100:.1f}%) have real news sentiment")

    if save:
        panel.to_csv(C.RL_PANEL_PATH, index=False)
        log.info(f"Saved panel to {C.RL_PANEL_PATH}")
    return panel


if __name__ == "__main__":
    build_panel()
