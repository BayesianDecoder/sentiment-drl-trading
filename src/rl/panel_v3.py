"""
Builds the "v3" feature panel for the 10-stock portfolio env: takes the existing real data in
rl_panel.csv (real prices, real Qwen sentiment) and adds everything from the paper-inspired
improvement plan (see EXPERIMENT_LOG.md) that ISN'T already in v1:

  - Price-normalized technical indicators (EMA10/50 relative to close, RSI recentered to
    [-1,1], MACD/close, BollingerB) instead of v1's raw-scale MACD/RSI/CCI/ADX -- v1's shared
    network sees 10 stocks at wildly different price levels (ITC ~Rs170 vs Maruti ~Rs6650)
    through unnormalized indicators, which is a real handicap for one shared policy.
  - 6 calendar cyclical encodings (day-of-week/month/year as sin/cos) -- global, not per-asset,
    since they don't depend on which stock you're looking at.
  - Cross-sectional relative return: each stock's return minus that day's mean return across
    all 10 -- something the paper doesn't need (it's single-asset) but is arguably more useful
    here than any single stock's own indicators in isolation, since this is a portfolio problem.
  - A portfolio-level volatility-regime feature (rolling realized vol of the equal-weight
    benchmark) -- gives the agent explicit information to detect a bull->bear shift, rather
    than hoping it infers regime change implicitly (this is the paper's own diagnosed weakness
    in Section 6.2, applied constructively here).
  - Sentiment smoothing: a 3-day rolling mean (real sentiment is sparse -- 91 real headlines
    across ~1200 ticker-days -- so a single day's value is mostly a bootstrap-resampled filler;
    smoothing reduces noise from that sparsity) plus a "days since last real news" feature so
    the agent can tell fresh real signal from stale/resampled filler.
  - Holdings expressed as a fraction of portfolio value, not raw share count -- v1 fed raw
    share counts into the shared network, which have the same cross-asset scale problem as
    unnormalized indicators (1000 shares of Rs170 ITC vs 10 shares of Rs6650 Maruti aren't
    comparable numbers). Handled in trading_env_v3.py, not here (depends on live cash/holdings).
"""
import numpy as np
import pandas as pd

from src.paper_replication.features import add_technical_indicators
from src.rl import config as C

PANEL_PATH = C.RL_PANEL_PATH
SENTIMENT_SMOOTH_WINDOW = 3
DAYS_SINCE_NEWS_CAP = 20.0


def build_v3_panel() -> pd.DataFrame:
    raw = pd.read_csv(PANEL_PATH, parse_dates=["date"])

    per_ticker = []
    for ticker in C.TICKERS:
        sub = raw[raw["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        feat = add_technical_indicators(sub[["date", "close", "volume"]].copy())

        has_real = sub["has_real_sentiment"].astype(bool) if "has_real_sentiment" in sub.columns else (sub["sentiment"] != 0)
        real_sent = sub["sentiment"].where(has_real)
        feat["sentiment_smoothed"] = real_sent.rolling(SENTIMENT_SMOOTH_WINDOW, min_periods=1).mean()
        feat["sentiment_smoothed"] = feat["sentiment_smoothed"].ffill().fillna(0.0)

        last_news_idx = pd.Series(np.where(has_real, np.arange(len(sub)), np.nan))
        last_news_idx = last_news_idx.ffill()
        days_since = (np.arange(len(sub)) - last_news_idx).fillna(DAYS_SINCE_NEWS_CAP)
        feat["days_since_news"] = np.minimum(days_since, DAYS_SINCE_NEWS_CAP) / DAYS_SINCE_NEWS_CAP

        feat["ticker"] = ticker
        feat["raw_close"] = sub["close"].to_numpy()
        per_ticker.append(feat)

    panel = pd.concat(per_ticker, ignore_index=True)

    daily_ret = panel.pivot(index="date", columns="ticker", values="log_return")
    cross_sectional_mean = daily_ret.mean(axis=1)
    rel_return = daily_ret.sub(cross_sectional_mean, axis=0)
    rel_long = rel_return.reset_index().melt(id_vars="date", value_name="cross_sectional_rel_return")
    panel = panel.merge(rel_long, on=["date", "ticker"], how="left")
    panel["cross_sectional_rel_return"] = panel["cross_sectional_rel_return"].fillna(0.0)

    bh_value = (1.0 + daily_ret.mean(axis=1)).cumprod()
    bh_ret = bh_value.pct_change().fillna(0.0)
    vol_regime = bh_ret.rolling(20, min_periods=5).std().fillna(0.0)
    vol_df = vol_regime.reset_index().rename(columns={0: "vol_regime"})
    vol_df.columns = ["date", "vol_regime"]
    panel = panel.merge(vol_df, on="date", how="left")
    panel["vol_regime"] = panel["vol_regime"].fillna(0.0)

    return panel


PER_ASSET_COLS = [
    "log_return", "ema10_rel", "ema50_rel", "rsi14_norm", "macd_norm", "bollinger_b",
    "log_vol_change", "sentiment_smoothed", "days_since_news", "cross_sectional_rel_return",
]
GLOBAL_COLS = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "moy_sin", "moy_cos", "vol_regime"]
