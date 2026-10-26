# feature_eng.py — 15-dimensional state vector matching the paper exactly
import numpy as np
import pandas as pd
from typing import Optional


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from the paper's Table 2.
    df must have columns: [date, close, volume]
    """
    df = df.copy()
    c = df["close"]
    v = df["volume"]

    # ── EMA10, EMA50 (normalized by close) ────────────────────────────────
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df["ema10_feat"] = ema10 / c - 1      # scale-invariant
    df["ema50_feat"] = ema50 / c - 1

    # ── RSI14 (recentered to [-1, 1]) ─────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    rsi14 = 100 - (100 / (1 + rs))
    df["rsi14_feat"] = rsi14 / 50 - 1    # [0,100] → [-1,1]

    # ── MACD histogram (normalized by close) ──────────────────────────────
    ema12     = c.ewm(span=12, adjust=False).mean()
    ema26     = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_feat"] = (macd_line - signal) / (c + 1e-10)

    # ── Bollinger Band B ──────────────────────────────────────────────────
    sma20     = c.rolling(20).mean()
    std20     = c.rolling(20).std()
    df["bollb_feat"] = (c - sma20) / (2 * std20 + 1e-10)

    # ── Log volume change ─────────────────────────────────────────────────
    df["logvol_feat"] = np.log(v / v.shift(1) + 1e-10)

    # ── Log return ────────────────────────────────────────────────────────
    df["log_return"] = np.log(c / c.shift(1) + 1e-10)

    # ── Daily return for environment ──────────────────────────────────────
    df["daily_return"] = c.pct_change()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar encodings (sin/cos for DoW, DoM, MoY)."""
    df = df.copy()
    dates = pd.DatetimeIndex(df["date"])

    dow = dates.dayofweek.astype(float)   # 0–6
    dom = dates.day.astype(float)         # 1–31
    moy = dates.month.astype(float)       # 1–12

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    df["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    df["moy_sin"] = np.sin(2 * np.pi * moy / 12)
    df["moy_cos"] = np.cos(2 * np.pi * moy / 12)

    return df


def build_state_matrix(df: pd.DataFrame,
                       sentiment: pd.Series,
                       warmup_days: int = 60) -> tuple:
    """
    Build the complete 15-dimensional state matrix and daily returns array.

    Returns:
        states:  np.ndarray [T, 15]  — state vectors (position=0, filled in env)
        returns: np.ndarray [T]      — daily returns (c_t/c_{t-1} - 1)
        dates:   list of dates

    State vector layout (matching paper's Table 2):
        0:  log_return
        1:  position (prev) ← set to 0 here, env updates it
        2:  dow_sin
        3:  dow_cos
        4:  dom_sin
        5:  dom_cos
        6:  moy_sin
        7:  moy_cos
        8:  ema10_feat
        9:  ema50_feat
        10: rsi14_feat
        11: macd_feat
        12: bollb_feat
        13: logvol_feat
        14: sentiment
    """
    df = compute_technical_indicators(df)
    df = add_calendar_features(df)

    # Align sentiment
    df["sentiment"] = df["date"].map(sentiment).fillna(0.0)

    # Drop warmup rows (cold-start artifacts from rolling indicators)
    df = df.iloc[warmup_days:].reset_index(drop=True)

    # Drop any remaining NaN rows
    feature_cols = [
        "log_return", "dow_sin", "dow_cos", "dom_sin", "dom_cos",
        "moy_sin", "moy_cos", "ema10_feat", "ema50_feat",
        "rsi14_feat", "macd_feat", "bollb_feat", "logvol_feat", "sentiment"
    ]
    df = df.dropna(subset=feature_cols + ["daily_return"])

    # Build state matrix
    states = np.zeros((len(df), 15), dtype=np.float32)
    states[:, 0]  = df["log_return"].values      # feature 1
    states[:, 1]  = 0.0                          # position (env fills in)
    states[:, 2]  = df["dow_sin"].values
    states[:, 3]  = df["dow_cos"].values
    states[:, 4]  = df["dom_sin"].values
    states[:, 5]  = df["dom_cos"].values
    states[:, 6]  = df["moy_sin"].values
    states[:, 7]  = df["moy_cos"].values
    states[:, 8]  = df["ema10_feat"].values
    states[:, 9]  = df["ema50_feat"].values
    states[:, 10] = df["rsi14_feat"].values
    states[:, 11] = df["macd_feat"].values
    states[:, 12] = df["bollb_feat"].values
    states[:, 13] = df["logvol_feat"].values
    states[:, 14] = df["sentiment"].values

    # Clip outliers (robust to extreme values in crypto)
    states = np.clip(states, -10, 10)

    returns = df["daily_return"].values.astype(np.float32)
    dates   = df["date"].tolist()

    return states, returns, dates


def compute_market_stats(returns: np.ndarray,
                         initial_value: float = 100_000) -> dict:
    """
    Compute Sharpe ratio, cumulative return, MDD for a returns series.
    Used for evaluation metrics.
    """
    values = initial_value * np.cumprod(1 + returns)
    cum_ret = (values[-1] / initial_value - 1) * 100

    daily_mean = np.mean(returns)
    daily_std  = np.std(returns) + 1e-10
    sharpe     = (daily_mean / daily_std) * np.sqrt(252)

    peak   = np.maximum.accumulate(values)
    dd     = (values - peak) / peak
    mdd    = abs(dd.min()) * 100

    ann_vol = daily_std * np.sqrt(252) * 100

    return {
        "cumulative_return": round(cum_ret, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown":      round(mdd, 2),
        "ann_volatility":    round(ann_vol, 2),
    }
