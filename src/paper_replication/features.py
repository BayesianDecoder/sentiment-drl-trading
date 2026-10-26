"""
Feature engineering replicating Table 2 of the paper: 14 market dimensions + previous
position = 15 total state inputs. All formulas match the paper exactly (EMA10/50 relative
to close, RSI14 re-centered to [-1,1], MACD normalized by close, BollingerB, log volume
change, cyclical calendar encodings, sentiment as P(+) - P(-)).
"""
import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: date, close, volume. Returns df with all Table 2 columns added."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    c = df["close"]

    df["log_return"] = np.log(c / c.shift(1))

    dow = pd.to_datetime(df["date"]).dt.dayofweek
    dom = pd.to_datetime(df["date"]).dt.day
    moy = pd.to_datetime(df["date"]).dt.month
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    df["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    df["moy_sin"] = np.sin(2 * np.pi * moy / 12)
    df["moy_cos"] = np.cos(2 * np.pi * moy / 12)

    ema10 = _ema(c, 10)
    ema50 = _ema(c, 50)
    df["ema10_rel"] = ema10 / c - 1
    df["ema50_rel"] = ema50 / c - 1

    df["rsi14_norm"] = _rsi(c, 14) / 50 - 1

    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    df["macd_norm"] = (macd_line - signal) / c

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bollinger_b"] = (c - sma20) / (2 * std20.replace(0, np.nan))

    v = df["volume"].astype(float)
    df["log_vol_change"] = np.log(v / v.shift(1).replace(0, np.nan))

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


STATE_COLS_NO_SENTIMENT = [
    "log_return", "position",
    "dow_sin", "dow_cos", "dom_sin", "dom_cos", "moy_sin", "moy_cos",
    "ema10_rel", "ema50_rel", "rsi14_norm", "macd_norm", "bollinger_b", "log_vol_change",
]
STATE_COLS = STATE_COLS_NO_SENTIMENT + ["sentiment"]
