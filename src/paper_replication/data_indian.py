"""
Reshapes the existing, real Qwen-sentiment-scored rl_panel.csv (10 Indian stocks, produced by
src/rl/data_pipeline.py) into the paper's per-asset 15-dim schema, so the paper's env/algorithms
(env.py, algos.py) can run per-ticker with real sentiment throughout -- no gated dataset, no
neutral placeholder, matching the same train/val/test split and sentiment-bootstrap approach
already validated in Track 1 (src/rl/train_v1.py's load_splits).
"""
import pandas as pd

from src.paper_replication.features import add_technical_indicators
from src.rl import config as C

PANEL_PATH = "data/processed/rl_panel.csv"


def build_panels_for_ticker(ticker: str):
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    dates = panel["date"].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * C.TRAIN_FRAC)]
    val_end = dates[int(n * (C.TRAIN_FRAC + C.VAL_FRAC))]

    sub = panel[panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if "sentiment_bootstrap" in sub.columns:
        sub["sentiment_for_train"] = sub["sentiment_bootstrap"]
    else:
        sub["sentiment_for_train"] = sub["sentiment"]

    train = sub[sub["date"] <= train_end].copy()
    val = sub[(sub["date"] > train_end) & (sub["date"] <= val_end)].copy()
    test = sub[sub["date"] > val_end].copy()

    train["sentiment"] = train["sentiment_for_train"]
    val["sentiment"] = val["sentiment_for_train"]
    # test keeps the real (sparse, non-bootstrapped) sentiment column as-is

    out = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        d = split[["date", "close", "volume", "sentiment"]].reset_index(drop=True)
        d = add_technical_indicators(d)
        d["position"] = 0.0
        out[name] = d
    return out["train"], out["val"], out["test"]
