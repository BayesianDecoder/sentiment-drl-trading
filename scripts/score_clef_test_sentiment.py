"""
Scores the CLEF Task3 test-period news (302 days x {TSLA, BTC}) with the user's fine-tuned
Qwen adapter, matching the paper's article-truncation and prompt structure (Section 4.3), just
with the Qwen backbone in place of LLaMA-3.2-1B.

Checkpoints after every single day (appends to the output CSV immediately, not just at the
end) and skips dates already scored on restart -- an earlier run was killed mid-way for an
unrelated reason and the un-checkpointed partial progress (100 days) was lost, which this
avoids happening again.

Run: PYTHONPATH=. .venv/bin/python -u scripts/score_clef_test_sentiment.py
"""
import csv
import os
import time

import pandas as pd

from src.paper_replication import sentiment as sent

ASSETS = {"TSLA": "Tesla", "BTC": "Bitcoin"}


def main():
    for asset, friendly in ASSETS.items():
        df = pd.read_parquet(f"data/raw/paper_replication/clef_task3/data/{asset}-00000-of-00001.parquet")
        df = df.sort_values("date").reset_index(drop=True)

        out_path = f"data/processed/paper_replication/{asset}_test_sentiment.csv"
        done_dates = set()
        if os.path.exists(out_path):
            done_dates = set(pd.read_csv(out_path)["date"].astype(str))
        else:
            with open(out_path, "w", newline="") as f:
                csv.writer(f).writerow(["date", "sentiment"])

        print(f"=== {asset}: {len(df)} days ({len(done_dates)} already scored) ===", flush=True)
        t0 = time.time()
        n_scored_this_run = 0
        for i, r in df.iterrows():
            date_str = str(r["date"])
            if date_str in done_dates:
                continue
            news_items = r["news"]
            text = " ".join(str(x) for x in news_items) if len(news_items) else ""
            score = sent.score_article(friendly, text) if text.strip() else 0.0
            with open(out_path, "a", newline="") as f:
                csv.writer(f).writerow([date_str, score])
            n_scored_this_run += 1
            if n_scored_this_run % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {asset} {i+1}/{len(df)} | elapsed={elapsed:.0f}s | last_score={score:.3f}", flush=True)
        print(f"Saved {out_path} ({len(done_dates) + n_scored_this_run} rows)", flush=True)


if __name__ == "__main__":
    main()
