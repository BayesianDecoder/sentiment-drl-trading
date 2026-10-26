"""
Prints our replication's results side-by-side with the paper's Table 4 numbers.

Run: PYTHONPATH=. .venv/bin/python scripts/compare_to_paper.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "paper_replication")

PAPER_TABLE_4 = {
    ("TSLA", "B&H"): {"cr": 16.45, "sr": 0.61, "mdd": 29.93},
    ("TSLA", "PG"): {"cr": 13.96, "sr": 0.55, "mdd": 29.93},
    ("TSLA", "PPO"): {"cr": 2.54, "sr": 0.24, "mdd": 28.38},
    ("TSLA", "DQL"): {"cr": 52.62, "sr": 1.38, "mdd": 35.76},
    ("TSLA", "DDPG"): {"cr": 54.96, "sr": 1.44, "mdd": 19.20},
    ("BTC", "B&H"): {"cr": -34.27, "sr": -0.87, "mdd": 49.72},
    ("BTC", "PG"): {"cr": -14.53, "sr": -0.33, "mdd": 29.98},
    ("BTC", "PPO"): {"cr": -33.80, "sr": -0.85, "mdd": 59.18},
    ("BTC", "DQL"): {"cr": -23.22, "sr": -0.55, "mdd": 37.41},
    ("BTC", "DDPG"): {"cr": 1.58, "sr": 0.23, "mdd": 35.30},
}


def main():
    rows = []
    for asset in ["TSLA", "BTC"]:
        path = os.path.join(RESULTS_DIR, f"{asset}_summary.csv")
        if not os.path.exists(path):
            print(f"{asset}: no results yet ({path} not found)")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            model = r["model"]
            paper = PAPER_TABLE_4.get((asset, model))
            rows.append({
                "asset": asset, "model": model,
                "ours_CR%": round(r["cumulative_return_pct"], 2),
                "paper_CR%": paper["cr"] if paper else None,
                "ours_SR": round(r["sharpe"], 2),
                "paper_SR": paper["sr"] if paper else None,
                "ours_MDD%": round(r["max_drawdown_pct"], 2),
                "paper_MDD%": paper["mdd"] if paper else None,
            })
    if not rows:
        print("No results to compare yet.")
        return
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(RESULTS_DIR, "paper_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
