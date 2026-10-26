"""
Pushes the real results from src/rl/train_v1.py's Qwen-sentiment retrain to the dashboard.

Deliberately reuses train_v1.py's own load_splits/make_env/backtest functions (not the older
collect_rl_results.py, which points at a different, stale notebooks/saved_models/ directory
from a prior pipeline) so the numbers pushed here are exactly the numbers that were measured.

Run: .venv/bin/python scripts/push_v1_results.py [--tag-suffix sentiment]
"""
import argparse
import os

import requests

from src.rl import config as C
from src.rl import train_v1 as T

EXPRESS_API = os.environ.get("EXPRESS_API_URL", "http://localhost:4000/api")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-suffix", default="sentiment")
    parser.add_argument("--agents", nargs="+", default=list(T.AGENT_CLASSES.keys()))
    args = parser.parse_args()

    _, _, test_panel = T.load_splits()
    close = test_panel.pivot(index="date", columns="ticker", values="close")[C.TICKERS].sort_index()
    shares = (C.INITIAL_CASH / len(C.TICKERS)) / close.iloc[0]
    bh_values = (close * shares).sum(axis=1).to_numpy()
    bh_returns = (bh_values[1:] - bh_values[:-1]) / bh_values[:-1]
    bh_ret = (bh_values[-1] / bh_values[0] - 1) * 100
    bh_sharpe = (bh_returns.mean() / (bh_returns.std() + 1e-9)) * (252 ** 0.5)
    bh_curve = (bh_values / bh_values[0]).tolist()

    for agent in args.agents:
        tag = f"{agent}_{args.tag_suffix}"
        model_path = os.path.join(C.MODELS_DIR, f"{tag}.zip")
        norm_path = os.path.join(C.MODELS_DIR, f"{tag}_vecnormalize.pkl")
        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            print(f"{agent}: no saved model at {model_path}, skipping")
            continue

        model = T.AGENT_CLASSES[agent].load(model_path)
        metrics = T.backtest(model, test_panel, True, norm_path)
        values = metrics["values"]
        payload = {
            "variant": args.tag_suffix,
            "test_return_pct": metrics["cumulative_return_pct"],
            "test_sharpe": metrics["sharpe"],
            "test_max_drawdown_pct": metrics["max_drawdown_pct"],
            "beat_buy_and_hold": bool(metrics["cumulative_return_pct"] > bh_ret and metrics["sharpe"] > bh_sharpe),
            "portfolio_curve": (values / values[0]).tolist(),
            "buy_hold_curve": bh_curve,
            "status": "complete",
        }
        resp = requests.put(f"{EXPRESS_API}/rl-models/{agent}", json=payload)
        print(f"{agent}: return={metrics['cumulative_return_pct']:.2f}% sharpe={metrics['sharpe']:.3f} "
              f"max_dd={metrics['max_drawdown_pct']:.2f}% beat_bh={payload['beat_buy_and_hold']} "
              f"-> pushed, status={resp.status_code}")


if __name__ == "__main__":
    main()
