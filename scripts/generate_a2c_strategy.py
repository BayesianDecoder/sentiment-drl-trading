"""
Replays A2C's test-period backtest (the winning agent) and logs a full per-day, per-stock
timeline: price, shares held, the action taken (buy/hold/sell, derived from the actual change
in holdings — not just the raw network output), and the sentiment value used that day.

This is what drives the detailed strategy view in the dashboard — every number in it comes from
actually running the saved model through the environment, not a mock.

Drives the RAW gymnasium env directly (not through SB3's VecEnv wrapper) — VecEnv auto-resets
the underlying env the instant an episode ends, which would silently zero out the very last
day's holdings/date by the time we read them. VecNormalize's saved obs statistics are applied
manually instead, since the model was trained on normalized observations.

Run: .venv/bin/python scripts/generate_a2c_strategy.py
Writes: data/processed/a2c_strategy.json
"""
import os
import json

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_rl_results import (
    PortfolioTradingEnv, load_splits, TICKERS, NOTEBOOKS_DIR, make_env, buy_and_hold, INITIAL_CASH
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data", "processed", "a2c_strategy.json")

ACTION_THRESHOLD = 1  # minimum whole-share change to count as a real buy/sell vs. rounding noise


def main():
    test_panel = load_splits()
    model_path = os.path.join(NOTEBOOKS_DIR, "saved_models", "A2C_sentiment.zip")
    norm_path = os.path.join(NOTEBOOKS_DIR, "saved_models", "A2C_sentiment_vecnormalize.pkl")
    if not (os.path.exists(model_path) and os.path.exists(norm_path)):
        raise FileNotFoundError("A2C_sentiment model not found -- has training finished?")

    model = A2C.load(model_path)

    # Load VecNormalize purely to recover its fitted obs_rms stats -- we normalize manually below.
    vecnorm = VecNormalize.load(norm_path, make_env(test_panel))

    raw_env = PortfolioTradingEnv(test_panel)
    dates = [str(d.date()) for d in raw_env.dates]

    obs, _ = raw_env.reset()
    prev_holdings = raw_env.holdings.copy()
    days = []
    terminated = truncated = False

    while not (terminated or truncated):
        prices_today = raw_env._get_prices(raw_env.t).tolist()
        sentiment_today = raw_env.indicators["sentiment"].iloc[raw_env.t].tolist()

        norm_obs = vecnorm.normalize_obs(obs).reshape(1, -1)
        action, _ = model.predict(norm_obs, deterministic=True)

        obs, reward, terminated, truncated, info = raw_env.step(action[0])

        new_holdings = raw_env.holdings.copy()
        deltas = new_holdings - prev_holdings

        per_ticker = {}
        for i, ticker in enumerate(TICKERS):
            if deltas[i] > ACTION_THRESHOLD:
                label = "buy"
            elif deltas[i] < -ACTION_THRESHOLD:
                label = "sell"
            else:
                label = "hold"
            per_ticker[ticker] = {
                "price": round(float(prices_today[i]), 2),
                "shares_held": round(float(new_holdings[i]), 2),
                "shares_delta": round(float(deltas[i]), 2),
                "action": label,
                "sentiment": round(float(sentiment_today[i]), 4),
            }

        days.append({
            "date": dates[min(raw_env.t, len(dates) - 1)],
            "portfolio_value": round(float(info["portfolio_value"]), 2),
            "cash": round(float(raw_env.cash), 2),
            "stocks": per_ticker,
        })

        prev_holdings = new_holdings

    bh_values = buy_and_hold(test_panel)
    bh_normalized = (bh_values / bh_values[0]).tolist()

    result = {
        "agent": "A2C",
        "tickers": TICKERS,
        "initial_cash": INITIAL_CASH,
        "days": days,
        "buy_hold_curve": bh_normalized,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f)

    n_buys = sum(1 for d in days for s in d["stocks"].values() if s["action"] == "buy")
    n_sells = sum(1 for d in days for s in d["stocks"].values() if s["action"] == "sell")
    print(f"Wrote {len(days)} days x {len(TICKERS)} stocks to {OUT_PATH}")
    print(f"Date range: {days[0]['date']} -> {days[-1]['date']}")
    print(f"Total buy actions: {n_buys}, sell actions: {n_sells}")


if __name__ == "__main__":
    main()
