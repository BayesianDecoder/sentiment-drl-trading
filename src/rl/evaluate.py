"""
evaluate.py — Part B (RL trading agents)

Backtests every trained agent (5 agents x {with, without} sentiment) on the
held-out TEST split, compares against a buy-and-hold benchmark, and reports
cumulative return / Sharpe / max drawdown. Also builds a simple "ensemble"
agent: whichever single agent had the best validation Sharpe (a simplified,
single-selection version of the ICAIF-2020 paper's rolling ensemble, since
we only have one val window here rather than several).

Run:
    python -m src.rl.evaluate
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize

from src.rl import config as C
from src.rl.train_agents import load_splits, make_env, quick_backtest, AGENT_CLASSES


def backtest_series(model, panel_split, use_sentiment: bool, norm_path: str):
    """Same as train_agents.quick_backtest but also returns the full value series."""
    eval_env = make_env(panel_split, use_sentiment, training=False)
    eval_env = VecNormalize.load(norm_path, eval_env.venv)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    values = []
    turnover_info = {}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        done = bool(done[0])
        values.append(info[0]["portfolio_value"])
        if "total_shares_traded" in info[0]:
            turnover_info = info[0]

    n_steps = turnover_info.get("n_steps", len(values))
    diagnostics = {
        "avg_daily_shares_traded": turnover_info.get("total_shares_traded", 0.0) / max(n_steps, 1),
        "pct_days_traded": turnover_info.get("n_trade_days", 0) / max(n_steps, 1) * 100,
        "total_transaction_cost": turnover_info.get("total_transaction_cost", 0.0),
    }
    return np.array(values), diagnostics


def buy_and_hold_series(panel_split, tickers=C.TICKERS, initial_cash=C.INITIAL_CASH):
    """Equal-weight buy-and-hold benchmark over the same period."""
    close = panel_split.pivot(index="date", columns="ticker", values="close")[tickers].sort_index()
    per_ticker_cash = initial_cash / len(tickers)
    shares = per_ticker_cash / close.iloc[0]
    values = (close * shares).sum(axis=1)
    return values.to_numpy()


def compute_metrics(values: np.ndarray) -> dict:
    returns = np.diff(values) / values[:-1]
    cumulative_return_pct = (values[-1] / values[0] - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100
    return {"cumulative_return_pct": cumulative_return_pct, "sharpe": sharpe, "max_drawdown_pct": max_drawdown_pct}


def main():
    _, val_panel, test_panel = load_splits()

    summary_path = os.path.join(C.RESULTS_DIR, "training_summary.csv")
    training_summary = pd.read_csv(summary_path)

    rows = []
    series = {}
    for _, row in training_summary.iterrows():
        agent_name, use_sentiment = row["agent"], bool(row["use_sentiment"])
        tag = f"{agent_name}_{'sentiment' if use_sentiment else 'nosentiment'}"
        model_path = os.path.join(C.MODELS_DIR, f"{tag}.zip")
        norm_path = os.path.join(C.MODELS_DIR, f"{tag}_vecnormalize.pkl")
        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            print(f"Skipping {tag} — model/normalize file missing")
            continue

        model = AGENT_CLASSES[agent_name].load(model_path)
        values, diagnostics = backtest_series(model, test_panel, use_sentiment, norm_path)
        series[tag] = values
        metrics = compute_metrics(values)
        rows.append({"agent": agent_name, "use_sentiment": use_sentiment, **metrics, **diagnostics})
        print(f"{tag}: return={metrics['cumulative_return_pct']:.2f}% "
              f"sharpe={metrics['sharpe']:.3f} max_dd={metrics['max_drawdown_pct']:.2f}% "
              f"avg_daily_shares_traded={diagnostics['avg_daily_shares_traded']:.1f} "
              f"pct_days_traded={diagnostics['pct_days_traded']:.1f}% "
              f"total_txn_cost=Rs.{diagnostics['total_transaction_cost']:.0f}")

    bh_values = buy_and_hold_series(test_panel)
    bh_metrics = compute_metrics(bh_values)
    rows.append({"agent": "Buy&Hold", "use_sentiment": None, **bh_metrics})
    series["Buy&Hold"] = bh_values
    print(f"Buy&Hold: return={bh_metrics['cumulative_return_pct']:.2f}% "
          f"sharpe={bh_metrics['sharpe']:.3f} max_dd={bh_metrics['max_drawdown_pct']:.2f}%")

    results_df = pd.DataFrame(rows)

    # "Ensemble": champion = best val Sharpe among the sentiment-fused agents (real fused-sentiment
    # models are the ones meant to be deployed) — a simplified, single-selection stand-in for the
    # ICAIF-2020 paper's rolling multi-window ensemble.
    sentiment_rows = training_summary[training_summary["use_sentiment"] == True]
    if len(sentiment_rows):
        champion = sentiment_rows.loc[sentiment_rows["val_sharpe"].idxmax()]
        print(f"\nEnsemble champion (best val Sharpe, sentiment-fused): {champion['agent']} "
              f"(val_sharpe={champion['val_sharpe']:.3f})")

    out_path = os.path.join(C.RESULTS_DIR, "test_backtest_summary.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved backtest summary to {out_path}")

    # ── With vs without sentiment, per agent ──
    pivot = results_df[results_df["agent"] != "Buy&Hold"].pivot(
        index="agent", columns="use_sentiment", values="sharpe"
    )
    if True in pivot.columns and False in pivot.columns:
        pivot.columns = ["Sharpe (no sentiment)", "Sharpe (with sentiment)"]
        pivot["delta"] = pivot["Sharpe (with sentiment)"] - pivot["Sharpe (no sentiment)"]
        print("\n" + "=" * 60)
        print("SENTIMENT FUSION EFFECT ON TEST SHARPE (per agent)")
        print("=" * 60)
        print(pivot.to_string())

    # ── Plot portfolio value curves ──
    fig, ax = plt.subplots(figsize=(11, 6))
    for tag, values in series.items():
        style = "--" if tag == "Buy&Hold" else "-"
        ax.plot(values / values[0], style, label=tag, alpha=0.8)
    ax.set_title("Test-period portfolio value (normalized to 1.0 at start)")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Portfolio value / initial value")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plot_path = os.path.join(C.RESULTS_DIR, "test_backtest_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
