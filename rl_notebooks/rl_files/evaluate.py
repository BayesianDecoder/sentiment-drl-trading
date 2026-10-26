# evaluate.py — Final benchmark evaluation on test set
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from trading_env import TradingEnv
from trainer import evaluate_agent, compute_sharpe

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_full_evaluation(agents: Dict,
                        test_env_fn,
                        val_env_fn,
                        asset: str) -> pd.DataFrame:
    """
    Run final evaluation of all agents on val and test sets.
    Produces benchmark table matching the paper's Table 4.
    """
    rows = []

    for name, agent in agents.items():
        # Val evaluation
        val_env = val_env_fn()
        val_sharpe, val_metrics = evaluate_agent(agent, val_env)

        # Test evaluation (touched once at the end)
        test_env = test_env_fn()
        test_sharpe, test_metrics = evaluate_agent(agent, test_env)

        row = {
            "Asset":     asset,
            "Model":     name,
            "Val CR%":   val_metrics["cumulative_return"],
            "Val SR":    round(val_sharpe, 3),
            "Test CR%":  test_metrics["cumulative_return"],
            "Test SR":   round(test_sharpe, 3),
            "MDD%":      test_metrics["max_drawdown"],
            "AV%":       test_metrics["ann_volatility"],
            "N_Trades":  test_metrics["n_trades"],
        }
        rows.append(row)
        print(f"  {name:8s} | Val SR={val_sharpe:.3f}  Val CR={val_metrics['cumulative_return']:.1f}% | "
              f"Test SR={test_sharpe:.3f}  Test CR={test_metrics['cumulative_return']:.1f}%  MDD={test_metrics['max_drawdown']:.1f}%")

    # Buy-and-hold baseline
    bnh_test = test_env_fn()
    bnh_test.reset()
    for i in range(bnh_test.T - 1):
        bnh_test.step(2)   # always LONG
    bnh_metrics = bnh_test.get_metrics()
    bnh_sharpe  = compute_sharpe(np.array(bnh_test.portfolio_values))

    bnh_val  = val_env_fn()
    bnh_val.reset()
    for i in range(bnh_val.T - 1):
        bnh_val.step(2)
    bnh_val_metrics = bnh_val.get_metrics()
    bnh_val_sharpe  = compute_sharpe(np.array(bnh_val.portfolio_values))

    rows.insert(0, {
        "Asset":    asset,
        "Model":    "B&H",
        "Val CR%":  bnh_val_metrics["cumulative_return"],
        "Val SR":   round(bnh_val_sharpe, 3),
        "Test CR%": bnh_metrics["cumulative_return"],
        "Test SR":  round(bnh_sharpe, 3),
        "MDD%":     bnh_metrics["max_drawdown"],
        "AV%":      bnh_metrics["ann_volatility"],
        "N_Trades": 0,
    })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"benchmark_{asset}.csv", index=False)
    return df


def plot_portfolio_trajectories(agents: Dict,
                                test_env_fn,
                                asset: str,
                                initial_value: float = 100_000):
    """Plot portfolio value trajectories vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Buy-and-hold
    bnh = test_env_fn()
    bnh.reset()
    for i in range(bnh.T - 1):
        bnh.step(2)
    ax.plot(bnh.portfolio_values, label="Buy & Hold", color="black",
            linestyle="--", linewidth=2)

    colors = {"PG": "#1f77b4", "PPO": "#ff7f0e",
              "DQL": "#2ca02c", "DDPG": "#d62728"}

    for name, agent in agents.items():
        env = test_env_fn()
        env.reset()
        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden()
        done = False
        obs = env._get_obs()
        while not done:
            action = agent.greedy_action(obs)
            obs, _, done, _, _ = env.step(action)

        ax.plot(env.portfolio_values, label=name,
                color=colors.get(name, "gray"), linewidth=1.5)

    ax.set_title(f"{asset} — Portfolio Value (Test Set)", fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    path = RESULTS_DIR / f"trajectories_{asset}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_final_table(results: Dict[str, pd.DataFrame]):
    """Print the final benchmark table."""
    print("\n" + "="*90)
    print("FINAL BENCHMARK — Test Set Results")
    print("="*90)

    for asset, df in results.items():
        print(f"\n{asset}:")
        print(df.to_string(index=False))

    print("="*90)
    print("Metrics: CR=Cumulative Return | SR=Sharpe Ratio | MDD=Max Drawdown | AV=Ann. Volatility")

    # Save combined
    combined = pd.concat(list(results.values()))
    combined.to_csv(RESULTS_DIR / "final_benchmark.csv", index=False)
    print(f"\nSaved to results/final_benchmark.csv")
