# main.py — Complete pipeline: data → sentiment → features → train → evaluate
"""
Sentiment-Augmented DRL Trading System
Replicates + improves CLaC@FinMMEval2026 with fine-tuned Qwen2.5-3B sentiment

Usage:
  python main.py --asset TSLA              # single asset
  python main.py --asset BTC-USD           # Bitcoin
  python main.py --asset TSLA --mock-sent  # skip sentiment model (fast test)
  python main.py --asset TSLA --device mps # Apple Silicon
"""
import argparse
import numpy as np
import torch
from pathlib import Path

from config import (
    ASSETS, TRAIN_END, VAL_START, VAL_END, TEST_START,
    INITIAL_VALUE, WARMUP_DAYS, DEVICE,
    PGConfig, PPOConfig, DQLConfig, DDPGConfig
)
from data_pipeline  import build_dataset, split_dataset
from sentiment_scorer import SEntFiNScorer, MockScorer
from feature_eng    import build_state_matrix
from trading_env    import TradingEnv
from agents         import PGAgent, PPOAgent, DQLAgent, DDPGAgent
from trainer        import train_agent
from evaluate       import run_full_evaluation, plot_portfolio_trajectories, print_final_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--asset",     default="TSLA", choices=["TSLA", "BTC-USD"])
    p.add_argument("--device",    default=DEVICE)
    p.add_argument("--episodes",  type=int, default=2000)
    p.add_argument("--mock-sent", action="store_true",
                   help="Use mock sentiment (skip GPU model, for testing)")
    p.add_argument("--agents",    nargs="+", default=["PG","PPO","DQL","DDPG"],
                   help="Which agents to train")
    return p.parse_args()


def build_env(states, returns, random_start=True, apply_cost=True):
    return TradingEnv(
        states=states,
        returns=returns,
        transaction_cost=0.002 if apply_cost else 0.0,
        initial_value=INITIAL_VALUE,
        random_start=random_start,
        apply_cost=apply_cost
    )


def main():
    args = parse_args()
    asset   = args.asset
    device  = args.device
    episodes = args.episodes

    print(f"\n{'='*60}")
    print(f"Sentiment-Augmented DRL Trading System")
    print(f"Asset: {asset} | Device: {device} | Episodes: {episodes}")
    print(f"{'='*60}\n")

    # ── Step 1: Load data ─────────────────────────────────────────────
    print("📊 Step 1: Loading data...")
    df = build_dataset(asset)
    train_df, val_df, test_df = split_dataset(df)

    # ── Step 2: Sentiment scoring ─────────────────────────────────────
    print("\n🧠 Step 2: Computing sentiment scores...")
    if args.mock_sent:
        print("  Using mock sentiment (--mock-sent flag)")
        scorer = MockScorer()
    else:
        scorer = SEntFiNScorer(device=device)

    train_sent = scorer.score_dataset(train_df, asset, f"train_{asset}")
    val_sent   = scorer.score_dataset(val_df,   asset, f"val_{asset}")
    test_sent  = scorer.score_dataset(test_df,  asset, f"test_{asset}")

    print(f"  Train sentiment: mean={train_sent.mean():.3f} std={train_sent.std():.3f}")
    print(f"  Val   sentiment: mean={val_sent.mean():.3f} std={val_sent.std():.3f}")

    # ── Step 3: Feature engineering ───────────────────────────────────
    print("\n⚙️  Step 3: Building state vectors...")

    # Prepend warmup days from previous split to warm up indicators
    # (matching paper: 60 warmup days prepended to val/test)
    train_with_warmup = train_df.copy()
    val_with_warmup   = pd.concat([train_df.tail(WARMUP_DAYS), val_df]).reset_index(drop=True)
    test_with_warmup  = pd.concat([val_df.tail(WARMUP_DAYS), test_df]).reset_index(drop=True)

    # Combine sentiments for warmup periods
    all_sent = pd.concat([train_sent, val_sent, test_sent]).drop_duplicates()

    train_states, train_returns, train_dates = build_state_matrix(
        train_with_warmup, all_sent, warmup_days=WARMUP_DAYS
    )
    val_states, val_returns, val_dates = build_state_matrix(
        val_with_warmup, all_sent, warmup_days=WARMUP_DAYS
    )
    test_states, test_returns, test_dates = build_state_matrix(
        test_with_warmup, all_sent, warmup_days=WARMUP_DAYS
    )

    print(f"  Train: {len(train_states)} days | Val: {len(val_states)} days | Test: {len(test_states)} days")

    import pandas as pd

    # ── Step 4: Create environments ───────────────────────────────────
    print("\n🎮 Step 4: Creating trading environments...")

    def make_train_env():
        return build_env(train_states, train_returns, random_start=True, apply_cost=True)

    def make_val_env():
        return build_env(val_states, val_returns, random_start=False, apply_cost=False)

    def make_test_env():
        return build_env(test_states, test_returns, random_start=False, apply_cost=False)

    # ── Step 5: Train all agents ──────────────────────────────────────
    print(f"\n🤖 Step 5: Training {args.agents}...")

    trained_agents = {}

    if "PG" in args.agents:
        print("\n── Policy Gradient (PG) ──")
        pg_cfg   = PGConfig()
        pg_agent = PGAgent(cfg=pg_cfg, device=device)
        pg_agent, pg_hist = train_agent(
            "PG", pg_agent, make_train_env(), make_val_env(),
            max_episodes=episodes
        )
        trained_agents["PG"] = pg_agent
        print(f"  PG best val SR: {pg_hist['best_val_sharpe']:.3f}")

    if "PPO" in args.agents:
        print("\n── Proximal Policy Optimization (PPO) ──")
        ppo_cfg   = PPOConfig()
        ppo_agent = PPOAgent(cfg=ppo_cfg, device=device)
        ppo_agent, ppo_hist = train_agent(
            "PPO", ppo_agent, make_train_env(), make_val_env(),
            max_episodes=episodes
        )
        trained_agents["PPO"] = ppo_agent
        print(f"  PPO best val SR: {ppo_hist['best_val_sharpe']:.3f}")

    if "DQL" in args.agents:
        print("\n── Deep Q-Learning (DQL) ──")
        dql_cfg   = DQLConfig()
        dql_agent = DQLAgent(cfg=dql_cfg, device=device)
        dql_agent, dql_hist = train_agent(
            "DQL", dql_agent, make_train_env(), make_val_env(),
            max_episodes=episodes
        )
        trained_agents["DQL"] = dql_agent
        print(f"  DQL best val SR: {dql_hist['best_val_sharpe']:.3f}")

    if "DDPG" in args.agents:
        print("\n── Deep Deterministic Policy Gradient (DDPG) ──")
        ddpg_cfg   = DDPGConfig()
        ddpg_agent = DDPGAgent(cfg=ddpg_cfg, device=device)
        ddpg_agent, ddpg_hist = train_agent(
            "DDPG", ddpg_agent, make_train_env(), make_val_env(),
            max_episodes=episodes
        )
        trained_agents["DDPG"] = ddpg_agent
        print(f"  DDPG best val SR: {ddpg_hist['best_val_sharpe']:.3f}")

    # ── Step 6: Final evaluation ──────────────────────────────────────
    print(f"\n📈 Step 6: Final evaluation on test set...")
    results_df = run_full_evaluation(
        agents     = trained_agents,
        test_env_fn = make_test_env,
        val_env_fn  = make_val_env,
        asset       = asset
    )

    # ── Step 7: Plot trajectories ─────────────────────────────────────
    print("\n📊 Step 7: Plotting portfolio trajectories...")
    plot_portfolio_trajectories(trained_agents, make_test_env, asset)

    # ── Final summary ─────────────────────────────────────────────────
    print_final_table({asset: results_df})

    print(f"\n✅ {asset} complete! Results in results/")
    return results_df


if __name__ == "__main__":
    main()
