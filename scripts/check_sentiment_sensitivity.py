"""
Verifies the trained agents actually respond to sentiment, rather than just having it present
in the observation vector and ignoring it. Runs each finished agent's test-period backtest
twice: once with real sentiment values, once with sentiment forced to 0 on every day. If the
resulting actions/portfolio values differ, the policy is genuinely using the feature.
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_rl_results import (
    PortfolioTradingEnv, load_splits, TICKERS, AGENT_CLASSES, NOTEBOOKS_DIR
)


def make_env_zero_sentiment(panel_split):
    """Same env, but sentiment column forced to 0 for every row -- isolates the sentiment effect."""
    panel_zeroed = panel_split.copy()
    panel_zeroed["sentiment"] = 0.0

    def _init():
        return Monitor(PortfolioTradingEnv(panel_zeroed))
    return DummyVecEnv([_init])


def make_env_real(panel_split):
    def _init():
        return Monitor(PortfolioTradingEnv(panel_split))
    return DummyVecEnv([_init])


def rollout(model, env, norm_path):
    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    actions, values = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        obs, reward, done, info = env.step(action)
        done = bool(done[0])
        values.append(info[0]["portfolio_value"])
    return np.array(actions), np.array(values)


def main():
    test_panel = load_splits()
    n_real_sentiment_days = test_panel["has_real_sentiment"].sum() if "has_real_sentiment" in test_panel.columns else "?"
    print(f"Test period real-sentiment rows: {n_real_sentiment_days}\n")

    for agent in AGENT_CLASSES:
        model_path = os.path.join(NOTEBOOKS_DIR, "saved_models", f"{agent}_sentiment.zip")
        norm_path = os.path.join(NOTEBOOKS_DIR, "saved_models", f"{agent}_sentiment_vecnormalize.pkl")
        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            print(f"{agent}: not finished yet, skipping")
            continue

        model = AGENT_CLASSES[agent].load(model_path)

        actions_real, values_real = rollout(model, make_env_real(test_panel), norm_path)
        actions_zero, values_zero = rollout(model, make_env_zero_sentiment(test_panel), norm_path)

        action_diff = np.abs(actions_real - actions_zero)
        mean_action_diff = action_diff.mean()
        max_action_diff = action_diff.max()
        final_value_diff_pct = abs(values_real[-1] - values_zero[-1]) / values_zero[-1] * 100

        print(f"{agent}:")
        print(f"  mean |action_real - action_zero_sentiment| = {mean_action_diff:.5f}")
        print(f"  max  |action_real - action_zero_sentiment| = {max_action_diff:.5f}")
        print(f"  final portfolio value difference: {final_value_diff_pct:.3f}%")
        if mean_action_diff < 1e-4:
            print(f"  -> Sentiment appears to have NEGLIGIBLE effect on {agent}'s actions.")
        else:
            print(f"  -> Sentiment measurably changes {agent}'s actions -- the policy IS using it.")
        print()


if __name__ == "__main__":
    main()
