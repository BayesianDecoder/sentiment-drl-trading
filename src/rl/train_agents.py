"""
train_agents.py — Part B (RL trading agents)

Trains PPO, A2C, DDPG, SAC, TD3 (stable-baselines3) on the PortfolioTradingEnv,
once with the sentiment feature and once without, so the two can be compared
head-to-head on the same price history and split.

Run:
    python -m src.rl.train_agents                      # all 5 agents, both variants
    python -m src.rl.train_agents --agents PPO SAC      # subset
    python -m src.rl.train_agents --timesteps 20000     # override budget
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import mlflow
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from src.rl import config as C
from src.rl.trading_env import PortfolioTradingEnv

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG, "SAC": SAC, "TD3": TD3}
ON_POLICY = {"PPO", "A2C"}   # these don't take action_noise


def load_splits():
    panel = pd.read_csv(C.RL_PANEL_PATH, parse_dates=["date"])
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    train_end = dates[int(n * C.TRAIN_FRAC)]
    val_end = dates[int(n * (C.TRAIN_FRAC + C.VAL_FRAC))]

    train = panel[panel["date"] <= train_end].copy()
    val = panel[(panel["date"] > train_end) & (panel["date"] <= val_end)].copy()
    test = panel[panel["date"] > val_end].copy()

    # TRAIN/VAL: swap in the bootstrap-filled sentiment (real values resampled onto days
    # without real news) so the sentiment-fused agent actually sees varied, non-zero
    # sentiment while learning. TEST keeps the genuine real+0-fill column untouched —
    # that's the actual held-out evaluation and shouldn't contain fabricated values.
    if "sentiment_bootstrap" in panel.columns:
        train["sentiment"] = train["sentiment_bootstrap"]
        val["sentiment"] = val["sentiment_bootstrap"]

    print(f"Split sizes (dates): train={train['date'].nunique()} "
          f"val={val['date'].nunique()} test={test['date'].nunique()}")
    print(f"  train: {train['date'].min().date()} -> {train['date'].max().date()}")
    print(f"  val:   {val['date'].min().date()} -> {val['date'].max().date()}")
    print(f"  test:  {test['date'].min().date()} -> {test['date'].max().date()}")
    return train, val, test


def make_env(panel_split: pd.DataFrame, use_sentiment: bool, training: bool = True):
    def _init():
        env = PortfolioTradingEnv(panel_split, use_sentiment=use_sentiment)
        return Monitor(env)
    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=training, clip_obs=10.0)
    return vec_env


def build_agent(name: str, env, seed=C.SEED):
    cls = AGENT_CLASSES[name]
    kwargs = dict(policy="MlpPolicy", env=env, seed=seed, verbose=0)
    if name not in ON_POLICY:
        n_actions = env.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    return cls(**kwargs)


def train_one(agent_name: str, train_panel, val_panel, use_sentiment: bool, total_timesteps: int):
    tag = f"{agent_name}_{'sentiment' if use_sentiment else 'nosentiment'}"
    print(f"\n{'='*70}\nTraining {tag} for {total_timesteps} timesteps\n{'='*70}")

    train_env = make_env(train_panel, use_sentiment, training=True)
    model = build_agent(agent_name, train_env)

    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    train_time = time.time() - start

    model_path = os.path.join(C.MODELS_DIR, f"{tag}.zip")
    norm_path = os.path.join(C.MODELS_DIR, f"{tag}_vecnormalize.pkl")
    model.save(model_path)
    train_env.save(norm_path)

    val_return, val_sharpe, diagnostics = quick_backtest(model, val_panel, use_sentiment, norm_path)

    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"agent": agent_name, "use_sentiment": use_sentiment,
                            "total_timesteps": total_timesteps})
        mlflow.log_metric("train_time_seconds", round(train_time, 2))
        mlflow.log_metric("val_cumulative_return_pct", val_return)
        mlflow.log_metric("val_sharpe", val_sharpe)
        for k, v in diagnostics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(model_path)

    print(f"{tag}: train_time={train_time:.1f}s | val_return={val_return:.2f}% | val_sharpe={val_sharpe:.3f} | "
          f"avg_daily_shares_traded={diagnostics['avg_daily_shares_traded']:.1f} | "
          f"pct_days_traded={diagnostics['pct_days_traded']:.1f}% | "
          f"total_txn_cost=Rs.{diagnostics['total_transaction_cost']:.0f}")
    return {"agent": agent_name, "use_sentiment": use_sentiment, "train_time": train_time,
            "val_return_pct": val_return, "val_sharpe": val_sharpe, **diagnostics}


def quick_backtest(model, panel_split, use_sentiment: bool, norm_path: str):
    """Deterministic rollout over panel_split, using the training VecNormalize stats (frozen)."""
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

    values = np.array(values)
    daily_returns = np.diff(values) / values[:-1]
    cumulative_return_pct = (values[-1] / C.INITIAL_CASH - 1) * 100
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252) if len(daily_returns) > 1 else 0.0

    n_steps = turnover_info.get("n_steps", len(values))
    diagnostics = {
        "avg_daily_shares_traded": turnover_info.get("total_shares_traded", 0.0) / max(n_steps, 1),
        "pct_days_traded": turnover_info.get("n_trade_days", 0) / max(n_steps, 1) * 100,
        "total_transaction_cost": turnover_info.get("total_transaction_cost", 0.0),
    }
    return cumulative_return_pct, sharpe, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=C.AGENT_NAMES, choices=list(AGENT_CLASSES.keys()))
    parser.add_argument("--timesteps", type=int, default=None,
                         help="Override timesteps for ALL agents. Default: on-policy agents "
                              f"(PPO/A2C) get {C.ON_POLICY_TIMESTEPS}, off-policy (DDPG/SAC/TD3) "
                              f"get {C.OFF_POLICY_TIMESTEPS} (they need more samples to converge).")
    parser.add_argument("--include-baseline", action="store_true",
                         help="Also train the no-sentiment baseline variant for comparison. "
                              "Off by default — sentiment fusion is the project's actual goal, "
                              "so by default only the sentiment-enabled variant is trained.")
    args = parser.parse_args()

    train_panel, val_panel, _ = load_splits()
    sentiment_settings = [False, True] if args.include_baseline else [True]

    results = []
    for use_sentiment in sentiment_settings:
        for agent_name in args.agents:
            if args.timesteps is not None:
                total_timesteps = args.timesteps
            elif agent_name in C.ON_POLICY_AGENTS:
                total_timesteps = C.ON_POLICY_TIMESTEPS
            else:
                total_timesteps = C.OFF_POLICY_TIMESTEPS
            result = train_one(agent_name, train_panel, val_panel, use_sentiment, total_timesteps)
            results.append(result)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(C.RESULTS_DIR, "training_summary.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved training summary to {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
