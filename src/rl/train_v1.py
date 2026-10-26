"""
train_v1.py — full two-stage HPT pipeline (random search + continued training), matching the
exact rigor used for the with-sentiment notebooks, so a --no-sentiment run is a fair,
apples-to-apples comparison rather than a token single-shot run.

Run:
    python -m src.rl.train_v1 --no-sentiment                  # all 5 agents, no sentiment
    python -m src.rl.train_v1 --no-sentiment --agents PPO A2C # subset
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
ON_POLICY_AGENTS = {"PPO", "A2C"}

FAMILY_BUDGETS = {
    "on_policy": {"search_trials": 10, "search_timesteps": 15_000, "final_timesteps": 250_000},
    "off_policy": {"search_trials": 10, "search_timesteps": 30_000, "final_timesteps": 400_000},
}

SEARCH_SPACES = {
    "PPO": {
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3], "n_steps": [512, 1024, 2048],
        "batch_size": [32, 64, 128], "n_epochs": [5, 10, 20], "gamma": [0.95, 0.99, 0.995],
        "gae_lambda": [0.9, 0.95, 0.98], "clip_range": [0.1, 0.2, 0.3], "ent_coef": [0.0, 0.001, 0.01],
    },
    "A2C": {
        "learning_rate": [1e-4, 7e-4, 1e-3, 3e-3], "n_steps": [5, 20, 50],
        "gamma": [0.95, 0.99, 0.995], "gae_lambda": [0.9, 0.95, 1.0],
        "ent_coef": [0.0, 0.001, 0.01], "vf_coef": [0.25, 0.5, 0.75],
    },
    "DDPG": {
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3], "batch_size": [64, 100, 256],
        "tau": [0.001, 0.005, 0.01], "gamma": [0.95, 0.99, 0.995],
        "action_noise_sigma": [0.05, 0.1, 0.2, 0.3],
    },
    "SAC": {
        "learning_rate": [1e-4, 3e-4, 1e-3], "batch_size": [128, 256, 512],
        "tau": [0.001, 0.005, 0.01], "gamma": [0.95, 0.99, 0.995],
    },
    "TD3": {
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3], "batch_size": [64, 100, 256],
        "tau": [0.001, 0.005, 0.01], "gamma": [0.95, 0.99, 0.995],
        "policy_delay": [1, 2, 3], "action_noise_sigma": [0.05, 0.1, 0.2, 0.3],
    },
}


def load_splits():
    panel = pd.read_csv(C.RL_PANEL_PATH, parse_dates=["date"])
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    train_end = dates[int(n * C.TRAIN_FRAC)]
    val_end = dates[int(n * (C.TRAIN_FRAC + C.VAL_FRAC))]

    train = panel[panel["date"] <= train_end].copy()
    val = panel[(panel["date"] > train_end) & (panel["date"] <= val_end)].copy()
    test = panel[panel["date"] > val_end].copy()

    if "sentiment_bootstrap" in panel.columns:
        train["sentiment"] = train["sentiment_bootstrap"]
        val["sentiment"] = val["sentiment_bootstrap"]

    print(f"Split sizes: train={train['date'].nunique()} val={val['date'].nunique()} test={test['date'].nunique()}")
    return train, val, test


def make_env(panel_split, use_sentiment: bool, batch_size: int, training: bool):
    def _init():
        return Monitor(PortfolioTradingEnv(panel_split, use_sentiment=use_sentiment))
    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=training, clip_obs=10.0)
    return vec_env


def build_agent(name: str, env, params: dict, seed=C.SEED):
    cls = AGENT_CLASSES[name]
    params = dict(params)
    kwargs = dict(policy="MlpPolicy", env=env, seed=seed, verbose=0)
    if name not in ON_POLICY_AGENTS:
        sigma = params.pop("action_noise_sigma", 0.1)
        n_actions = env.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))
    kwargs.update(params)
    return cls(**kwargs)


def backtest(model, panel_split, use_sentiment, norm_env_or_path):
    eval_env = make_env(panel_split, use_sentiment, 64, training=False)
    if isinstance(norm_env_or_path, str):
        eval_env = VecNormalize.load(norm_env_or_path, eval_env.venv)
    else:
        eval_env.obs_rms = norm_env_or_path.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    values, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        done = bool(done[0])
        values.append(info[0]["portfolio_value"])
    values = np.array(values)
    returns = np.diff(values) / values[:-1]
    cumulative_return_pct = (values[-1] / C.INITIAL_CASH - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(values)
    max_dd = ((values - running_max) / running_max).min() * 100
    return {"cumulative_return_pct": cumulative_return_pct, "sharpe": sharpe, "max_drawdown_pct": max_dd, "values": values}


def run_trial(agent_name, config, train_panel, use_sentiment, timesteps):
    env = make_env(train_panel, use_sentiment, config.get("batch_size", 64), training=True)
    model = build_agent(agent_name, env, config)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model, env


def train_one(agent_name: str, train_panel, val_panel, test_panel, use_sentiment: bool, tag_suffix: str,
              search_trials: int | None = None, seed: int = C.SEED, final_timesteps: int | None = None):
    family = "on_policy" if agent_name in ON_POLICY_AGENTS else "off_policy"
    budget = dict(FAMILY_BUDGETS[family])
    if search_trials is not None:
        budget["search_trials"] = search_trials
    if final_timesteps is not None:
        budget["final_timesteps"] = final_timesteps
    tag = f"{agent_name}_{tag_suffix}"
    print(f"\n{'='*70}\n{tag}: {budget['search_trials']}-trial search ({budget['search_timesteps']} steps each), seed={seed}\n{'='*70}")

    rng = np.random.RandomState(seed)
    space = SEARCH_SPACES[agent_name]
    best = None
    leaderboard = []

    for trial in range(1, budget["search_trials"] + 1):
        config = {k: rng.choice(v) for k, v in space.items()}
        model, env = run_trial(agent_name, config, train_panel, use_sentiment, budget["search_timesteps"])
        metrics = backtest(model, val_panel, use_sentiment, env)
        print(f"  trial {trial}/{budget['search_trials']} | val_sharpe={metrics['sharpe']:.3f} | "
              f"val_return={metrics['cumulative_return_pct']:.2f}% | config={config}")
        leaderboard.append({"trial": trial, **config, "val_sharpe": metrics["sharpe"]})
        if best is None or metrics["sharpe"] > best["sharpe"]:
            best = {"model": model, "env": env, "config": config, "sharpe": metrics["sharpe"]}

    print(f"\nBest search config (val_sharpe={best['sharpe']:.3f}): {best['config']}")
    print(f"Continuing training for {budget['final_timesteps']} more steps...")

    start = time.time()
    best["model"].learn(total_timesteps=budget["final_timesteps"], reset_num_timesteps=False, progress_bar=False)
    train_time = time.time() - start

    model_path = os.path.join(C.MODELS_DIR, f"{tag}.zip")
    norm_path = os.path.join(C.MODELS_DIR, f"{tag}_vecnormalize.pkl")
    best["model"].save(model_path)
    best["env"].save(norm_path)

    test_metrics = backtest(best["model"], test_panel, use_sentiment, norm_path)
    print(f"{tag} TEST: return={test_metrics['cumulative_return_pct']:.2f}% | "
          f"sharpe={test_metrics['sharpe']:.3f} | max_dd={test_metrics['max_drawdown_pct']:.2f}% | "
          f"continued_train_time={train_time:.1f}s")

    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"agent": agent_name, "use_sentiment": use_sentiment, **best["config"]})
        mlflow.log_metric("test_return_pct", test_metrics["cumulative_return_pct"])
        mlflow.log_metric("test_sharpe", test_metrics["sharpe"])
        mlflow.log_metric("test_max_drawdown_pct", test_metrics["max_drawdown_pct"])
        mlflow.log_artifact(model_path)

    return {"agent": agent_name, "use_sentiment": use_sentiment,
            "test_return_pct": test_metrics["cumulative_return_pct"],
            "test_sharpe": test_metrics["sharpe"], "test_max_drawdown_pct": test_metrics["max_drawdown_pct"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=list(AGENT_CLASSES.keys()), choices=list(AGENT_CLASSES.keys()))
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--search-trials", type=int, default=None, help="Override FAMILY_BUDGETS search_trials")
    parser.add_argument("--final-timesteps", type=int, default=None, help="Override FAMILY_BUDGETS final_timesteps")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--tag-suffix", type=str, default=None, help="Override output tag suffix (avoids overwriting prior run's model/mlflow tag)")
    args = parser.parse_args()
    use_sentiment = not args.no_sentiment
    tag_suffix = args.tag_suffix or ("sentiment" if use_sentiment else "nosentiment_hpt")

    train_panel, val_panel, test_panel = load_splits()

    results = []
    for agent_name in args.agents:
        result = train_one(agent_name, train_panel, val_panel, test_panel, use_sentiment, tag_suffix,
                            search_trials=args.search_trials, seed=args.seed, final_timesteps=args.final_timesteps)
        results.append(result)

    df = pd.DataFrame(results)
    out_path = os.path.join(C.RESULTS_DIR, f"v1_{tag_suffix}_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
