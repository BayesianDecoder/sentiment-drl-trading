"""
train_v3.py — two-stage HPT (random search + continued training) on PortfolioTradingEnvV3
(alpha reward, scale-normalized features, calendar encodings, cross-sectional features,
volatility regime, sentiment smoothing, random episode starts -- see EXPERIMENT_LOG.md).

Adds real Sharpe-based early stopping during the final training phase (SharpeEvalCallback
below) -- periodically backtests on the validation split, keeps the best-Sharpe checkpoint,
stops after `patience` non-improving evaluations -- rather than just taking whatever the last
training step produces. This is the mid-training discipline the v1 pipeline didn't have.

Every completed agent appends a real, numbered entry to EXPERIMENT_LOG.md automatically, so
results can never again go missing the way the undocumented "v2" experiment's did.

Run:
    PYTHONPATH=. .venv/bin/python -m src.rl.train_v3 --agents SAC DDPG
"""
import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import mlflow
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from src.rl import config as C
from src.rl.panel_v3 import build_v3_panel
from src.rl.trading_env_v3 import PortfolioTradingEnvV3

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG, "SAC": SAC, "TD3": TD3}
ON_POLICY_AGENTS = {"PPO", "A2C"}

FAMILY_BUDGETS = {
    "on_policy": {"search_trials": 8, "search_timesteps": 15_000, "final_timesteps": 300_000, "eval_freq": 15_000},
    "off_policy": {"search_trials": 8, "search_timesteps": 30_000, "final_timesteps": 400_000, "eval_freq": 20_000},
}
PATIENCE = 8

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
    panel = build_v3_panel()
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    train_end = dates[int(n * C.TRAIN_FRAC)]
    val_end = dates[int(n * (C.TRAIN_FRAC + C.VAL_FRAC))]
    train = panel[panel["date"] <= train_end].copy()
    val = panel[(panel["date"] > train_end) & (panel["date"] <= val_end)].copy()
    test = panel[panel["date"] > val_end].copy()
    print(f"Split sizes: train={train['date'].nunique()} val={val['date'].nunique()} test={test['date'].nunique()}", flush=True)
    return train, val, test


def make_env(panel_split, training: bool, seed=C.SEED, random_start: bool | None = None):
    def _init():
        return Monitor(PortfolioTradingEnvV3(panel_split, training=training, seed=seed, random_start=random_start))
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


def backtest(model, panel_split, norm_env_or_path):
    eval_env = make_env(panel_split, training=False)
    if isinstance(norm_env_or_path, str):
        eval_env = VecNormalize.load(norm_env_or_path, eval_env.venv)
    else:
        eval_env.obs_rms = norm_env_or_path.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    values, bh_values, done = [], [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        done = bool(done[0])
        values.append(info[0]["portfolio_value"])
        bh_values.append(info[0]["benchmark_value"])
    values = np.array(values)
    bh_values = np.array(bh_values)
    returns = np.diff(values) / values[:-1]
    cumulative_return_pct = (values[-1] / C.INITIAL_CASH - 1) * 100
    bh_return_pct = (bh_values[-1] / C.INITIAL_CASH - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(values)
    max_dd = ((values - running_max) / running_max).min() * 100
    return {"cumulative_return_pct": cumulative_return_pct, "sharpe": sharpe,
            "max_drawdown_pct": max_dd, "benchmark_return_pct": bh_return_pct, "values": values}


class SharpeEvalCallback(BaseCallback):
    """Periodically backtests on the validation split, tracks the best-Sharpe checkpoint in
    memory, stops training after `patience` consecutive non-improving evaluations."""

    def __init__(self, val_panel, train_env, eval_freq: int, patience: int = PATIENCE, verbose=0):
        super().__init__(verbose)
        self.val_panel = val_panel
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_sharpe = -np.inf
        self.best_weights = None
        self.no_improve_count = 0
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        metrics = backtest(self.model, self.val_panel, self.train_env)
        sharpe = metrics["sharpe"]
        self.history.append({"step": self.n_calls, "val_sharpe": sharpe, "val_return": metrics["cumulative_return_pct"]})
        if self.verbose:
            print(f"    [eval @ {self.n_calls}] val_sharpe={sharpe:.3f} val_return={metrics['cumulative_return_pct']:.2f}%", flush=True)
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_weights = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        if self.no_improve_count >= self.patience:
            if self.verbose:
                print(f"    early stopping: no improvement for {self.patience} evals", flush=True)
            return False
        return True

    def restore_best(self):
        if self.best_weights is not None:
            self.model.policy.load_state_dict(self.best_weights)


def append_to_experiment_log(tag: str, config: dict, val_sharpe: float, test_metrics: dict, elapsed_s: float):
    log_path = os.path.join(C.ROOT, "EXPERIMENT_LOG.md")
    entry = f"""
## v3 — {tag} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})

**Config**: `{config}`
**Best validation Sharpe during search/training**: {val_sharpe:.3f}
**Test result**: return={test_metrics['cumulative_return_pct']:.2f}% | sharpe={test_metrics['sharpe']:.3f} | max_dd={test_metrics['max_drawdown_pct']:.2f}% | benchmark_return={test_metrics['benchmark_return_pct']:.2f}%
**Beats benchmark**: {test_metrics['cumulative_return_pct'] > test_metrics['benchmark_return_pct'] and test_metrics['sharpe'] > 0}
**Training time**: {elapsed_s:.0f}s
"""
    with open(log_path, "a") as f:
        f.write(entry)


def train_one(agent_name: str, train_panel, val_panel, test_panel, tag_suffix: str):
    family = "on_policy" if agent_name in ON_POLICY_AGENTS else "off_policy"
    budget = FAMILY_BUDGETS[family]
    tag = f"{agent_name}_{tag_suffix}"
    t_start = time.time()
    print(f"\n{'='*70}\n{tag}: {budget['search_trials']}-trial search ({budget['search_timesteps']} steps each)\n{'='*70}", flush=True)

    rng = np.random.RandomState(C.SEED)
    space = SEARCH_SPACES[agent_name]
    best = None

    for trial in range(1, budget["search_trials"] + 1):
        config = {k: rng.choice(v).item() if hasattr(rng.choice(v), "item") else rng.choice(v) for k, v in space.items()}
        env = make_env(train_panel, training=True, seed=C.SEED + trial)
        model = build_agent(agent_name, env, config, seed=C.SEED + trial)
        model.learn(total_timesteps=budget["search_timesteps"], progress_bar=False)
        metrics = backtest(model, val_panel, env)
        print(f"  trial {trial}/{budget['search_trials']} | val_sharpe={metrics['sharpe']:.3f} | "
              f"val_return={metrics['cumulative_return_pct']:.2f}% | config={config}", flush=True)
        if best is None or metrics["sharpe"] > best["sharpe"]:
            best = {"config": config, "sharpe": metrics["sharpe"]}

    assert best is not None
    print(f"\nBest search config (val_sharpe={best['sharpe']:.3f}): {best['config']}", flush=True)
    print(f"Continuing training for up to {budget['final_timesteps']} steps, Sharpe-based early stopping (patience={PATIENCE})...", flush=True)

    env = make_env(train_panel, training=True, seed=C.SEED)
    model = build_agent(agent_name, env, best["config"], seed=C.SEED)
    callback = SharpeEvalCallback(val_panel, env, eval_freq=budget["eval_freq"], patience=PATIENCE, verbose=1)
    model.learn(total_timesteps=budget["final_timesteps"], reset_num_timesteps=False, progress_bar=False, callback=callback)
    callback.restore_best()

    model_path = os.path.join(C.MODELS_DIR, f"{tag}.zip")
    norm_path = os.path.join(C.MODELS_DIR, f"{tag}_vecnormalize.pkl")
    model.save(model_path)
    env.save(norm_path)

    test_metrics = backtest(model, test_panel, norm_path)
    elapsed = time.time() - t_start
    print(f"{tag} TEST: return={test_metrics['cumulative_return_pct']:.2f}% | "
          f"sharpe={test_metrics['sharpe']:.3f} | max_dd={test_metrics['max_drawdown_pct']:.2f}% | "
          f"benchmark_return={test_metrics['benchmark_return_pct']:.2f}% | time={elapsed:.0f}s", flush=True)

    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({"agent": agent_name, "version": "v3", **best["config"]})
        mlflow.log_metric("test_return_pct", test_metrics["cumulative_return_pct"])
        mlflow.log_metric("test_sharpe", test_metrics["sharpe"])
        mlflow.log_metric("test_max_drawdown_pct", test_metrics["max_drawdown_pct"])
        mlflow.log_metric("benchmark_return_pct", test_metrics["benchmark_return_pct"])
        mlflow.log_metric("best_val_sharpe", callback.best_sharpe)
        mlflow.log_artifact(model_path)

    append_to_experiment_log(tag, best["config"], callback.best_sharpe, test_metrics, elapsed)

    return {"agent": agent_name, **test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=list(AGENT_CLASSES.keys()), choices=list(AGENT_CLASSES.keys()))
    parser.add_argument("--tag-suffix", type=str, default="v3")
    args = parser.parse_args()

    train_panel, val_panel, test_panel = load_splits()

    results = []
    for agent_name in args.agents:
        result = train_one(agent_name, train_panel, val_panel, test_panel, args.tag_suffix)
        results.append(result)

    df = pd.DataFrame(results)
    out_path = os.path.join(C.RESULTS_DIR, f"{args.tag_suffix}_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
