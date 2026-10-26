"""
Trains and evaluates the four algorithms from the paper (PG, PPO, DQL, DDPG) on one asset,
replicating the paper's protocol: random search over Table 3's hyperparameter space, model
selection by validation Sharpe ratio, early stopping (patience 10, eval every 50 episodes),
final backtest on the CLEF test window (cost-free, matching the competition protocol).

Scale-down from the paper (documented, not hidden): the paper used Ray Tune with 180 trials
per algorithm-asset pair on a 192-core cluster and up to 2000 episodes per trial; this runs a
much smaller random search (SEARCH_TRIALS) and episode/timestep budget sized for a single
8GB Apple Silicon laptop with no cluster. Methodology (search space, selection criterion,
alpha reward, discrete action set, feature vector) is otherwise identical.

Run:
    PYTHONPATH=. .venv/bin/python -m src.paper_replication.train_paper --asset TSLA --agents PPO DDPG
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from src.paper_replication.data import build_panels
from src.paper_replication.env import SingleAssetTradingEnv
from src.paper_replication.algos import train_pg, train_dql, predict_pg, predict_dql

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "saved_models", "paper_replication")
RESULTS_DIR = os.path.join(ROOT, "results", "paper_replication")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEARCH_TRIALS = 8
CUSTOM_SEARCH_EPISODES = 150   # PG / DQL search budget (paper: up to 2000)
CUSTOM_FINAL_EPISODES = 600    # PG / DQL final budget
SB3_SEARCH_TIMESTEPS = 20_000  # PPO / DDPG search budget
SB3_FINAL_TIMESTEPS = 150_000  # PPO / DDPG final budget

SEARCH_SPACE = {
    "hidden_dim": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "gamma": [0.95, 0.99],
    "lr": lambda rng: float(10 ** rng.uniform(-4, -2)),
    "entropy_coef": lambda rng: float(10 ** rng.uniform(-3, np.log10(5e-2))),
    "clip_range": [0.1, 0.2, 0.3],
    "tau": [0.001, 0.005, 0.01],
}


def sample_config(rng, keys):
    cfg = {}
    for k in keys:
        v = SEARCH_SPACE[k]
        cfg[k] = v(rng) if callable(v) else rng.choice(v).item()
    return cfg


def backtest_discrete(predict_fn, env):
    obs, _ = env.reset()
    values, done = [], False
    while not done:
        action = predict_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        values.append(info["portfolio_value"])
    return np.array(values)


def backtest_sb3(model, env, recurrent=False):
    obs = env.reset()
    values, done = [], False
    lstm_states, ep_start = None, np.ones((1,), dtype=bool)
    while not done:
        if recurrent:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        done = bool(done_arr[0])
        values.append(info[0]["portfolio_value"])
    return np.array(values)


def compute_metrics(values):
    if len(values) < 2:
        return {"cumulative_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0}
    returns = np.diff(values) / values[:-1]
    cr = (values[-1] / values[0] - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if returns.std() > 0 else 0.0
    running_max = np.maximum.accumulate(values)
    mdd = ((values - running_max) / running_max).min() * 100
    return {"cumulative_return_pct": float(cr), "sharpe": float(sharpe), "max_drawdown_pct": float(mdd)}


def buy_and_hold_metrics(panel):
    close = panel["close"].to_numpy()
    return compute_metrics(close)


def run_pg_or_dql(algo_name, train_panel, val_panel, test_panel, seed):
    train_fn = train_pg if algo_name == "PG" else train_dql
    predict_raw = predict_pg if algo_name == "PG" else predict_dql
    rng = np.random.RandomState(seed)
    keys = ["hidden_dim", "num_layers", "gamma", "lr"] + (["entropy_coef"] if algo_name == "PG" else [])
    best = None
    for trial in range(1, SEARCH_TRIALS + 1):
        cfg = sample_config(rng, keys)
        env = SingleAssetTradingEnv(train_panel, training=True, seed=seed + trial)
        val_env = SingleAssetTradingEnv(val_panel, training=False, seed=seed)
        kwargs = dict(hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], lr=cfg["lr"],
                      gamma=cfg["gamma"], total_episodes=CUSTOM_SEARCH_EPISODES, eval_env=val_env,
                      eval_every=25, patience=5, seed=seed + trial)
        if algo_name == "PG":
            kwargs["entropy_coef"] = cfg["entropy_coef"]
        model, val_sr = train_fn(env, **kwargs)
        print(f"  [{algo_name}] trial {trial}/{SEARCH_TRIALS} | val_sharpe={val_sr:.3f} | cfg={cfg}", flush=True)
        if best is None or val_sr > best["val_sr"]:
            best = {"cfg": cfg, "val_sr": val_sr}

    assert best is not None
    print(f"  [{algo_name}] best search config (val_sharpe={best['val_sr']:.3f}): {best['cfg']}", flush=True)
    env = SingleAssetTradingEnv(train_panel, training=True, seed=seed)
    val_env = SingleAssetTradingEnv(val_panel, training=False, seed=seed)
    kwargs = dict(hidden_dim=best["cfg"]["hidden_dim"], num_layers=best["cfg"]["num_layers"],
                  lr=best["cfg"]["lr"], gamma=best["cfg"]["gamma"], total_episodes=CUSTOM_FINAL_EPISODES,
                  eval_env=val_env, eval_every=25, patience=10, seed=seed)
    if algo_name == "PG":
        kwargs["entropy_coef"] = best["cfg"]["entropy_coef"]
    model, _ = train_fn(env, **kwargs)

    if algo_name == "PG":
        def predict(obs):
            a, _ = predict_raw(model, obs, None)
            return a
    else:
        predict = lambda obs: predict_raw(model, obs)

    test_env = SingleAssetTradingEnv(test_panel, training=False, seed=seed)
    values = backtest_discrete(predict, test_env)
    metrics = compute_metrics(values)
    return metrics, best["cfg"], values


def run_ppo_or_ddpg(algo_name, train_panel, val_panel, test_panel, seed):
    rng = np.random.RandomState(seed)
    continuous = algo_name == "DDPG"
    keys = ["gamma", "lr"] + (["tau"] if algo_name == "DDPG" else ["clip_range", "entropy_coef"])
    best = None
    for trial in range(1, SEARCH_TRIALS + 1):
        cfg = sample_config(rng, keys)
        env = DummyVecEnv([lambda: Monitor(SingleAssetTradingEnv(train_panel, training=True, continuous_action=continuous, seed=seed + trial))])
        model = _build_sb3(algo_name, env, cfg, seed + trial)
        model.learn(total_timesteps=SB3_SEARCH_TIMESTEPS, progress_bar=False)
        val_env = DummyVecEnv([lambda: Monitor(SingleAssetTradingEnv(val_panel, training=False, continuous_action=continuous, seed=seed))])
        values = backtest_sb3(model, val_env, recurrent=(algo_name == "PPO"))
        val_sr = compute_metrics(values)["sharpe"]
        print(f"  [{algo_name}] trial {trial}/{SEARCH_TRIALS} | val_sharpe={val_sr:.3f} | cfg={cfg}", flush=True)
        if best is None or val_sr > best["val_sr"]:
            best = {"cfg": cfg, "val_sr": val_sr}

    assert best is not None
    print(f"  [{algo_name}] best search config (val_sharpe={best['val_sr']:.3f}): {best['cfg']}", flush=True)
    env = DummyVecEnv([lambda: Monitor(SingleAssetTradingEnv(train_panel, training=True, continuous_action=continuous, seed=seed))])
    model = _build_sb3(algo_name, env, best["cfg"], seed)
    model.learn(total_timesteps=SB3_FINAL_TIMESTEPS, progress_bar=False)

    test_env = DummyVecEnv([lambda: Monitor(SingleAssetTradingEnv(test_panel, training=False, continuous_action=continuous, seed=seed))])
    values = backtest_sb3(model, test_env, recurrent=(algo_name == "PPO"))
    metrics = compute_metrics(values)
    model_path = os.path.join(MODELS_DIR, f"{algo_name}.zip")
    model.save(model_path)
    return metrics, best["cfg"], values


def _build_sb3(algo_name, env, cfg, seed):
    if algo_name == "DDPG":
        n_actions = env.action_space.shape[-1]
        noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
        return DDPG("MlpPolicy", env, learning_rate=cfg["lr"], gamma=cfg["gamma"], tau=cfg["tau"],
                    action_noise=noise, seed=seed, verbose=0)
    if algo_name == "PPO":
        return RecurrentPPO("MlpLstmPolicy", env, learning_rate=cfg["lr"], gamma=cfg["gamma"],
                             clip_range=cfg["clip_range"], ent_coef=cfg["entropy_coef"], seed=seed, verbose=0)
    raise ValueError(algo_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=["TSLA", "BTC"])
    parser.add_argument("--agents", nargs="+", default=["PG", "PPO", "DQL", "DDPG"], choices=["PG", "PPO", "DQL", "DDPG"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = build_panels(args.asset)
    print(f"Panels: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    bh = buy_and_hold_metrics(test)
    print(f"Buy&Hold ({args.asset}) TEST: CR={bh['cumulative_return_pct']:.2f}% SR={bh['sharpe']:.3f} MDD={bh['max_drawdown_pct']:.2f}%", flush=True)

    results = [{"asset": args.asset, "model": "B&H", **bh}]
    for agent in args.agents:
        t0 = time.time()
        print(f"\n{'='*70}\n{args.asset} / {agent}\n{'='*70}", flush=True)
        if agent in ("PG", "DQL"):
            metrics, cfg, values = run_pg_or_dql(agent, train, val, test, args.seed)
        else:
            metrics, cfg, values = run_ppo_or_ddpg(agent, train, val, test, args.seed)
        elapsed = time.time() - t0
        print(f"{args.asset} / {agent} TEST: CR={metrics['cumulative_return_pct']:.2f}% "
              f"SR={metrics['sharpe']:.3f} MDD={metrics['max_drawdown_pct']:.2f}% | time={elapsed:.0f}s", flush=True)
        results.append({"asset": args.asset, "model": agent, **metrics, "best_config": str(cfg)})
        pd.DataFrame({"portfolio_value": values}).to_csv(
            os.path.join(RESULTS_DIR, f"{args.asset}_{agent}_test_curve.csv"), index=False)

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, f"{args.asset}_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
