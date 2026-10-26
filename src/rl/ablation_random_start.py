"""
Ablation: isolates whether v3's random episode starts caused SAC's regression vs. v1
(see EXPERIMENT_LOG.md v3 summary). Same alpha reward, same normalized/cross-sectional/
volatility/sentiment features as v3 -- only difference is random_start=False, i.e. full-length
training episodes like v1 used. If SAC recovers toward v1's -3.12%/-0.014, random starts are
the likely cause; if it doesn't, the regression is coming from somewhere else (reward shape,
features, or just this agent's sensitivity to alpha reward specifically).

Run: PYTHONPATH=. .venv/bin/python -m src.rl.ablation_random_start
"""
import time

from src.rl import config as C
from src.rl.train_v3 import (
    load_splits, make_env, build_agent, backtest, SharpeEvalCallback,
    FAMILY_BUDGETS, PATIENCE, append_to_experiment_log,
)

SAC_BEST_CONFIG = {"learning_rate": 0.0001, "batch_size": 512, "tau": 0.001, "gamma": 0.99}  # v3's winning search config


def main():
    train_panel, val_panel, test_panel = load_splits()
    budget = FAMILY_BUDGETS["off_policy"]
    t0 = time.time()

    env = make_env(train_panel, training=True, seed=C.SEED, random_start=False)
    model = build_agent("SAC", env, SAC_BEST_CONFIG, seed=C.SEED)
    callback = SharpeEvalCallback(val_panel, env, eval_freq=budget["eval_freq"], patience=PATIENCE, verbose=1)
    model.learn(total_timesteps=budget["final_timesteps"], reset_num_timesteps=False, progress_bar=False, callback=callback)
    callback.restore_best()

    tag = "SAC_v3_ablation_fullep"
    model_path = f"{C.MODELS_DIR}/{tag}.zip"
    norm_path = f"{C.MODELS_DIR}/{tag}_vecnormalize.pkl"
    model.save(model_path)
    env.save(norm_path)

    test_metrics = backtest(model, test_panel, norm_path)
    elapsed = time.time() - t0
    print(f"{tag} TEST: return={test_metrics['cumulative_return_pct']:.2f}% | "
          f"sharpe={test_metrics['sharpe']:.3f} | max_dd={test_metrics['max_drawdown_pct']:.2f}% | "
          f"benchmark_return={test_metrics['benchmark_return_pct']:.2f}% | time={elapsed:.0f}s", flush=True)

    append_to_experiment_log(tag, {**SAC_BEST_CONFIG, "random_start": False}, callback.best_sharpe, test_metrics, elapsed)
    print("\nComparison:")
    print(f"  v1 SAC (full episodes, raw reward):        return=-3.12%  sharpe=-0.014")
    print(f"  v3 SAC (random starts, alpha reward):       return=-8.37%  sharpe=-0.506")
    print(f"  v3 SAC ablation (full episodes, alpha reward): return={test_metrics['cumulative_return_pct']:.2f}%  sharpe={test_metrics['sharpe']:.3f}")


if __name__ == "__main__":
    main()
