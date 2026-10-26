"""
Ablation: isolates whether the ALPHA REWARD itself (as opposed to v3's new feature set) caused
SAC's regression. v1's exact features/mechanics (raw MACD/RSI-14/CCI-20/ADX-14, sentiment,
no calendar/cross-sectional/vol-regime, no random starts) + alpha reward instead of v1's raw
portfolio-return reward. Same SAC hyperparameters as v3's winning config, to hold everything
but the reward/features variable constant across this and the random-start ablation.

Run: PYTHONPATH=. .venv/bin/python -m src.rl.ablation_alpha_reward_only
"""
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from src.rl import config as C
from src.rl.train_v1 import load_splits
from src.rl.trading_env_alpha_only import PortfolioTradingEnvAlphaOnly
from src.rl.train_v3 import SharpeEvalCallback, append_to_experiment_log

SAC_CONFIG = {"learning_rate": 0.0001, "batch_size": 512, "tau": 0.001, "gamma": 0.99}  # same as v3's winning config


def make_env(panel_split, use_sentiment, training):
    def _init():
        return Monitor(PortfolioTradingEnvAlphaOnly(panel_split, use_sentiment=use_sentiment))
    vec_env = DummyVecEnv([_init])
    return VecNormalize(vec_env, norm_obs=True, norm_reward=training, clip_obs=10.0)


def backtest(model, panel_split, norm_env_or_path):
    eval_env = make_env(panel_split, True, training=False)
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
        obs, reward, done_arr, info = eval_env.step(action)
        done = bool(done_arr[0])
        values.append(info[0]["portfolio_value"])
        bh_values.append(info[0]["benchmark_value"])
    values = np.array(values)
    bh_values = np.array(bh_values)
    returns = np.diff(values) / values[:-1]
    cr = (values[-1] / C.INITIAL_CASH - 1) * 100
    bh_cr = (bh_values[-1] / C.INITIAL_CASH - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(values)
    mdd = ((values - running_max) / running_max).min() * 100
    return {"cumulative_return_pct": cr, "sharpe": sharpe, "max_drawdown_pct": mdd, "benchmark_return_pct": bh_cr}


def main():
    train_panel, val_panel, test_panel = load_splits()
    t0 = time.time()

    env = make_env(train_panel, True, training=True)
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = SAC("MlpPolicy", env, seed=C.SEED, verbose=0, action_noise=noise, **SAC_CONFIG)

    callback = SharpeEvalCallback(val_panel, env, eval_freq=20_000, patience=8, verbose=1)
    # SharpeEvalCallback calls train_v3.backtest internally; monkeypatch-free approach: reimplement eval inline
    callback.backtest_fn = None

    class _Callback(SharpeEvalCallback):
        def _on_step(self):
            if self.n_calls % self.eval_freq != 0:
                return True
            metrics = backtest(self.model, self.val_panel, self.train_env)
            sharpe = metrics["sharpe"]
            self.history.append({"step": self.n_calls, "val_sharpe": sharpe, "val_return": metrics["cumulative_return_pct"]})
            print(f"    [eval @ {self.n_calls}] val_sharpe={sharpe:.3f} val_return={metrics['cumulative_return_pct']:.2f}%", flush=True)
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_weights = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                print(f"    early stopping: no improvement for {self.patience} evals", flush=True)
                return False
            return True

    callback = _Callback(val_panel, env, eval_freq=20_000, patience=8, verbose=1)
    model.learn(total_timesteps=400_000, reset_num_timesteps=False, progress_bar=False, callback=callback)
    callback.restore_best()

    tag = "SAC_ablation_alphareward_v1features"
    model_path = f"{C.MODELS_DIR}/{tag}.zip"
    norm_path = f"{C.MODELS_DIR}/{tag}_vecnormalize.pkl"
    model.save(model_path)
    env.save(norm_path)

    test_metrics = backtest(model, test_panel, norm_path)
    elapsed = time.time() - t0
    print(f"{tag} TEST: return={test_metrics['cumulative_return_pct']:.2f}% | "
          f"sharpe={test_metrics['sharpe']:.3f} | max_dd={test_metrics['max_drawdown_pct']:.2f}% | "
          f"benchmark_return={test_metrics['benchmark_return_pct']:.2f}% | time={elapsed:.0f}s", flush=True)

    append_to_experiment_log(tag, {**SAC_CONFIG, "features": "v1 (unnormalized)", "reward": "alpha"}, callback.best_sharpe, test_metrics, elapsed)
    print("\nComparison:")
    print("  v1 SAC (v1 features, raw reward):              return=-3.12%  sharpe=-0.014")
    print("  v3 SAC (v3 features, alpha reward, rand start): return=-8.37%  sharpe=-0.506")
    print("  v3 SAC ablation (v3 features, alpha, full ep):  return=-11.96% sharpe=-0.775")
    print(f"  this: SAC (v1 features, alpha reward):          return={test_metrics['cumulative_return_pct']:.2f}%  sharpe={test_metrics['sharpe']:.3f}")


if __name__ == "__main__":
    main()
