"""
Loads each finished RL notebook's saved model + VecNormalize stats, re-runs the exact same
backtest logic the notebook itself used (dollar-consistent trade sizing + log-return reward
env — NOT the older src/rl/trading_env.py version, which the notebooks superseded), and pushes
the resulting metrics + portfolio-value curves to the Express API so the dashboard can chart them.

The notebooks already printed their own final metrics; this script's numbers are a consistency
check against those, plus the raw curve data the notebooks didn't persist to disk.

Run: .venv/bin/python scripts/collect_rl_results.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import requests
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")
PANEL_PATH = os.path.join(ROOT, "data", "processed", "rl_panel.csv")
EXPRESS_API = os.environ.get("EXPRESS_API_URL", "http://localhost:4000/api")

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG, "SAC": SAC, "TD3": TD3}
TICKERS = ["INFY.NS", "RELIANCE.NS", "TCS.NS", "SBIN.NS", "MARUTI.NS",
           "WIPRO.NS", "TMPV.NS", "LT.NS", "ITC.NS", "SUNPHARMA.NS"]
INDICATORS = ["macd", "rsi_14", "cci_20", "adx_14"]
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
INITIAL_CASH = 1_000_000.0
MAX_TRADE_FRACTION = 0.15
TRANSACTION_COST_PCT = 0.001
REWARD_LOG_SCALE = 100.0
TURNOVER_PENALTY_COEF = 0.01


# ── Same env class as embedded in the notebooks (gen_rl_notebooks.py) — NOT src/rl/trading_env.py ──
class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel_split, tickers=TICKERS, use_sentiment=True,
                 initial_cash=INITIAL_CASH, max_trade_frac=MAX_TRADE_FRACTION,
                 transaction_cost_pct=TRANSACTION_COST_PCT,
                 turnover_penalty_coef=TURNOVER_PENALTY_COEF, reward_log_scale=REWARD_LOG_SCALE):
        super().__init__()
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.use_sentiment = use_sentiment
        self.initial_cash = initial_cash
        self.max_trade_frac = max_trade_frac
        self.transaction_cost_pct = transaction_cost_pct
        self.turnover_penalty_coef = turnover_penalty_coef
        self.reward_log_scale = reward_log_scale
        self.indicator_cols = list(INDICATORS) + (["sentiment"] if use_sentiment else [])
        self.close = panel_split.pivot(index="date", columns="ticker", values="close")[self.tickers]
        self.indicators = {c: panel_split.pivot(index="date", columns="ticker", values=c)[self.tickers] for c in self.indicator_cols}
        self.dates = self.close.index.sort_values()
        self.n_steps = len(self.dates) - 1
        n_feat = 2 + len(self.indicator_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1 + n_feat * self.n_assets,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.reset()

    def _get_prices(self, t):
        return self.close.iloc[t].to_numpy(dtype=np.float32)

    def _get_obs(self, t):
        prices = self._get_prices(t)
        parts = [np.array([self.cash], dtype=np.float32), self.holdings.astype(np.float32), prices]
        for col in self.indicator_cols:
            parts.append(self.indicators[col].iloc[t].to_numpy(dtype=np.float32))
        return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)

    def _portfolio_value(self, t):
        return float(self.cash + np.dot(self.holdings, self._get_prices(t)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        return self._get_obs(self.t), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        prices = self._get_prices(self.t)
        portfolio_value = self._portfolio_value(self.t)
        target_dollar = action * self.max_trade_frac * portfolio_value
        trade_shares = np.round(np.where(prices > 0, target_dollar / np.maximum(prices, 1e-9), 0.0))
        for i in np.argsort(trade_shares):
            shares, price = trade_shares[i], prices[i]
            if price <= 0 or np.isnan(price):
                continue
            if shares < 0:
                sell = min(-shares, self.holdings[i])
                self.cash += sell * price * (1 - self.transaction_cost_pct)
                self.holdings[i] -= sell
            elif shares > 0:
                cost_per_share = price * (1 + self.transaction_cost_pct)
                affordable = np.floor(self.cash / cost_per_share) if cost_per_share > 0 else 0
                buy = min(shares, affordable)
                self.cash -= buy * cost_per_share
                self.holdings[i] += buy
        prev_value = self._portfolio_value(self.t)
        self.t += 1
        terminated = self.t >= self.n_steps
        new_value = self._portfolio_value(min(self.t, self.n_steps))
        reward = np.log(max(new_value, 1e-9) / max(prev_value, 1e-9)) * self.reward_log_scale
        obs = self._get_obs(min(self.t, self.n_steps))
        return obs, reward, terminated, False, {"portfolio_value": new_value}


def load_splits():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    train_end = dates[int(n * TRAIN_FRAC)]
    val_end = dates[int(n * (TRAIN_FRAC + VAL_FRAC))]
    test = panel[panel["date"] > val_end].copy()
    return test


def make_env(panel_split):
    def _init():
        return Monitor(PortfolioTradingEnv(panel_split))
    return DummyVecEnv([_init])


def backtest(model, panel_split, norm_path):
    eval_env = make_env(panel_split)
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    obs = eval_env.reset()
    values, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        done = bool(done[0])
        values.append(info[0]["portfolio_value"])
    return np.array(values)


def compute_metrics(values):
    returns = np.diff(values) / values[:-1]
    cumulative_return_pct = (values[-1] / values[0] - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(values)
    max_dd = ((values - running_max) / running_max).min() * 100
    return cumulative_return_pct, sharpe, max_dd


def buy_and_hold(panel_split):
    close = panel_split.pivot(index="date", columns="ticker", values="close")[TICKERS].sort_index()
    shares = (INITIAL_CASH / len(TICKERS)) / close.iloc[0]
    return (close * shares).sum(axis=1).to_numpy()


def main():
    test_panel = load_splits()
    bh_values = buy_and_hold(test_panel)
    bh_return, bh_sharpe, bh_dd = compute_metrics(bh_values)
    bh_normalized = (bh_values / bh_values[0]).tolist()

    for agent in AGENT_CLASSES:
        model_path = os.path.join(NOTEBOOKS_DIR, "saved_models", f"{agent}_sentiment.zip")
        norm_path = os.path.join(NOTEBOOKS_DIR, "saved_models", f"{agent}_sentiment_vecnormalize.pkl")
        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            print(f"{agent}: not finished yet, skipping")
            continue

        model = AGENT_CLASSES[agent].load(model_path)
        values = backtest(model, test_panel, norm_path)
        ret, sharpe, dd = compute_metrics(values)

        payload = {
            "test_return_pct": ret,
            "test_sharpe": sharpe,
            "test_max_drawdown_pct": dd,
            "beat_buy_and_hold": bool(ret > bh_return and sharpe > bh_sharpe),
            "portfolio_curve": (values / values[0]).tolist(),
            "buy_hold_curve": bh_normalized,
            "status": "complete",
        }
        resp = requests.put(f"{EXPRESS_API}/rl-models/{agent}", json=payload)
        print(f"{agent}: return={ret:.2f}% sharpe={sharpe:.3f} max_dd={dd:.2f}% "
              f"-> pushed to Express, status={resp.status_code}")


if __name__ == "__main__":
    main()
