"""
trading_env.py — Part B (RL trading agents), v1 (dashboard-backing design)

A Gymnasium multi-asset portfolio trading environment.

    state = [cash] + [holdings per asset] + [close price per asset]
            + [technical indicators per asset] + [sentiment per asset]  (optional)

    action = continuous vector in [-1, 1] per asset. action[i] maps to a FRACTION OF PORTFOLIO
             VALUE to trade in asset i (not a flat share count) -- this keeps a given action
             magnitude meaning roughly the same dollar impact regardless of a ticker's per-share
             price (₹400 ITC vs ₹11,000+ MARUTI would otherwise get wildly different exposure
             for the same action value).

    reward = log-return of portfolio value, minus a small turnover penalty to discourage
             excessive churn (an earlier raw-currency-delta reward let agents trade on 90%+ of
             days and bleed real money to transaction costs regardless of trade quality).

This is the version actually trained, backtested, and wired into the dashboard.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.rl import config as C


class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel: pd.DataFrame, tickers=C.TICKERS, use_sentiment: bool = True,
                 initial_cash: float = C.INITIAL_CASH, max_trade_frac: float = 0.15,
                 transaction_cost_pct: float = C.TRANSACTION_COST_PCT,
                 turnover_penalty_coef: float = 0.01, reward_log_scale: float = 100.0):
        super().__init__()
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.use_sentiment = use_sentiment
        self.initial_cash = initial_cash
        self.max_trade_frac = max_trade_frac
        self.transaction_cost_pct = transaction_cost_pct
        self.turnover_penalty_coef = turnover_penalty_coef
        self.reward_log_scale = reward_log_scale

        self.indicator_cols = list(C.INDICATORS) + (["sentiment"] if use_sentiment else [])

        self.close = panel.pivot(index="date", columns="ticker", values="close")[self.tickers]
        self.indicators = {
            col: panel.pivot(index="date", columns="ticker", values=col)[self.tickers]
            for col in self.indicator_cols
        }
        self.dates = self.close.index.sort_values()
        self.n_steps = len(self.dates) - 1

        n_features_per_asset = 2 + len(self.indicator_cols)
        obs_dim = 1 + n_features_per_asset * self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        self.reset()

    def _get_prices(self, t):
        return self.close.iloc[t].to_numpy(dtype=np.float32)

    def _get_obs(self, t):
        prices = self._get_prices(t)
        parts = [np.array([self.cash], dtype=np.float32), self.holdings.astype(np.float32), prices]
        for col in self.indicator_cols:
            parts.append(self.indicators[col].iloc[t].to_numpy(dtype=np.float32))
        obs = np.concatenate(parts)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _portfolio_value(self, t):
        return float(self.cash + np.dot(self.holdings, self._get_prices(t)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.total_shares_traded = 0.0
        self.total_transaction_cost = 0.0
        self.n_trade_days = 0
        return self._get_obs(self.t), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        prices = self._get_prices(self.t)
        portfolio_value = self._portfolio_value(self.t)

        target_dollar_trade = action * self.max_trade_frac * portfolio_value
        trade_shares = np.where(prices > 0, target_dollar_trade / np.maximum(prices, 1e-9), 0.0)
        trade_shares = np.round(trade_shares)

        day_shares_traded = 0.0
        day_transaction_cost = 0.0
        day_dollar_traded = 0.0

        for i in np.argsort(trade_shares):   # sells (negative) first, frees up cash for buys
            shares, price = trade_shares[i], prices[i]
            if price <= 0 or np.isnan(price):
                continue
            if shares < 0:
                sell = min(-shares, self.holdings[i])
                self.cash += sell * price * (1 - self.transaction_cost_pct)
                self.holdings[i] -= sell
                day_shares_traded += sell
                day_transaction_cost += sell * price * self.transaction_cost_pct
                day_dollar_traded += sell * price
            elif shares > 0:
                cost_per_share = price * (1 + self.transaction_cost_pct)
                affordable = np.floor(self.cash / cost_per_share) if cost_per_share > 0 else 0
                buy = min(shares, affordable)
                self.cash -= buy * cost_per_share
                self.holdings[i] += buy
                day_shares_traded += buy
                day_transaction_cost += buy * price * self.transaction_cost_pct
                day_dollar_traded += buy * price

        self.total_shares_traded += day_shares_traded
        self.total_transaction_cost += day_transaction_cost
        if day_shares_traded > 0:
            self.n_trade_days += 1

        prev_value = self._portfolio_value(self.t)
        self.t += 1
        terminated = self.t >= self.n_steps
        new_value = self._portfolio_value(min(self.t, self.n_steps))

        log_return = np.log(max(new_value, 1e-9) / max(prev_value, 1e-9))
        turnover_frac = day_dollar_traded / max(prev_value, 1e-9)
        reward = log_return * self.reward_log_scale - self.turnover_penalty_coef * turnover_frac

        obs = self._get_obs(min(self.t, self.n_steps))
        info = {
            "portfolio_value": new_value,
            "date": str(self.dates[min(self.t, self.n_steps)].date()),
            "day_shares_traded": day_shares_traded,
            "day_transaction_cost": day_transaction_cost,
        }
        if terminated:
            info["total_shares_traded"] = self.total_shares_traded
            info["total_transaction_cost"] = self.total_transaction_cost
            info["n_trade_days"] = self.n_trade_days
            info["n_steps"] = self.n_steps
        return obs, reward, terminated, False, info
