"""
trading_env_v3.py — multi-asset portfolio env, v1's mechanics (dollar-based trading,
transaction costs, continuous per-asset weights) plus the paper-inspired improvements from
EXPERIMENT_LOG.md that v1 didn't have:

  - Alpha reward: r_t = log(V_t/V_{t-1}) - log(BH_t/BH_{t-1}) -- portfolio return minus the
    equal-weight buy-and-hold benchmark's return, instead of v1's raw portfolio return. Directly
    trains the objective actually being evaluated (beat the benchmark), not just "grow the
    portfolio" -- matters most exactly when the market itself is falling, since raw-return
    training has no signal telling the agent "everyone's losing money, losing less is a win."
  - Holdings expressed as a fraction of portfolio value (not raw share count) in the
    observation -- 1000 shares of Rs170 ITC and 10 shares of Rs6650 Maruti aren't comparable
    numbers for one shared network; this matches the price-normalized indicators already
    computed in panel_v3.py.
  - Random episode starts during training (any day with enough days remaining) -- full-period
    rollout always used for evaluation. Targets the same overfitting pattern already observed
    empirically in this project (SAC's 25-trial search beating validation but losing to the
    original 10-trial model on the real test set).
  - No separate turnover penalty on top of the reward (v1 had one). Transaction costs are
    already reflected in V_t itself; stacking an extra ad hoc penalty term muddies the alpha
    reward's theoretical variance-reduction property (subtracting a policy-independent market
    return as a control variate) for a benefit v1's own penalty was only weakly justified for.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.rl import config as C
from src.rl.panel_v3 import PER_ASSET_COLS, GLOBAL_COLS

MIN_DAYS_REMAINING_FOR_START = 60


class PortfolioTradingEnvV3(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel: pd.DataFrame, tickers=C.TICKERS, training: bool = True,
                 initial_cash: float = C.INITIAL_CASH, max_trade_frac: float = 0.15,
                 transaction_cost_pct: float = C.TRANSACTION_COST_PCT, seed: int | None = None,
                 random_start: bool | None = None):
        super().__init__()
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.training = training
        # Separate from `training` so an ablation can isolate this one variable: keep
        # training-mode transaction costs but force full-length episodes like v1 had.
        self.random_start = training if random_start is None else random_start
        self.initial_cash = initial_cash
        self.max_trade_frac = max_trade_frac
        self.transaction_cost_pct = transaction_cost_pct
        self.rng = np.random.RandomState(seed)

        self.close_prices = panel.pivot(index="date", columns="ticker", values="close")[self.tickers]
        self.per_asset = {
            col: panel.pivot(index="date", columns="ticker", values=col)[self.tickers]
            for col in PER_ASSET_COLS
        }
        global_src = panel.drop_duplicates("date").set_index("date")
        self.global_feat = global_src[GLOBAL_COLS].reindex(self.close_prices.index)

        self.dates = self.close_prices.index.sort_values()
        self.n_steps = len(self.dates) - 1

        bh_shares = (self.initial_cash / self.n_assets) / self.close_prices.iloc[0]
        self.bh_value = (self.close_prices * bh_shares).sum(axis=1).to_numpy()

        n_global = 1 + len(GLOBAL_COLS)  # cash_frac + calendar/vol features
        n_per_asset = 1 + len(PER_ASSET_COLS)  # holdings_frac + technical/sentiment features
        obs_dim = n_global + n_per_asset * self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def _get_prices(self, t):
        return self.close_prices.iloc[t].to_numpy(dtype=np.float32)

    def _portfolio_value(self, t):
        return float(self.cash + np.dot(self.holdings, self._get_prices(t)))

    def _get_obs(self, t):
        pv = self._portfolio_value(t)
        cash_frac = self.cash / pv if pv > 0 else 0.0
        holdings_frac = (self.holdings * self._get_prices(t)) / pv if pv > 0 else np.zeros(self.n_assets)

        global_vals = self.global_feat.iloc[t].to_numpy(dtype=np.float32)
        parts = [np.array([cash_frac], dtype=np.float32), global_vals, holdings_frac.astype(np.float32)]
        for col in PER_ASSET_COLS:
            parts.append(self.per_asset[col].iloc[t].to_numpy(dtype=np.float32))
        obs = np.concatenate(parts)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            max_start = max(1, self.n_steps - MIN_DAYS_REMAINING_FOR_START)
            self.t = int(self.rng.randint(0, max_start)) if max_start > 1 else 0
        else:
            self.t = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.episode_start_t = self.t
        return self._get_obs(self.t), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        prices = self._get_prices(self.t)
        portfolio_value = self._portfolio_value(self.t)

        target_dollar_trade = action * self.max_trade_frac * portfolio_value
        trade_shares = np.where(prices > 0, target_dollar_trade / np.maximum(prices, 1e-9), 0.0)
        trade_shares = np.round(trade_shares)

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
        bh_prev = self.bh_value[self.t]
        self.t += 1
        terminated = self.t >= self.n_steps
        t_clamped = min(self.t, self.n_steps)
        new_value = self._portfolio_value(t_clamped)
        bh_new = self.bh_value[t_clamped]

        portfolio_log_ret = np.log(max(new_value, 1e-9) / max(prev_value, 1e-9))
        bh_log_ret = np.log(max(bh_new, 1e-9) / max(bh_prev, 1e-9))
        reward = float((portfolio_log_ret - bh_log_ret) * 100.0)

        truncated = self.random_start and (self.t - self.episode_start_t) >= (self.n_steps - MIN_DAYS_REMAINING_FOR_START + 1)
        obs = self._get_obs(t_clamped)
        info = {"portfolio_value": new_value, "benchmark_value": bh_new,
                "date": str(self.dates[t_clamped].date())}
        return obs, reward, bool(terminated), bool(truncated), info
