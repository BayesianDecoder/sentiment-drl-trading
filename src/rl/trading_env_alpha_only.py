"""
Isolates the alpha-reward variable from the v3 regression investigation (EXPERIMENT_LOG.md):
v1's exact environment, features, and mechanics (raw MACD/RSI-14/CCI-20/ADX-14, no calendar/
cross-sectional/volatility-regime/sentiment-smoothing, no random episode starts, raw share-
count holdings) -- the ONLY change is the reward, swapped from v1's raw portfolio log-return
to the alpha reward (portfolio return minus equal-weight buy-and-hold return). If SAC recovers
toward v1's -3.12%/-0.014 here, the regression traces to alpha reward specifically; if not, it
traces to v3's new feature set instead.
"""
import numpy as np

from src.rl.trading_env import PortfolioTradingEnv


class PortfolioTradingEnvAlphaOnly(PortfolioTradingEnv):
    def __init__(self, panel, **kwargs):
        super().__init__(panel, **kwargs)
        bh_shares = (self.initial_cash / self.n_assets) / self.close.iloc[0]
        self.bh_value = (self.close * bh_shares).sum(axis=1).to_numpy()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        prices = self._get_prices(self.t)

        target_dollar_trade = action * self.max_trade_frac * self._portfolio_value(self.t)
        trade_shares = np.where(prices > 0, target_dollar_trade / np.maximum(prices, 1e-9), 0.0)
        trade_shares = np.round(trade_shares)

        day_shares_traded = 0.0
        day_transaction_cost = 0.0
        for i in np.argsort(trade_shares):
            shares, price = trade_shares[i], prices[i]
            if price <= 0 or np.isnan(price):
                continue
            if shares < 0:
                sell = min(-shares, self.holdings[i])
                self.cash += sell * price * (1 - self.transaction_cost_pct)
                self.holdings[i] -= sell
                day_shares_traded += sell
                day_transaction_cost += sell * price * self.transaction_cost_pct
            elif shares > 0:
                cost_per_share = price * (1 + self.transaction_cost_pct)
                affordable = np.floor(self.cash / cost_per_share) if cost_per_share > 0 else 0
                buy = min(shares, affordable)
                self.cash -= buy * cost_per_share
                self.holdings[i] += buy
                day_shares_traded += buy
                day_transaction_cost += buy * price * self.transaction_cost_pct

        self.total_shares_traded += day_shares_traded
        self.total_transaction_cost += day_transaction_cost
        if day_shares_traded > 0:
            self.n_trade_days += 1

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

        obs = self._get_obs(t_clamped)
        info = {"portfolio_value": new_value, "benchmark_value": bh_new,
                "date": str(self.dates[t_clamped].date())}
        if terminated:
            info["total_shares_traded"] = self.total_shares_traded
            info["total_transaction_cost"] = self.total_transaction_cost
            info["n_trade_days"] = self.n_trade_days
        return obs, reward, terminated, False, info
