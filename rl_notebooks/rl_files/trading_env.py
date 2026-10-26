# trading_env.py — Custom Gym environment with alpha reward
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Optional

from config import TRANSACTION_COST, INITIAL_VALUE, MIN_DAYS_LEFT, ACTION_MAP


class TradingEnv(gym.Env):
    """
    Custom trading environment implementing the MDP from the paper.

    State space:  15-dimensional vector (features from feature_eng.py)
    Action space: Discrete(3) → {0: SHORT, 1: FLAT, 2: LONG}
    Reward:       Alpha reward = log(V_t/V_{t-1}) - log(c_t/c_{t-1})
                  (excess return over buy-and-hold)

    Training: random episode start, transaction costs applied
    Eval:     fixed start (day 0), no transaction costs
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 states: np.ndarray,
                 returns: np.ndarray,
                 transaction_cost: float = TRANSACTION_COST,
                 initial_value: float = INITIAL_VALUE,
                 random_start: bool = True,
                 min_days_left: int = MIN_DAYS_LEFT,
                 apply_cost: bool = True):
        super().__init__()

        self.states           = states.astype(np.float32)
        self.returns          = returns.astype(np.float32)
        self.transaction_cost = transaction_cost
        self.initial_value    = initial_value
        self.random_start     = random_start
        self.min_days_left    = min_days_left
        self.apply_cost       = apply_cost
        self.T                = len(states)

        # Action: 0=SHORT, 1=FLAT, 2=LONG
        self.action_space      = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Action index → position {-1, 0, 1}
        self.action_to_pos = ACTION_MAP  # {0: -1, 1: 0, 2: 1}

        # Trajectory tracking
        self.portfolio_values = []
        self.positions        = []
        self.rewards          = []

        self.reset()

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            max_start = max(1, self.T - self.min_days_left)
            self.t = int(np.random.randint(1, max(2, max_start)))
        else:
            self.t = 1

        self.portfolio_value  = self.initial_value
        self.prev_position    = 0    # start FLAT
        self.portfolio_values = [self.initial_value]
        self.positions        = [0]
        self.rewards          = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs = self.states[self.t].copy()
        obs[1] = float(self.prev_position)   # update position feature
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert 0 <= action <= 2, f"Invalid action {action}"

        current_pos  = self.action_to_pos[action]
        daily_return = float(self.returns[self.t])

        # Transaction cost if position changed
        pos_changed = (current_pos != self.prev_position)
        cost = self.transaction_cost if (self.apply_cost and pos_changed) else 0.0

        # Portfolio update: V_t = V_{t-1} * (1 + p_{t-1} * ret_t) * (1 - cost)
        factor = (1 + current_pos * daily_return) * (1 - cost)
        new_value = max(self.portfolio_value * factor, 1.0)  # floor at $1

        # ── Alpha reward (paper's Equation 3) ─────────────────────────────
        # r_t = log(V_t / V_{t-1}) - log(c_t / c_{t-1})
        # = excess return over buy-and-hold market
        port_log_ret   = np.log(new_value / self.portfolio_value + 1e-10)
        market_log_ret = np.log(1.0 + daily_return + 1e-10)
        alpha_reward   = float(port_log_ret - market_log_ret)

        # Update state
        self.portfolio_value = new_value
        self.prev_position   = current_pos
        self.t              += 1

        # Record
        self.portfolio_values.append(new_value)
        self.positions.append(current_pos)
        self.rewards.append(alpha_reward)

        done = (self.t >= self.T - 1)
        obs  = self._get_obs() if not done else np.zeros(15, dtype=np.float32)

        info = {
            "portfolio_value": new_value,
            "daily_return":    daily_return,
            "position":        current_pos,
            "alpha_reward":    alpha_reward,
            "cost_incurred":   cost > 0,
        }

        return obs, alpha_reward, done, False, info

    def get_metrics(self) -> Dict:
        """Compute performance metrics for the current episode."""
        values  = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        cum_ret  = (values[-1] / values[0] - 1) * 100
        ann_vol  = np.std(returns) * np.sqrt(252) * 100
        sharpe   = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)
        peak     = np.maximum.accumulate(values)
        mdd      = abs(((values - peak) / peak).min()) * 100

        # Position counts
        pos_arr = np.array(self.positions)
        n_long  = (pos_arr == 1).sum()
        n_flat  = (pos_arr == 0).sum()
        n_short = (pos_arr == -1).sum()
        n_trades = sum(
            1 for i in range(1, len(pos_arr))
            if pos_arr[i] != pos_arr[i-1]
        )

        return {
            "cumulative_return": round(cum_ret, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown":      round(mdd, 2),
            "ann_volatility":    round(ann_vol, 2),
            "final_value":       round(values[-1], 2),
            "n_long":  int(n_long),
            "n_flat":  int(n_flat),
            "n_short": int(n_short),
            "n_trades": int(n_trades),
        }

    def compute_bnh_metrics(self) -> Dict:
        """Compute buy-and-hold baseline metrics."""
        bnh_returns = self.returns[1:len(self.portfolio_values)]
        bnh_values  = self.initial_value * np.cumprod(1 + bnh_returns)
        bnh_values  = np.concatenate([[self.initial_value], bnh_values])

        cum_ret = (bnh_values[-1] / bnh_values[0] - 1) * 100
        rets    = np.diff(bnh_values) / bnh_values[:-1]
        sharpe  = (np.mean(rets) / (np.std(rets) + 1e-10)) * np.sqrt(252)
        peak    = np.maximum.accumulate(bnh_values)
        mdd     = abs(((bnh_values - peak) / peak).min()) * 100

        return {
            "cumulative_return": round(cum_ret, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown":      round(mdd, 2),
            "final_value":       round(bnh_values[-1], 2),
        }
