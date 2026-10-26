"""
Single-asset discrete-action trading environment replicating the paper's MDP (Section 3) and
alpha reward (Section 4.4).

Action space: {-1 SHORT, 0 FLAT, +1 LONG}. Portfolio value evolves per Eq. 1:
    V_t = V_{t-1} * (1 + p_{t-1} * return_t) * (1 - delta * 1{a_t != p_{t-1}})
Reward is the alpha reward (Eq. 3): r_t = log(V_t/V_{t-1}) - log(c_t/c_{t-1}) -- excess
log-return over buy-and-hold, zero under a passive market-tracking strategy.

Training uses random episode starts (any day with >=100 days remaining) and a 0.2% transaction
cost; evaluation always starts at day 0 of the split and is cost-free, matching the paper's
CLEF Task 3 submission protocol (Section 5.2).

Training episodes are also capped at max_episode_steps (default 250). The paper's own PPO uses
truncated back-propagation through time for exactly this reason -- an LSTM policy backpropagated
through an untruncated ~3000-step episode (TSLA's full training span) is prohibitively slow/
memory-heavy; the cap keeps BPTT bounded for the custom LSTM-PG implementation while full-length
rollouts are still used for validation/test evaluation (max_episode_steps has no effect there).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.paper_replication.features import STATE_COLS

TRANSACTION_COST = 0.002
MIN_DAYS_REMAINING_FOR_START = 100
DEFAULT_MAX_EPISODE_STEPS = 250


class SingleAssetTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel, training: bool, continuous_action: bool = False, seed: int | None = None,
                 max_episode_steps: int | None = None):
        super().__init__()
        self.panel = panel.reset_index(drop=True)
        self.n_steps = len(self.panel) - 1
        self.training = training
        self.continuous_action = continuous_action
        self.transaction_cost = TRANSACTION_COST if training else 0.0
        self.max_episode_steps = (max_episode_steps if max_episode_steps is not None
                                   else (DEFAULT_MAX_EPISODE_STEPS if training else None))
        self.rng = np.random.RandomState(seed)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(STATE_COLS),), dtype=np.float32)
        if continuous_action:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)  # 0=SHORT(-1), 1=FLAT(0), 2=LONG(+1)

        self.close_prices = self.panel["close"].to_numpy(dtype=np.float64)
        self.simple_return = np.zeros(len(self.panel))
        self.simple_return[1:] = (self.close_prices[1:] - self.close_prices[:-1]) / self.close_prices[:-1]
        self.log_market_return = np.zeros(len(self.panel))
        self.log_market_return[1:] = np.log(self.close_prices[1:] / self.close_prices[:-1])

        self.feat = self.panel[[c for c in STATE_COLS if c != "position"]].to_numpy(dtype=np.float32)

    def _decode_action(self, action):
        if self.continuous_action:
            a = float(np.clip(action, -1.0, 1.0)[0] if hasattr(action, "__len__") else action)
            if a < -1 / 3:
                return -1.0
            if a > 1 / 3:
                return 1.0
            return 0.0
        return {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]

    def _get_obs(self, t, position):
        row = self.feat[t]
        out = np.zeros(len(STATE_COLS), dtype=np.float32)
        j = 0
        for i, c in enumerate(STATE_COLS):
            if c == "position":
                out[i] = position
            else:
                out[i] = row[j]
                j += 1
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.training:
            max_start = max(1, self.n_steps - MIN_DAYS_REMAINING_FOR_START)
            self.t = int(self.rng.randint(1, max_start)) if max_start > 1 else 1
        else:
            self.t = 1
        self.position = 0.0
        self.value = 1.0
        self._episode_step_count = 0
        return self._get_obs(self.t, self.position), {}

    def step(self, action):
        a_signed = self._decode_action(action)
        ret = self.simple_return[self.t]
        prev_value = self.value
        new_value = prev_value * (1 + self.position * ret) * (1 - self.transaction_cost * (a_signed != self.position))
        market_log_ret = self.log_market_return[self.t]
        portfolio_log_ret = np.log(max(new_value, 1e-9) / max(prev_value, 1e-9))
        reward = portfolio_log_ret - market_log_ret

        self.value = new_value
        self.position = a_signed
        self.t += 1
        self._episode_step_count += 1
        terminated = self.t >= self.n_steps
        truncated = self.max_episode_steps is not None and self._episode_step_count >= self.max_episode_steps
        obs = self._get_obs(min(self.t, self.n_steps), self.position)
        return obs, float(reward), bool(terminated), bool(truncated), {"portfolio_value": self.value, "market_close": self.close_prices[min(self.t, self.n_steps)]}
