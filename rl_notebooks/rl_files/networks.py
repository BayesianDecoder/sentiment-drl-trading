# networks.py — Neural network architectures for all 4 DRL agents
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── LSTM networks (for PG and PPO — temporal dependencies) ────────────────

class LSTMPolicy(nn.Module):
    """LSTM policy network for Policy Gradient."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128,
                 num_layers: int = 2, n_actions: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc   = nn.Linear(hidden_dim, n_actions)
        self.hidden: Optional[Tuple] = None

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
        )

    def forward(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        dev = x.device
        if reset or self.hidden is None:
            self.init_hidden(1, str(dev))

        if x.dim() == 1:   # single step: [15]
            x = x.unsqueeze(0).unsqueeze(0)     # [1, 1, 15]
        elif x.dim() == 2:  # batch of steps: [T, 15]
            x = x.unsqueeze(0)                   # [1, T, 15]

        out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = tuple(h.detach() for h in self.hidden)
        logits = self.fc(out[:, -1, :])          # [1, n_actions]
        return logits.squeeze(0)                  # [n_actions]


class LSTMActorCritic(nn.Module):
    """LSTM Actor-Critic for PPO."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128,
                 num_layers: int = 2, n_actions: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm   = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.actor  = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.hidden: Optional[Tuple] = None

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
        )

    def forward(self, x: torch.Tensor,
                reset: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = x.device
        if reset or self.hidden is None:
            self.init_hidden(1, str(dev))

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = tuple(h.detach() for h in self.hidden)
        h = out[:, -1, :]                   # [1, hidden_dim]

        logits = self.actor(h).squeeze(0)   # [n_actions]
        value  = self.critic(h).squeeze()   # scalar

        return logits, value

    def forward_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward over a full sequence for PPO batch updates.
        x: [T, 15]
        """
        x = x.unsqueeze(0)          # [1, T, 15]
        self.init_hidden(1, str(x.device))
        out, self.hidden = self.lstm(x, self.hidden)
        h = out.squeeze(0)           # [T, hidden_dim]
        return self.actor(h), self.critic(h)  # [T, n_actions], [T, 1]


# ── FFNN networks (for DQL and DDPG — replay buffer breaks temporal order) ─

def make_ffnn(input_dim: int, hidden_dim: int, output_dim: int,
              num_layers: int = 2) -> nn.Sequential:
    layers: list = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Double DQL Q-network: Q(s, a) for all actions simultaneously."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 256,
                 n_actions: int = 3, num_layers: int = 2):
        super().__init__()
        self.net = make_ffnn(input_dim, hidden_dim, n_actions, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [batch, n_actions]


class DDPGActor(nn.Module):
    """DDPG Actor: μ(s) → continuous action ∈ [-1, 1]."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        layers: list = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1), nn.Tanh()]   # output ∈ [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [batch, 1]

    def get_action(self, x: torch.Tensor) -> int:
        """Convert continuous output to discrete action {SHORT, FLAT, LONG}."""
        with torch.no_grad():
            val = self.net(x).item()
        # Discretize: threshold ±1/3 gives equal-width intervals (paper)
        if val < -1/3:
            return 0   # SHORT
        elif val > 1/3:
            return 2   # LONG
        else:
            return 1   # FLAT


class DDPGCritic(nn.Module):
    """DDPG Critic: Q(s, a) → scalar."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 256,
                 num_layers: int = 2):
        super().__init__()
        # Concatenate state (15) + action (1) = 16 input features
        self.net = make_ffnn(input_dim + 1, hidden_dim, 1, num_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)   # [batch, 1]


# ── Ornstein-Uhlenbeck noise for DDPG exploration ─────────────────────────

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2):
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self.reset()

    def reset(self):
        self.state = self.mu

    def __call__(self) -> float:
        dx = (self.theta * (self.mu - self.state) * self.dt +
              self.sigma * np.sqrt(self.dt) * np.random.randn())
        self.state += dx
        return self.state


import numpy as np  # needed for OUNoise


# ── Replay buffer (shared by DQL and DDPG) ────────────────────────────────

class ReplayBuffer:
    """Circular replay buffer for off-policy RL."""

    def __init__(self, capacity: int = 10_000, state_dim: int = 15):
        self.capacity  = capacity
        self.state_dim = state_dim
        self.ptr       = 0
        self.size      = 0

        self.states  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_st = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)
        # For DDPG: continuous actions
        self.cont_actions = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done,
             cont_action: float = 0.0):
        self.states[self.ptr]       = state
        self.actions[self.ptr]      = action
        self.rewards[self.ptr]      = reward
        self.next_st[self.ptr]      = next_state
        self.dones[self.ptr]        = float(done)
        self.cont_actions[self.ptr] = cont_action
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu"):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.LongTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_st[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
            torch.FloatTensor(self.cont_actions[idx]).to(device),
        )

    def __len__(self):
        return self.size
