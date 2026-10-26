# agents.py — PG, PPO, DQL, DDPG implementations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from networks import (
    LSTMPolicy, LSTMActorCritic,
    QNetwork, DDPGActor, DDPGCritic,
    OUNoise, ReplayBuffer
)
from config import PGConfig, PPOConfig, DQLConfig, DDPGConfig


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Policy Gradient (LSTM)
# ══════════════════════════════════════════════════════════════════════════════

class PGAgent:
    """
    REINFORCE with baseline and entropy bonus.
    LSTM policy for temporal dependencies.
    Monte Carlo returns normalized within each episode.
    """

    def __init__(self, cfg: PGConfig = PGConfig(), device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        self.policy = LSTMPolicy(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr,
                                          weight_decay=1e-4)

        # Episode buffer
        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float]         = []
        self.entropies: List[torch.Tensor]  = []

    def select_action(self, state: np.ndarray, reset_hidden: bool = False) -> int:
        state_t = torch.FloatTensor(state).to(self.device)
        logits  = self.policy(state_t, reset=reset_hidden)
        dist    = torch.distributions.Categorical(logits=logits)
        action  = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> Dict:
        """Update policy at episode end using Monte Carlo returns."""
        if len(self.rewards) == 0:
            return {}

        # Compute discounted returns
        G  = 0.0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.cfg.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns (variance reduction)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss + entropy bonus
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        pg_loss      = -(log_probs * returns).mean()
        entropy_loss = -self.cfg.entropy_coeff * entropies.mean()
        loss         = pg_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

        return {"pg_loss": pg_loss.item(), "entropy": entropies.mean().item()}

    def greedy_action(self, state: np.ndarray) -> int:
        """Deterministic action for evaluation."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            logits  = self.policy(state_t)
            return logits.argmax().item()

    def reset_hidden(self):
        self.policy.hidden = None


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Proximal Policy Optimization (LSTM)
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    PPO with LSTM Actor-Critic, GAE-λ advantage estimation,
    clipped surrogate objective.
    """

    def __init__(self, cfg: PPOConfig = PPOConfig(), device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        self.ac = LSTMActorCritic(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers
        ).to(device)

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr,
                                          weight_decay=1e-4)

        # Rollout buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.states:    List[np.ndarray]    = []
        self.actions:   List[int]           = []
        self.log_probs: List[float]         = []
        self.rewards:   List[float]         = []
        self.values:    List[float]         = []
        self.dones:     List[bool]          = []

    def select_action(self, state: np.ndarray, reset_hidden: bool = False) -> int:
        state_t = torch.FloatTensor(state).to(self.device)
        if reset_hidden:
            self.ac.hidden = None

        with torch.no_grad():
            logits, value = self.ac(state_t)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())
        return action.item()

    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self, last_value: float = 0.0) -> Dict:
        """Update using PPO clipped objective with GAE."""
        if len(self.rewards) < 2:
            self._reset_buffer()
            return {}

        # ── Compute GAE advantages ────────────────────────────────────────
        values  = self.values + [last_value]
        adv     = []
        gae     = 0.0

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] +
                     self.cfg.gamma * values[t+1] * (1 - self.dones[t]) -
                     values[t])
            gae   = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - self.dones[t]) * gae
            adv.insert(0, gae)

        adv   = torch.FloatTensor(adv).to(self.device)
        adv   = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret   = adv + torch.FloatTensor(self.values).to(self.device)

        states_t    = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t   = torch.LongTensor(self.actions).to(self.device)
        old_logp_t  = torch.FloatTensor(self.log_probs).to(self.device)

        total_loss = 0.0

        # ── PPO epochs ────────────────────────────────────────────────────
        for _ in range(self.cfg.n_epochs):
            self.ac.hidden = None
            logits, values_new = self.ac.forward_seq(states_t)

            dist       = torch.distributions.Categorical(logits=logits)
            new_logp   = dist.log_prob(actions_t)
            entropy    = dist.entropy().mean()

            ratio      = (new_logp - old_logp_t).exp()
            clip_ratio = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)

            actor_loss  = -torch.min(ratio * adv, clip_ratio * adv).mean()
            critic_loss = F.mse_loss(values_new.squeeze(), ret)
            loss        = (actor_loss +
                           self.cfg.vf_coeff * critic_loss -
                           self.cfg.entropy_coeff * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self._reset_buffer()
        return {"ppo_loss": total_loss / self.cfg.n_epochs}

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            logits, _ = self.ac(state_t)
            return logits.argmax().item()

    def reset_hidden(self):
        self.ac.hidden = None


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Deep Q-Learning (FFNN + Double DQL)
# ══════════════════════════════════════════════════════════════════════════════

class DQLAgent:
    """
    Double DQL with experience replay, target network, epsilon-greedy.
    FFNN (not LSTM) because replay buffer breaks temporal ordering.
    """

    def __init__(self, cfg: DQLConfig = DQLConfig(), device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        self.online = QNetwork(hidden_dim=cfg.hidden_dim,
                               num_layers=cfg.num_layers).to(device)
        self.target = QNetwork(hidden_dim=cfg.hidden_dim,
                               num_layers=cfg.num_layers).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr,
                                          weight_decay=1e-4)
        self.buffer    = ReplayBuffer(cfg.buffer_size)
        self.steps     = 0
        self.epsilon   = cfg.eps_start

    def _update_epsilon(self):
        self.epsilon = max(
            self.cfg.eps_end,
            self.cfg.eps_start - self.steps / self.cfg.eps_decay *
            (self.cfg.eps_start - self.cfg.eps_end)
        )

    def select_action(self, state: np.ndarray) -> int:
        self._update_epsilon()
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online(state_t)
            return q_vals.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1

    def update(self) -> Dict:
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        states, actions, rewards, next_states, dones, _ = \
            self.buffer.sample(self.cfg.batch_size, self.device)

        # ── Double DQL update ─────────────────────────────────────────────
        with torch.no_grad():
            # Select actions with online network
            next_actions = self.online(next_states).argmax(1, keepdim=True)
            # Evaluate with target network (reduces overestimation bias)
            next_q = self.target(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.cfg.gamma * next_q * (1 - dones)

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        # Periodic target update
        if self.steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {"dql_loss": loss.item(), "epsilon": self.epsilon}

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.online(state_t).argmax().item()


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Deep Deterministic Policy Gradient (FFNN)
# ══════════════════════════════════════════════════════════════════════════════

class DDPGAgent:
    """
    DDPG with continuous actor + critic, soft target updates,
    Ornstein-Uhlenbeck exploration noise.
    Output discretized to {SHORT, FLAT, LONG} via ±1/3 thresholds (paper).
    """

    def __init__(self, cfg: DDPGConfig = DDPGConfig(), device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        # Online networks
        self.actor  = DDPGActor(hidden_dim=cfg.hidden_dim,
                                num_layers=cfg.num_layers).to(device)
        self.critic = DDPGCritic(hidden_dim=cfg.hidden_dim,
                                 num_layers=cfg.num_layers).to(device)

        # Target networks (soft-update copies)
        self.actor_t  = DDPGActor(hidden_dim=cfg.hidden_dim,
                                  num_layers=cfg.num_layers).to(device)
        self.critic_t = DDPGCritic(hidden_dim=cfg.hidden_dim,
                                   num_layers=cfg.num_layers).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_t.eval(); self.critic_t.eval()

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),
                                           lr=cfg.lr_actor,  weight_decay=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=cfg.lr_critic, weight_decay=1e-4)

        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.noise  = OUNoise(theta=cfg.ou_theta, sigma=cfg.ou_sigma)
        self.steps  = 0

    def _soft_update(self, online: nn.Module, target: nn.Module):
        for p, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * p_t.data)

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cont_action = self.actor(state_t).item()   # ∈ [-1, 1]

        if explore:
            cont_action = np.clip(cont_action + self.noise(), -1.0, 1.0)

        # Discretize: ±1/3 equal-width thresholds (paper's Section 4.5)
        if cont_action < -1/3:
            discrete = 0   # SHORT
        elif cont_action > 1/3:
            discrete = 2   # LONG
        else:
            discrete = 1   # FLAT

        return discrete, cont_action

    def store(self, state, action, reward, next_state, done, cont_action):
        self.buffer.push(state, action, reward, next_state, done, cont_action)
        self.steps += 1

    def update(self) -> Dict:
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        states, actions, rewards, next_states, dones, cont_actions = \
            self.buffer.sample(self.cfg.batch_size, self.device)

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            next_actions_cont = self.actor_t(next_states)   # [B, 1]
            next_q = self.critic_t(next_states, next_actions_cont).squeeze()
            target_q = rewards + self.cfg.gamma * next_q * (1 - dones)

        current_q = self.critic(states, cont_actions.unsqueeze(1)).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────
        predicted_actions = self.actor(states)              # [B, 1]
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Soft target updates ───────────────────────────────────────────
        self._soft_update(self.actor,  self.actor_t)
        self._soft_update(self.critic, self.critic_t)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }

    def greedy_action(self, state: np.ndarray) -> int:
        discrete, _ = self.select_action(state, explore=False)
        return discrete

    def reset_noise(self):
        self.noise.reset()
