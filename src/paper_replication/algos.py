"""
Custom implementations of the two algorithms not natively available in stable-baselines3:

- PG (vanilla policy gradient / REINFORCE): Section 4.5 -- LSTM policy, Monte-Carlo returns
  normalized within each episode, entropy bonus to discourage premature collapse.
- DQL (Deep Q-Learning with the double-DQN update): Section 4.5 -- FFNN (paper's own choice,
  since the replay buffer breaks temporal ordering, making LSTM inappropriate), decoupled
  action selection (online net) / evaluation (target net) to reduce overestimation bias.

PPO (LSTM actor-critic via sb3-contrib RecurrentPPO) and DDPG (FFNN actor-critic via
stable-baselines3, continuous output discretized to {-1,0,1} inside the environment) reuse
existing, well-tested library implementations instead -- see train_paper.py.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── PG (REINFORCE, LSTM) ─────────────────────────────

class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs_seq, hidden=None):
        out, hidden = self.lstm(obs_seq, hidden)
        logits = self.head(out)
        return logits, hidden


def train_pg(env, total_episodes: int, hidden_dim=64, num_layers=1, lr=1e-3, gamma=0.99,
             entropy_coef=0.01, device="cpu", eval_every=50, eval_env=None, patience=10, seed=42):
    torch.manual_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = LSTMPolicy(obs_dim, n_actions, hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)

    best_sr, best_state, no_improve = -np.inf, None, 0

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset()
        hidden = None
        log_probs, entropies, rewards = [], [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            logits, hidden = policy(obs_t, hidden)
            dist = torch.distributions.Categorical(logits=logits.view(-1))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(reward)

        returns = np.zeros(len(rewards))
        g = 0.0
        for i in reversed(range(len(rewards))):
            g = rewards[i] + gamma * g
            returns[i] = g
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        policy_loss = -(torch.stack(log_probs) * returns_t).sum()
        entropy_loss = -entropy_coef * torch.stack(entropies).sum()
        loss = policy_loss + entropy_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if eval_env is not None and ep % eval_every == 0:
            sr = _eval_discrete_policy(policy, eval_env, device)
            if sr > best_sr:
                best_sr = sr
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        policy.load_state_dict(best_state)
    return policy, best_sr


def _eval_discrete_policy(policy, env, device):
    obs, _ = env.reset()
    hidden = None
    values, done = [], False
    with torch.no_grad():
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            logits, hidden = policy(obs_t, hidden)
            action = int(torch.argmax(logits.view(-1)).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            values.append(info["portfolio_value"])
    values = np.array(values)
    if len(values) < 2:
        return 0.0
    returns = np.diff(values) / values[:-1]
    return float((returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252))


def predict_pg(policy, obs, hidden, device="cpu"):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
        logits, hidden = policy(obs_t, hidden)
        action = int(torch.argmax(logits.view(-1)).item())
    return action, hidden


# ───────────────────────────── DQL (double DQN, FFNN) ─────────────────────────────

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (np.array(obs, dtype=np.float32), np.array(act, dtype=np.int64),
                np.array(rew, dtype=np.float32), np.array(next_obs, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def train_dql(env, total_episodes: int, hidden_dim=64, num_layers=1, lr=1e-3, gamma=0.99,
              batch_size=64, buffer_size=50_000, target_update_every=500, eps_start=1.0,
              eps_end=0.05, eps_decay_episodes=None, device="cpu", eval_every=50, eval_env=None,
              patience=10, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    eps_decay_episodes = eps_decay_episodes or max(1, total_episodes // 2)

    online = QNet(obs_dim, n_actions, hidden_dim, num_layers).to(device)
    target = QNet(obs_dim, n_actions, hidden_dim, num_layers).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr, weight_decay=1e-4)
    buffer = ReplayBuffer(buffer_size)

    best_sr, best_state, no_improve, step_count = -np.inf, None, 0, 0

    for ep in range(1, total_episodes + 1):
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - ep / eps_decay_episodes)
        obs, _ = env.reset()
        done = False
        while not done:
            if np.random.rand() < eps:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    q = online(torch.as_tensor(obs, dtype=torch.float32, device=device))
                    action = int(torch.argmax(q).item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            step_count += 1

            if len(buffer) >= batch_size:
                b_obs, b_act, b_rew, b_next, b_done = buffer.sample(batch_size)
                b_obs = torch.as_tensor(b_obs, device=device)
                b_act = torch.as_tensor(b_act, device=device)
                b_rew = torch.as_tensor(b_rew, device=device)
                b_next = torch.as_tensor(b_next, device=device)
                b_done = torch.as_tensor(b_done, device=device)

                q_values = online(b_obs).gather(1, b_act.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    next_actions = torch.argmax(online(b_next), dim=1)  # double DQN: select via online net
                    next_q = target(b_next).gather(1, next_actions.view(-1, 1)).squeeze(1)  # evaluate via target net
                    target_q = b_rew + gamma * next_q * (1 - b_done)
                loss = F.smooth_l1_loss(q_values, target_q)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                opt.step()

                if step_count % target_update_every == 0:
                    target.load_state_dict(online.state_dict())

        if eval_env is not None and ep % eval_every == 0:
            sr = _eval_dqn(online, eval_env, device)
            if sr > best_sr:
                best_sr = sr
                best_state = {k: v.clone() for k, v in online.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        online.load_state_dict(best_state)
    return online, best_sr


def _eval_dqn(qnet, env, device):
    obs, _ = env.reset()
    values, done = [], False
    with torch.no_grad():
        while not done:
            q = qnet(torch.as_tensor(obs, dtype=torch.float32, device=device))
            action = int(torch.argmax(q).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            values.append(info["portfolio_value"])
    values = np.array(values)
    if len(values) < 2:
        return 0.0
    returns = np.diff(values) / values[:-1]
    return float((returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252))


def predict_dql(qnet, obs, device="cpu"):
    with torch.no_grad():
        q = qnet(torch.as_tensor(obs, dtype=torch.float32, device=device))
        return int(torch.argmax(q).item())
