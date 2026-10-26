# trainer.py — Training loop with Sharpe-based early stopping
import numpy as np
import torch
import copy
import time
from typing import Dict, Optional, Tuple
from pathlib import Path

from trading_env import TradingEnv
from agents import PGAgent, PPOAgent, DQLAgent, DDPGAgent
from config import (
    MAX_EPISODES, EVAL_EVERY, EARLY_STOP_PAT,
    PGConfig, PPOConfig, DQLConfig, DDPGConfig,
    INITIAL_VALUE, TRANSACTION_COST
)

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


def compute_sharpe(portfolio_values: np.ndarray) -> float:
    """Annualized Sharpe ratio from portfolio value series."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def evaluate_agent(agent, env: TradingEnv) -> Tuple[float, Dict]:
    """Run one episode on eval env (no randomness, no costs)."""
    obs, _ = env.reset()
    if hasattr(agent, 'reset_hidden'):
        agent.reset_hidden()
    if hasattr(agent, 'reset_noise'):
        agent.reset_noise()

    done = False
    while not done:
        action = agent.greedy_action(obs)
        obs, _, done, _, _ = env.step(action)

    metrics = env.get_metrics()
    sharpe  = compute_sharpe(np.array(env.portfolio_values))
    return sharpe, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Generic training functions per agent type
# ══════════════════════════════════════════════════════════════════════════════

def train_pg(agent: PGAgent,
             train_env: TradingEnv,
             val_env: TradingEnv,
             max_episodes: int = MAX_EPISODES,
             eval_every: int = EVAL_EVERY,
             patience: int = EARLY_STOP_PAT,
             agent_name: str = "PG") -> Tuple[PGAgent, Dict]:

    best_sharpe = -np.inf
    best_state  = None
    no_improve  = 0
    history     = {"train_rewards": [], "val_sharpe": [], "val_metrics": []}

    for ep in range(1, max_episodes + 1):
        obs, _ = train_env.reset()
        agent.reset_hidden()
        agent.select_action(obs, reset_hidden=True)   # prime hidden state
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _, _ = train_env.step(action)
            agent.store_reward(reward)
            ep_reward += reward

        losses = agent.update()
        history["train_rewards"].append(ep_reward)

        if ep % eval_every == 0:
            val_sharpe, val_metrics = evaluate_agent(agent, val_env)
            history["val_sharpe"].append(val_sharpe)
            history["val_metrics"].append(val_metrics)

            print(f"  [{agent_name}] ep={ep:4d} | "
                  f"train_rew={ep_reward:.4f} | "
                  f"val_SR={val_sharpe:.3f} | "
                  f"val_CR={val_metrics['cumulative_return']:.1f}%")

            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_state  = copy.deepcopy(agent.policy.state_dict())
                no_improve  = 0
                # Save checkpoint
                torch.save(best_state, CKPT_DIR / f"{agent_name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{agent_name}] Early stopping at episode {ep}")
                    break

    # Restore best checkpoint
    if best_state:
        agent.policy.load_state_dict(best_state)
    history["best_val_sharpe"] = best_sharpe
    return agent, history


def train_ppo(agent: PPOAgent,
              train_env: TradingEnv,
              val_env: TradingEnv,
              max_episodes: int = MAX_EPISODES,
              eval_every: int = EVAL_EVERY,
              patience: int = EARLY_STOP_PAT,
              agent_name: str = "PPO") -> Tuple[PPOAgent, Dict]:

    best_sharpe = -np.inf
    best_state  = None
    no_improve  = 0
    history     = {"train_rewards": [], "val_sharpe": [], "val_metrics": []}

    for ep in range(1, max_episodes + 1):
        obs, _ = train_env.reset()
        agent.reset_hidden()
        done = False
        ep_reward = 0.0
        last_value = 0.0

        while not done:
            action = agent.select_action(obs, reset_hidden=False)
            obs, reward, done, _, info = train_env.step(action)
            agent.store_reward(reward, done)
            ep_reward += reward

        # Get value of last state for GAE
        if not done:
            state_t = torch.FloatTensor(obs).to(agent.device)
            with torch.no_grad():
                _, last_value = agent.ac(state_t)
                last_value = last_value.item()

        losses = agent.update(last_value=last_value)
        history["train_rewards"].append(ep_reward)

        if ep % eval_every == 0:
            val_sharpe, val_metrics = evaluate_agent(agent, val_env)
            history["val_sharpe"].append(val_sharpe)
            history["val_metrics"].append(val_metrics)

            print(f"  [{agent_name}] ep={ep:4d} | "
                  f"train_rew={ep_reward:.4f} | "
                  f"val_SR={val_sharpe:.3f} | "
                  f"val_CR={val_metrics['cumulative_return']:.1f}%")

            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_state  = copy.deepcopy(agent.ac.state_dict())
                no_improve  = 0
                torch.save(best_state, CKPT_DIR / f"{agent_name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{agent_name}] Early stopping at episode {ep}")
                    break

    if best_state:
        agent.ac.load_state_dict(best_state)
    history["best_val_sharpe"] = best_sharpe
    return agent, history


def train_dql(agent: DQLAgent,
              train_env: TradingEnv,
              val_env: TradingEnv,
              max_episodes: int = MAX_EPISODES,
              eval_every: int = EVAL_EVERY,
              patience: int = EARLY_STOP_PAT,
              agent_name: str = "DQL") -> Tuple[DQLAgent, Dict]:

    best_sharpe = -np.inf
    best_state  = None
    no_improve  = 0
    history     = {"train_rewards": [], "val_sharpe": [], "val_metrics": []}

    for ep in range(1, max_episodes + 1):
        obs, _ = train_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action  = agent.select_action(obs)
            next_obs, reward, done, _, _ = train_env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            losses = agent.update()
            obs = next_obs
            ep_reward += reward

        history["train_rewards"].append(ep_reward)

        if ep % eval_every == 0:
            val_sharpe, val_metrics = evaluate_agent(agent, val_env)
            history["val_sharpe"].append(val_sharpe)
            history["val_metrics"].append(val_metrics)

            print(f"  [{agent_name}] ep={ep:4d} | "
                  f"train_rew={ep_reward:.4f} | eps={agent.epsilon:.3f} | "
                  f"val_SR={val_sharpe:.3f} | "
                  f"val_CR={val_metrics['cumulative_return']:.1f}%")

            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_state  = copy.deepcopy(agent.online.state_dict())
                no_improve  = 0
                torch.save(best_state, CKPT_DIR / f"{agent_name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{agent_name}] Early stopping at episode {ep}")
                    break

    if best_state:
        agent.online.load_state_dict(best_state)
    history["best_val_sharpe"] = best_sharpe
    return agent, history


def train_ddpg(agent: DDPGAgent,
               train_env: TradingEnv,
               val_env: TradingEnv,
               max_episodes: int = MAX_EPISODES,
               eval_every: int = EVAL_EVERY,
               patience: int = EARLY_STOP_PAT,
               agent_name: str = "DDPG") -> Tuple[DDPGAgent, Dict]:

    best_sharpe = -np.inf
    best_actor  = None
    best_critic = None
    no_improve  = 0
    history     = {"train_rewards": [], "val_sharpe": [], "val_metrics": []}

    for ep in range(1, max_episodes + 1):
        obs, _ = train_env.reset()
        agent.reset_noise()
        done = False
        ep_reward = 0.0

        while not done:
            action, cont_action = agent.select_action(obs, explore=True)
            next_obs, reward, done, _, _ = train_env.step(action)
            agent.store(obs, action, reward, next_obs, done, cont_action)
            losses = agent.update()
            obs = next_obs
            ep_reward += reward

        history["train_rewards"].append(ep_reward)

        if ep % eval_every == 0:
            val_sharpe, val_metrics = evaluate_agent(agent, val_env)
            history["val_sharpe"].append(val_sharpe)
            history["val_metrics"].append(val_metrics)

            print(f"  [{agent_name}] ep={ep:4d} | "
                  f"train_rew={ep_reward:.4f} | "
                  f"val_SR={val_sharpe:.3f} | "
                  f"val_CR={val_metrics['cumulative_return']:.1f}%")

            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_actor  = copy.deepcopy(agent.actor.state_dict())
                best_critic = copy.deepcopy(agent.critic.state_dict())
                no_improve  = 0
                torch.save(best_actor,  CKPT_DIR / f"{agent_name}_actor_best.pt")
                torch.save(best_critic, CKPT_DIR / f"{agent_name}_critic_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{agent_name}] Early stopping at episode {ep}")
                    break

    if best_actor:
        agent.actor.load_state_dict(best_actor)
        agent.critic.load_state_dict(best_critic)
    history["best_val_sharpe"] = best_sharpe
    return agent, history


# ── Dispatcher ────────────────────────────────────────────────────────────

TRAIN_FN = {
    "PG":   train_pg,
    "PPO":  train_ppo,
    "DQL":  train_dql,
    "DDPG": train_ddpg,
}

def train_agent(name: str, agent, train_env, val_env, **kwargs):
    fn = TRAIN_FN[name]
    return fn(agent, train_env, val_env, agent_name=name, **kwargs)
