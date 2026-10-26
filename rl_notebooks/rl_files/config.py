# config.py — All hyperparameters and settings
from dataclasses import dataclass, field
from typing import List

# ── Assets ─────────────────────────────────────────────────────────────────
ASSETS = ["TSLA", "BTC-USD"]

# ── Data splits (matching the paper) ──────────────────────────────────────
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2024-07-31"
TEST_START  = "2024-08-01"

# ── Environment ────────────────────────────────────────────────────────────
TRANSACTION_COST  = 0.002     # 0.2% per trade (training only)
INITIAL_VALUE     = 100_000   # $100,000 starting capital
MIN_DAYS_LEFT     = 100       # minimum days remaining for random start
WARMUP_DAYS       = 60        # days to warm up rolling indicators

# ── State vector ───────────────────────────────────────────────────────────
STATE_DIM  = 15
N_ACTIONS  = 3   # SHORT=0, FLAT=1, LONG=2
ACTION_MAP = {0: -1, 1: 0, 2: 1}    # index → position

# ── Training ───────────────────────────────────────────────────────────────
MAX_EPISODES      = 2000
EVAL_EVERY        = 50        # eval on val set every N episodes
EARLY_STOP_PAT    = 10        # stop if no improvement for N evals
GAMMA             = 0.99      # discount factor
DEVICE            = "cuda"    # change to "cpu" if no GPU

# ── Sentiment model ────────────────────────────────────────────────────────
SENTIMENT_REPO    = "vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned"
SENTIMENT_BASE    = "Qwen/Qwen2.5-3B-Instruct"
MAX_ARTICLE_WORDS = 300       # truncate articles to first 300 words

# ── Default HPs per agent (overridden by Ray Tune) ─────────────────────────
@dataclass
class PGConfig:
    hidden_dim:    int   = 128
    num_layers:    int   = 2
    lr:            float = 1e-3
    gamma:         float = 0.99
    entropy_coeff: float = 0.01

@dataclass
class PPOConfig:
    hidden_dim:    int   = 128
    num_layers:    int   = 2
    lr:            float = 3e-4
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    clip_eps:      float = 0.2
    entropy_coeff: float = 0.01
    vf_coeff:      float = 0.5
    n_steps:       int   = 256
    n_epochs:      int   = 4
    batch_size:    int   = 64

@dataclass
class DQLConfig:
    hidden_dim:    int   = 256
    num_layers:    int   = 2
    lr:            float = 1e-3
    gamma:         float = 0.99
    buffer_size:   int   = 10_000
    batch_size:    int   = 64
    target_update: int   = 100   # update target network every N steps
    eps_start:     float = 1.0
    eps_end:       float = 0.05
    eps_decay:     int   = 5_000

@dataclass
class DDPGConfig:
    hidden_dim:    int   = 256
    num_layers:    int   = 2
    lr_actor:      float = 1e-4
    lr_critic:     float = 1e-3
    gamma:         float = 0.99
    tau:           float = 0.005  # soft update coefficient
    buffer_size:   int   = 10_000
    batch_size:    int   = 64
    ou_theta:      float = 0.15   # OU noise theta
    ou_sigma:      float = 0.2    # OU noise sigma
