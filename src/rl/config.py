"""
config.py — Part B (RL trading agents)
All constants for the trading environment + agent training live here.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Universe ──────────────────────────────────────────────────
# The 10 companies with the most entity mentions in SEntFiN-v1.1 itself (aliases like
# "RIL"/"Reliance Industries"/"Reliance" merged into one count) — this ties Part B's
# traded universe directly to what the Part A sentiment model actually trained on,
# instead of an arbitrary NIFTY-50 pick. (SEntFiN also mentions plenty of now-defunct/
# merged names — Kingfisher Airlines, Cairn India, Jet Airways, Sesa Sterlite, etc. —
# excluded here since they're not tradeable anymore.)
TICKERS = [
    "INFY.NS",        # Infosys        — 131 mentions
    "RELIANCE.NS",    # Reliance       —  92 (RIL + Reliance Industries + Reliance)
    "TCS.NS",         # TCS            —  74
    "SBIN.NS",        # SBI            —  54
    "MARUTI.NS",      # Maruti         —  52 (Maruti + Maruti Suzuki)
    "WIPRO.NS",       # Wipro          —  46
    "TMPV.NS",        # Tata Motors    —  38 (demerged 2026; TMPV.NS kept the pre-split price history,
                      # the new TMCV.NS commercial-vehicles spinoff only has ~175 days of trading so far)
    "LT.NS",          # L&T            —  37
    "ITC.NS",         # ITC            —  36
    "SUNPHARMA.NS",   # Sun Pharma     —  36
]

# ── Price history ─────────────────────────────────────────────
PRICE_YEARS_BACK = 5          # how far back to pull daily OHLCV
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# remaining ~0.15 is TEST

# ── Technical indicators (pandas_ta) ──────────────────────────
# Same core set used in the FinRL / ICAIF 2020 paper -- this is the "v1" design (the one with
# real, dashboard-backing results), reverted back after the CLaC@FinMMEval-2026 "alpha-reward"
# experiment (v2) underperformed and was dropped.
INDICATORS = ["macd", "rsi_14", "cci_20", "adx_14"]

# ── Sentiment ─────────────────────────────────────────────────
# yfinance's Ticker.news only returns a recent window (days, not years), so real
# dated sentiment can only cover that recent slice of the full price history.
# Historical days outside that window get a neutral placeholder (0.0) — the agent
# mostly learns portfolio dynamics from price/indicators, and the sentiment
# channel only carries real signal in the most recent stretch (which is also
# where the with-vs-without-sentiment comparison is actually meaningful).
MAX_NEWS_PER_TICKER = 40
SENTIMENT_NEUTRAL_FILL = 0.0

FINBERT_BASE_MODEL = "ProsusAI/finbert"
FINBERT_ADAPTER_DIR = os.path.join(
    ROOT, "saved_models", "finbert_peft_all_artifacts_v2", "saved_models", "finbert_lora_adapter"
)
FINBERT_TOKENIZER_DIR = os.path.join(
    ROOT, "saved_models", "finbert_peft_all_artifacts_v2", "saved_models", "finbert_tokenizer"
)

# ── Environment ───────────────────────────────────────────────
INITIAL_CASH = 1_000_000.0
TRANSACTION_COST_PCT = 0.001   # 10 bps per trade, each side
HMAX = 100                     # max shares tradable per asset per step
REWARD_SCALING = 1e-4

# ── Training ──────────────────────────────────────────────────
SEED = 42
AGENT_NAMES = ["PPO", "A2C", "DDPG", "SAC", "TD3"]
ON_POLICY_AGENTS = {"PPO", "A2C"}

# Off-policy agents (DDPG/SAC/TD3) are far more sample-hungry than on-policy ones and were
# consistently unprofitable at 50k steps in testing — giving them a bigger budget by default
# (they're cheap enough per-step to afford it: ~5-6 min per 50k steps measured locally).
# On-policy budget bumped too (50k->100k) since training only the sentiment variant (the
# project's actual goal) frees up roughly half the total compute budget either way.
ON_POLICY_TIMESTEPS = 100_000
OFF_POLICY_TIMESTEPS = 200_000
TOTAL_TIMESTEPS = ON_POLICY_TIMESTEPS   # kept for backwards compatibility / single-value override

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RL_PANEL_PATH = os.path.join(PROCESSED_DIR, "rl_panel.csv")
RL_SENTIMENT_PATH = os.path.join(PROCESSED_DIR, "rl_news_sentiment.csv")

RESULTS_DIR = os.path.join(ROOT, "results", "rl")
MODELS_DIR = os.path.join(ROOT, "saved_models", "rl")
for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

MLFLOW_EXPERIMENT = "DRLTrading"
