"""
config.py
All project-wide constants live here.
Change one value here → affects the entire project.
"""

# ── Labels ────────────────────────────────────────────────────
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL  = {0: "positive", 1: "neutral", 2: "negative"}
NUM_LABELS = 3

# ── Tokenizer ─────────────────────────────────────────────────
VOCAB_SIZE  = 30_000   # top N most frequent words kept
MAX_LENGTH  = 128      # pad/truncate all headlines to this
PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
PAD_IDX     = 0
UNK_IDX     = 1

# ── Training ──────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 64
NUM_EPOCHS   = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.01
PATIENCE      = 3       # early stopping patience

# ── Data split ────────────────────────────────────────────────
TEST_SIZE  = 0.15
VAL_SIZE   = 0.10      # of the remaining train set

# ── Paths ─────────────────────────────────────────────────────
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR      = os.path.join(ROOT, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR   = os.path.join(ROOT, "results")
MODELS_DIR    = os.path.join(ROOT, "saved_models")

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── MLflow ────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "IndianFinSentiment"
MLFLOW_MODEL_NAME = "IndianFinSentiment"

# ── HuggingFace model names ───────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
