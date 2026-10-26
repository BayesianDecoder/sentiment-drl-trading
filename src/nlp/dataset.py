"""
dataset.py
──────────
Handles everything data-related for the NLP benchmark:
  1. Load SEntFiN-v1.1 (entity-level Indian financial headline sentiment)
  2. Resolve multi-entity headlines to a single headline-level label
  3. Clean, dedupe
  4. Train / Val / Test split
  5. Build custom word-level tokenizer (word2index)
  6. PyTorch Dataset + DataLoader

Run this file directly to see EDA output:
  python src/nlp/dataset.py
"""

import os
import re
import ast
import json
import pickle
import random
import logging
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    LABEL2ID, ID2LABEL, VOCAB_SIZE, MAX_LENGTH,
    PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX,
    SEED, TEST_SIZE, VAL_SIZE, PROCESSED_DIR, RAW_DIR
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════

def _resolve_sentifin_path() -> str:
    """
    Find SEntFiN-v1.1.csv. Checks Kaggle input mounts first (for standalone
    Kaggle notebooks), then falls back to the local repo's data/ directories.
    """
    candidates = [
        os.path.join(RAW_DIR, "SEntFiN-v1.1.csv"),
        os.path.join(RAW_DIR, "..", "SEntFiN-v1.1.csv"),
    ]
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for root, _, files in os.walk(kaggle_input):
            for f in files:
                if f == "SEntFiN-v1.1.csv":
                    candidates.insert(0, os.path.join(root, f))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "SEntFiN-v1.1.csv not found. Place it at data/raw/SEntFiN-v1.1.csv "
        "(local) or attach it as a Kaggle dataset input (Kaggle)."
    )


def load_sentifin(path: Optional[str] = None) -> pd.DataFrame:
    """
    SEntFiN-v1.1 — 10,753 Indian financial news headlines with entity-level
    sentiment, e.g. Decisions = {"SpiceJet": "neutral"}.

    ~26% of headlines mention >1 entity. Since every downstream model here
    classifies a whole headline as one label (not entity-aware), we:
      - keep headlines where all entities share one sentiment (trivially
        true for single-entity headlines) and use that as the headline label
      - drop headlines where entities genuinely disagree (~11.5% of rows) —
        those are ambiguous for a single-label headline classifier
    """
    log.info("Loading SEntFiN-v1.1...")
    csv_path = path or _resolve_sentifin_path()
    raw = pd.read_csv(csv_path)

    texts, labels, n_entities = [], [], []
    dropped_conflicting = 0
    for _, row in raw.iterrows():
        try:
            decisions = ast.literal_eval(row["Decisions"])
        except (ValueError, SyntaxError):
            continue
        if not decisions:
            continue
        unique_labels = set(decisions.values())
        if len(unique_labels) != 1:
            dropped_conflicting += 1
            continue
        texts.append(row["Title"])
        labels.append(unique_labels.pop())
        n_entities.append(len(decisions))

    df = pd.DataFrame({"text": texts, "label": labels, "n_entities": n_entities})
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL2ID.keys())]
    df["label_id"] = df["label"].map(LABEL2ID)
    df["source"] = "sentifin"

    log.info(f"  Loaded {len(df)} headlines "
             f"(dropped {dropped_conflicting} with conflicting entity sentiment)")
    return df[["text", "label", "label_id", "source", "n_entities"]]


def load_and_combine() -> pd.DataFrame:
    """
    Master function: load SEntFiN, clean, deduplicate.
    Returns a single clean DataFrame ready for splitting.
    """
    df = load_sentifin()

    # ── clean ────────────────────────────────────────────────
    df = df.dropna(subset=["text", "label_id"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]        # remove very short
    df = df[df["text"].str.len() < 1000]      # remove very long
    df = df.drop_duplicates(subset=["text"])  # remove exact dupes

    log.info(f"\n{'='*50}")
    log.info(f"Combined dataset: {len(df)} samples")
    log.info(f"Label distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        log.info(f"  {label:>10}: {count:>6} ({pct:.1f}%)")
    log.info(f"{'='*50}\n")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split to maintain class balance across all splits.
    Returns: train_df, val_df, test_df
    """
    # first split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label_id"],
        random_state=SEED
    )

    # then split train into train + val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df["label_id"],
        random_state=SEED
    )

    log.info(f"Split sizes → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 3. TOKENIZER
# ══════════════════════════════════════════════════════════════

class FinancialTokenizer:
    """
    Simple word-level tokenizer built from scratch.
    Why not use HuggingFace tokenizer?
    → Models 1-6 use this custom tokenizer so you understand
      every step. FinBERT (Model 7) uses its own BERT tokenizer.

    Usage:
        tokenizer = FinancialTokenizer()
        tokenizer.build_vocab(train_texts)
        ids = tokenizer.encode("Reliance posts record profit")
        # → [45, 203, 891, 234, 0, 0, ...]  (padded to MAX_LENGTH)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 max_length: int = MAX_LENGTH):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX,
                                          UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN,
                                          UNK_IDX: UNK_TOKEN}
        self.vocab_built = False

    def _clean(self, text: str) -> str:
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)       # remove URLs
        text = re.sub(r'[^a-z0-9\s\-\%\.]', ' ', text) # keep alphanum
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Split cleaned text into word tokens."""
        return self._clean(text).split()

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build word2idx from training texts.
        Only call on TRAIN set — never val/test (avoid data leakage).
        """
        log.info("Building vocabulary from training texts...")
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # take top (vocab_size - 2) words (leaving room for PAD, UNK)
        most_common = counter.most_common(self.vocab_size - 2)

        for idx, (word, freq) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

        self.vocab_built = True
        actual_vocab = len(self.word2idx)
        log.info(f"  ✅ Vocab built: {actual_vocab} tokens "
                 f"(top word: '{most_common[0][0]}' freq={most_common[0][1]})")

    def encode(self, text: str) -> List[int]:
        """
        Convert text → fixed-length list of token IDs.
        - Unknown words → UNK_IDX
        - Shorter than max_length → pad with PAD_IDX
        - Longer than max_length → truncate
        """
        if not self.vocab_built:
            raise RuntimeError("Call build_vocab() first!")

        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]

        # truncate
        ids = ids[:self.max_length]

        # pad
        ids = ids + [PAD_IDX] * (self.max_length - len(ids))

        return ids

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts. Returns shape (N, max_length)."""
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        log.info(f"Tokenizer saved to {path}")

    @staticmethod
    def load(path: str) -> 'FinancialTokenizer':
        with open(path, 'rb') as f:
            tok = pickle.load(f)
        log.info(f"Tokenizer loaded from {path} "
                 f"(vocab size: {len(tok.word2idx)})")
        return tok

    @property
    def vocab_size_actual(self) -> int:
        return len(self.word2idx)


# ══════════════════════════════════════════════════════════════
# 4. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for the 5 custom deep learning models
    (TextCNN, BiLSTM, CNN-BiLSTM-Attn, Transformer from scratch).

    NOT used for FinBERT — that uses HuggingFace's tokenizer directly.

    Usage:
        train_ds = SentimentDataset(train_df, tokenizer)
        loader   = DataLoader(train_ds, batch_size=64, shuffle=True)
        for batch in loader:
            input_ids = batch['input_ids']  # (batch, 128)
            labels    = batch['labels']     # (batch,)
    """

    def __init__(self, df: pd.DataFrame, tokenizer: FinancialTokenizer):
        self.texts     = df["text"].tolist()
        self.labels    = df["label_id"].tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer.encode(self.texts[idx])
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels":    torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    tokenizer: FinancialTokenizer,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for PyTorch models."""

    train_ds = SentimentDataset(train_df, tokenizer)
    val_ds   = SentimentDataset(val_df,   tokenizer)
    test_ds  = SentimentDataset(test_df,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════
# 5. EDA  (runs when you execute this file directly)
# ══════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, tokenizer: FinancialTokenizer) -> None:
    """Print EDA summary and save plots."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')   # no display needed

    print("\n" + "="*60)
    print("EDA SUMMARY")
    print("="*60)

    # ── 1. Dataset size ──────────────────────────────────────
    print(f"\nTotal samples : {len(df)}")
    print(f"Columns       : {list(df.columns)}")

    # ── 2. Label distribution ────────────────────────────────
    print("\nLabel distribution:")
    for lbl, count in df["label"].value_counts().items():
        bar = "█" * int(count / len(df) * 50)
        print(f"  {lbl:>10}: {count:>6} ({count/len(df)*100:5.1f}%)  {bar}")

    # ── 3. Source distribution ───────────────────────────────
    print("\nSource distribution:")
    for src, count in df["source"].value_counts().items():
        print(f"  {src:>20}: {count}")

    # ── 4. Text length statistics ────────────────────────────
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()
    print(f"\nHeadline word count stats:")
    print(df["word_count"].describe().round(1).to_string())

    # ── 5. Sample headlines per class ───────────────────────
    print("\nSample headlines per class:")
    for label in ["positive", "neutral", "negative"]:
        samples = df[df["label"] == label]["text"].sample(
            min(3, sum(df["label"] == label)),
            random_state=SEED
        ).tolist()
        print(f"\n  [{label.upper()}]")
        for s in samples:
            print(f"    → {s[:100]}")

    # ── 6. Token coverage analysis ───────────────────────────
    unk_counts = []
    for text in df["text"].head(500):
        tokens = tokenizer._tokenize(text)
        unks = sum(1 for t in tokens if t not in tokenizer.word2idx)
        unk_counts.append(unks / max(len(tokens), 1))
    print(f"\nVocab coverage (500 samples):")
    print(f"  Avg OOV rate: {np.mean(unk_counts)*100:.1f}%")

    # ── 7. Save plots ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Dataset EDA — Indian Financial Sentiment", fontsize=13)

    # label distribution pie
    counts = df["label"].value_counts()
    axes[0].pie(counts.values, labels=counts.index,
                autopct="%1.1f%%", colors=["#4ade80","#fbbf24","#f87171"])
    axes[0].set_title("Label Distribution")

    # word count histogram
    axes[1].hist(df["word_count"], bins=30, color="#6366f1", edgecolor="white")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Headline Length Distribution")
    axes[1].axvline(df["word_count"].mean(), color="red",
                    linestyle="--", label=f"Mean: {df['word_count'].mean():.1f}")
    axes[1].legend()

    # source distribution bar
    source_counts = df["source"].value_counts()
    axes[2].bar(source_counts.index, source_counts.values,
                color=["#f59e0b","#3b82f6","#10b981"])
    axes[2].set_title("Samples per Source")
    axes[2].set_ylabel("Count")

    plt.tight_layout()

    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.config import RESULTS_DIR
    save_path = os.path.join(RESULTS_DIR, "eda_plots.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\n✅ EDA plots saved to: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN — run as script for EDA
# ══════════════════════════════════════════════════════════════

def prepare_data():
    """
    Full data preparation pipeline.
    Call this from train scripts to get everything ready.

    Returns:
        train_df, val_df, test_df, tokenizer
    """
    # try to load cached data first
    cache_path = os.path.join(PROCESSED_DIR, "data_splits.pkl")
    tok_path   = os.path.join(PROCESSED_DIR, "tokenizer.pkl")

    if os.path.exists(cache_path) and os.path.exists(tok_path):
        log.info("Loading cached data splits...")
        with open(cache_path, 'rb') as f:
            train_df, val_df, test_df = pickle.load(f)
        tokenizer = FinancialTokenizer.load(tok_path)
        log.info(f"  ✅ Loaded from cache: train={len(train_df)} "
                 f"val={len(val_df)} test={len(test_df)}")
        return train_df, val_df, test_df, tokenizer

    # load fresh
    df = load_and_combine()
    train_df, val_df, test_df = split_data(df)

    # build tokenizer on train only
    tokenizer = FinancialTokenizer()
    tokenizer.build_vocab(train_df["text"].tolist())

    # cache everything
    with open(cache_path, 'wb') as f:
        pickle.dump((train_df, val_df, test_df), f)
    tokenizer.save(tok_path)

    log.info(f"✅ Data ready → Train: {len(train_df)} | "
             f"Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df, tokenizer


if __name__ == "__main__":
    # Run this file directly to see EDA:
    # python src/nlp/dataset.py
    log.info("Running EDA...")
    train_df, val_df, test_df, tokenizer = prepare_data()

    # combine for EDA stats
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    run_eda(all_df, tokenizer)

    # show a few encoded examples
    print("\nEncoding examples:")
    for text in [
        "Reliance Industries posts record quarterly profit",
        "RBI raises repo rate amid inflation concerns",
        "Adani Group shares fall on short-seller report"
    ]:
        ids = tokenizer.encode(text)
        tokens = tokenizer._tokenize(text)
        print(f"\n  Text   : {text}")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids[:len(tokens)]}  (then padded)")

    print("\n✅ Dataset setup complete!")
