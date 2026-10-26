"""
Builds notebooks/13_llm_lora_qlora_sentiment.ipynb — a Kaggle-ready notebook (needs a GPU;
NOT feasible on this dev machine, an 8GB-RAM Mac) that answers: can a modern open LLM,
zero-shot or fine-tuned with LoRA/QLoRA, beat FinBERT+LoRA's 86.5% F1 on the same SEntFiN task?

Run: .venv/bin/python scripts/notebook_builders/build_llm_lora_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# LLM Benchmarking + Fine-Tuning for Financial Sentiment (SEntFiN-v1.1)

**Question this notebook answers**: FinBERT+LoRA hit 86.5% F1 on SEntFiN, but has a confirmed,
confidently-wrong blind spot on "corporate action" headlines (partnerships, new labs,
contract wins) that don't contain classic profit/loss vocabulary. Which open LLM -- across
several different model families, not just one -- actually does best here, zero-shot and
fine-tuned, and does any of them fix that blind spot?

**Two-stage design, so compute goes where it's actually earned**:

**Stage 1 -- broad zero-shot benchmark (cheap, no training)**: 8 candidates spanning ~0.5B-3.8B
params across 5 different organizations/families -- Qwen2.5 (0.5B/1.5B/3B-Instruct),
Gemma-2-2B-it, Phi-3.5-mini-instruct, SmolLM2-1.7B-Instruct, and Llama-3.2-1B/3B-Instruct
(gated on HF; skipped gracefully if you haven't accepted the license / set `HF_TOKEN`).
Deliberately excludes anything larger (e.g. Qwen2.5-7B): a real run measured 3.68h *per epoch*
for a 7B model even with QLoRA, and FinBERT -- smaller than every candidate here at 110M params
-- already hit 86.5% F1 on this exact task, so there's no strong evidence this narrow a
classification problem needs a larger model at all. Each candidate gets scored zero-shot via
log-probability extraction (the CLaC@FinMMEval-2026 paper's method: one forward pass, softmax
restricted to the positive/negative/neutral token IDs -- no text generation, no fine-tuning at
all). This alone tells you whether broad pretraining already beats a narrow fine-tuned model on
FinBERT's failure genre, before spending anything on training.

**Stage 2 -- LoRA + QLoRA fine-tuning (expensive, so only the `TOP_K_FOR_FINETUNING` zero-shot
winners advance)**: fine-tuning all 8 candidates with a full search each would be a lot more
compute for a question this cheap screen already answers well enough to narrow down -- the
standard way to keep a multi-model benchmark's compute bounded without giving up real search
rigor where it counts. Every remaining candidate tops out at 3.8B (Phi-3.5), small enough that
BOTH methods run for each: LoRA (bf16 base weights + trainable low-rank adapters) and QLoRA
(frozen base quantized to 4-bit NF4 first, smaller footprint at some precision cost), both
applied via `AutoModelForSequenceClassification` on the decoder backbone (analogous to what
FinBERT+LoRA did on an encoder). Whichever models "win" are also all real Ollama models
(`ollama pull qwen2.5:1.5b`, `gemma2:2b`, `phi3.5`, etc.) for local serving afterward.

Each fine-tuning method gets its own Optuna hyperparameter search (rank/alpha/dropout/lr/
weight_decay/batch_size/target_modules -- see the dedicated section below), with the search
budget itself scaling down for larger models (see `get_search_budget`), before committing to a
full training run on the complete training set. Mirrors how Part A tuned its classical/DL
models and Part B tuned its RL agents — "max achievable" means actually searching, not
guessing once.

**Requires a GPU.** Runs 4-bit quantization (bitsandbytes) -- infeasible on an 8GB-RAM
CPU-only machine. Run this on Kaggle (free T4, Settings -> Accelerator -> GPU T4 x2) or any
other CUDA GPU.
""")

code("""\
# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════
import subprocess, sys
def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("transformers>=4.46", "peft>=0.13", "bitsandbytes>=0.44", "accelerate>=1.0",
            "datasets", "scikit-learn", "mlflow", "sentencepiece", "optuna", "optuna-integration",
            "torchao>=0.16.0", "tqdm")
# Kaggle's base image ships an older torchao (0.10.0) than current peft requires (>=0.16.0).
# peft's LoRA-layer dispatch checks torchao's availability unconditionally on EVERY adapter
# attachment (regardless of whether the model actually uses torchao quantization), and that
# check raises ImportError -- not a clean "unavailable" -- when an incompatible version is
# merely installed. Upgrading it here avoids get_peft_model() crashing on every model.
""")

code("""\
import os
# Must be set before torch touches CUDA at all. PyTorch's default caching allocator can end up
# fragmented enough after a few load/free cycles that a NEW allocation fails even though the
# aggregate free memory would be enough -- no single contiguous block is large enough. This is
# PyTorch's own documented fix (it's literally what the OOM error message suggests).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ast
import gc
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from huggingface_hub import login as hf_login
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import mlflow
from IPython.display import FileLink, HTML, display
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
""")

code("""\
# ══════════════════════════════════════════════════════════════
# CONFIG — split/seed identical to notebooks/07_finbert_peft.ipynb for a fair comparison
# ══════════════════════════════════════════════════════════════
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}
NUM_LABELS = 3

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.10
# SEntFiN headlines are short (median 10 words, 99th percentile 17 words, max 23) -- 128 tokens
# was wasting most of its budget on padding for nearly every example, and attention cost scales
# worse than linearly with sequence length. 64 comfortably covers the longest real headline
# (~23 words x ~1.3 subwords/word + special tokens is well under 64) with real speedup on top.
MAX_LENGTH = 64

# Stage 1 casts a wide net across DIFFERENT model families and sizes (~0.5B-3.8B) -- zero-shot
# inference is cheap (no training), so there's no reason to guess which family is best up
# front. Stage 2 (the expensive LoRA/QLoRA search) only runs on whichever TOP_K win this
# screen -- fine-tuning all 8 with a full search each would be a lot more compute for a
# question this cheap screen already answers well enough to narrow down.
# "quantize": True loads that model 4-bit (NF4) even for zero-shot inference -- a T4 only has
# ~14.5GB usable, and a 7B model's weights ALONE cost ~14GB in bf16 (7B x 2 bytes), leaving
# essentially no headroom for activations. This exact model OOM'd during the forward pass
# (not loading) on a first pass at this notebook; 4-bit cuts that to ~4GB, leaving real margin.
# Everything above the tiny 0.5B model is quantized now, even ones that would technically fit
# in bf16 in isolation -- a real run showed even a 4-bit 7B OOM'ing on a GPU that had *just*
# finished a 1.5B model cleanly, which means the failure mode isn't really "model too big", it's
# accumulated CUDA allocator fragmentation across models sharing one process. Quantizing more
# aggressively shrinks everyone's peak footprint as a second layer of defense on top of the
# expandable_segments fix and the CPU fallback (see load_causal_lm_with_fallback below).
# attn_implementation: "sdpa" (PyTorch's built-in fused attention, no extra package needed) is
# meaningfully faster than "eager" and is the right default -- "eager" is reserved specifically
# for Gemma-2, whose sliding-window attention has known compatibility issues with SDPA/flash
# implementations in several transformers versions (this is the exact tradeoff documented
# earlier in this notebook; it was just applied to every model instead of only the one that
# actually needs it, leaving real speed on the table for the other 8 candidates).
ZERO_SHOT_CANDIDATES = {
    "Qwen2.5-0.5B-Instruct": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "quantize": False, "params_billions": 0.5, "attn_implementation": "sdpa"},
    "Qwen2.5-1.5B-Instruct": {"hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "quantize": True, "params_billions": 1.5, "attn_implementation": "sdpa"},
    "Qwen2.5-3B-Instruct": {"hf_id": "Qwen/Qwen2.5-3B-Instruct", "quantize": True, "params_billions": 3.0, "attn_implementation": "sdpa"},
    # Qwen2.5-7B-Instruct deliberately excluded -- measured at 3.68h/epoch even with QLoRA, which
    # made a full search+train run for it ~7-9h on its own. Given FinBERT (110M params, smaller
    # than every candidate below) already hit 86.5% F1 on this exact task with LoRA, there's no
    # strong reason to believe this narrow a classification task needs a 7B model at all -- the
    # time cost wasn't justified by any evidence that it'd meaningfully outperform a well-tuned
    # smaller one, and the smaller candidates get a far more thorough search budget besides.
    "Gemma-2-2B-it": {"hf_id": "google/gemma-2-2b-it", "quantize": True, "params_billions": 2.6, "attn_implementation": "eager"},
    "Phi-3.5-mini-instruct": {"hf_id": "microsoft/Phi-3.5-mini-instruct", "quantize": True, "params_billions": 3.8, "attn_implementation": "sdpa"},
    "SmolLM2-1.7B-Instruct": {"hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "quantize": True, "params_billions": 1.7, "attn_implementation": "sdpa"},
    # Meta/Google gate these on HF (needs license acceptance + HF_TOKEN) -- included best-effort;
    # the zero-shot loop below skips any candidate that fails to load rather than crashing.
    "Llama-3.2-1B-Instruct": {"hf_id": "meta-llama/Llama-3.2-1B-Instruct", "quantize": False, "params_billions": 1.0, "attn_implementation": "sdpa"},
    "Llama-3.2-3B-Instruct": {"hf_id": "meta-llama/Llama-3.2-3B-Instruct", "quantize": True, "params_billions": 3.0, "attn_implementation": "sdpa"},
}
TOP_K_FOR_FINETUNING = 2   # only this many advance from the zero-shot screen to the expensive stage

# Search budget SCALES DOWN with model size -- a real run measured 3.68 HOURS for a single
# training epoch of Qwen2.5-7B-Instruct's QLoRA search (batch_size=8, MAX_LENGTH=128). The
# original fixed budget (25 trials x up to 4 epochs, with 5 "startup" trials that MedianPruner
# always lets run to completion before it has anything to compare against) meant AT LEAST
# 5 trials x 4 epochs x ~3.7h = 74+ hours before pruning could even engage for that one model --
# a budget sized for a 0.5-1B model, applied uniformly regardless of actual per-step cost.
# TPE sampling + median pruning still apply at every tier; only the trial/epoch COUNT scales.
def get_search_budget(params_billions: float) -> dict:
    if params_billions <= 1.0:
        return {"search_trials": 25, "search_epochs": 4, "full_epochs": 10, "n_startup_trials": 5}
    elif params_billions <= 2.5:
        return {"search_trials": 15, "search_epochs": 3, "full_epochs": 8, "n_startup_trials": 4}
    elif params_billions <= 4.5:
        return {"search_trials": 8, "search_epochs": 2, "full_epochs": 6, "n_startup_trials": 3}
    else:
        return {"search_trials": 4, "search_epochs": 1, "full_epochs": 3, "n_startup_trials": 2}


# Search-phase trials train on a SUBSAMPLE of the training set, not the full ~7.6k rows -- we're
# ranking configs RELATIVE to each other during search, which a representative subsample answers
# just as well, at a fraction of the wall-clock cost. Once a winning config is found, it retrains
# on the FULL training set for the final "full_epochs" run, so the actual reported result still
# reflects the complete data, not the subsample.
SEARCH_DATA_FRAC = 0.3
PATIENCE = 3

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type != "cuda":
    print("WARNING: no GPU detected. QLoRA (4-bit) requires CUDA -- this notebook will fail "
          "or be impractically slow without one. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")

BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///" + os.path.join(BASE_DIR, "mlflow.db"))
mlflow.set_experiment("IndianFinSentiment")   # same experiment as FinBERT+LoRA -- directly comparable


def package_outputs(checkpoint_name: str, auto_download: bool = False):
    \"\"\"Zips everything produced SO FAR (results CSVs/JSON, saved adapters+tokenizers, the
    mlflow db) into one archive and renders a download link.

    Called repeatedly throughout the notebook -- after Stage 1, after EVERY model/method combo
    in Stage 2, and once more at the very end -- not just once at the end. A Kaggle session
    timeout kills whatever cell is running AT THAT MOMENT; if the only packaging step were a
    final cell, a timeout during the (long) Stage 2 training loop would mean execution never
    reaches it and nothing gets bundled at all. Calling this after every completed unit of work
    means whatever finished before a timeout is always already zipped and link-ready, even if
    the notebook never reaches its last cell.

    On Kaggle, everything under /kaggle/working is ALSO downloadable via the notebook's own
    Output tab regardless of this -- this just makes it one click instead of hunting through
    dozens of files, and works the same way in a plain Jupyter/local run too.

    auto_download=True additionally fires a real browser download of the zip straight to your
    laptop's Downloads folder, no click needed -- there's no way for cloud-sandboxed Python code
    to reach your local filesystem directly, so this works by rendering the exact link IPython
    already generates correctly, then using a small JS snippet to click it programmatically the
    instant the cell runs. Left OFF by default on the repeated checkpoint calls throughout the
    notebook -- most browsers flag and silently block a page that fires off many automatic
    downloads in a row, so it's reserved for the one-off cell at the very end you can also
    re-run manually whenever you actually want a fresh download.
    \"\"\"
    # Built explicitly over just RESULTS_DIR/MODELS_DIR/mlflow.db rather than shutil.make_archive
    # over the whole BASE_DIR -- the latter would also sweep up unrelated cache/working files.
    import zipfile
    zip_path = os.path.join(BASE_DIR, f"outputs_{checkpoint_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for base_path in [RESULTS_DIR, MODELS_DIR]:
            if not os.path.isdir(base_path):
                continue
            for root, _, files in os.walk(base_path):
                for f in files:
                    full = os.path.join(root, f)
                    zf.write(full, arcname=os.path.relpath(full, BASE_DIR))
        mlflow_db = os.path.join(BASE_DIR, "mlflow.db")
        if os.path.exists(mlflow_db):
            zf.write(mlflow_db, arcname="mlflow.db")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Packaged outputs so far into {zip_path} ({size_mb:.1f} MB)")

    if auto_download:
        import uuid
        wrapper_id = f"dl_{uuid.uuid4().hex[:8]}"
        display(FileLink(zip_path, result_html_prefix=f'<span id=\"{wrapper_id}\">', result_html_suffix="</span>"))
        display(HTML(f\"\"\"
        <script>
        setTimeout(function() {{
            var container = document.getElementById("{wrapper_id}");
            var link = container ? container.querySelector("a") : null;
            if (link) {{ link.click(); }}
            else {{ console.warn("auto-download: couldn't find the generated file link"); }}
        }}, 400);
        </script>
        \"\"\"))
        print("Triggered an automatic browser download -- check your Downloads folder. "
              "If nothing arrives, your browser likely blocked it; the link above still works manually.")
    else:
        display(FileLink(zip_path))
""")

md("""\
## Hugging Face authentication (required for Llama-3.2 AND Gemma-2 -- both are gated)

Qwen2.5, Phi-3.5, and SmolLM2 are fully open -- no token needed. **Llama-3.2 and Gemma-2 are
different**: Meta and Google both require you to accept their license before you can download
the weights at all, gated at the HF Hub level, not something any code change here can bypass.
Three steps, one-time, covers both:

1. Log into huggingface.co, then visit and click **"Agree and access repository"** on each of:
   [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct),
   [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),
   [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
   (approval is instant for all three).
2. Create a **Read** access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. On Kaggle: **Add-ons -> Secrets -> Add a new secret**, name it `HF_TOKEN`, paste the token
   value. Then attach it to this notebook (the secrets picker that appears when you open this
   notebook, or Add-ons -> Secrets again). This keeps the token out of the notebook's saved
   output entirely -- never paste a real token directly into a cell.

If you skip this, Llama-3.2 and Gemma-2 will keep getting skipped with a clear "gated" message
and everything else still runs fine.
""")

code("""\
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    hf_login(token=hf_token)
    print("Hugging Face login OK -- Llama-3.2 and Gemma-2 will be included.")
except Exception as e:
    print(f"No HF_TOKEN secret found or login failed ({type(e).__name__}: {str(e)[:150]}) -- "
          f"Llama-3.2 and Gemma-2 candidates will be skipped, everything else is unaffected. "
          f"See the markdown above this cell to fix it.")
""")

md("""\
## Data — SEntFiN-v1.1, identical split to the FinBERT+LoRA notebook

Same loading rule: keep headlines where every mentioned entity shares one sentiment label
(single-label headlines, plus the ~88.5% of multi-entity headlines that happen to agree),
drop the ~11.5% where entities disagree (ambiguous for a single whole-headline label).
Same 70/15/15 stratified split, same seed (42) -- so any accuracy difference from FinBERT+LoRA
is attributable to the model/method, not a different train/test split.
""")

code("""\
def resolve_sentifin_path() -> str:
    candidates = ["../data/raw/SEntFiN-v1.1.csv", "../data/SEntFiN-v1.1.csv"]
    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f == "SEntFiN-v1.1.csv":
                    candidates.insert(0, os.path.join(root, f))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "SEntFiN-v1.1.csv not found. On Kaggle: Add Data -> upload SEntFiN-v1.1.csv as a "
        "dataset and attach it to this notebook. Locally: place it at data/raw/SEntFiN-v1.1.csv."
    )


def load_sentifin() -> pd.DataFrame:
    # NOTE: uses ast.literal_eval, not json.loads(text.replace("'", '"')) -- the Decisions
    # column is often already valid double-quoted JSON, and a handful of rows contain an
    # apostrophe INSIDE a value (e.g. "Moody's", "Dr Reddy's Labs"). A blind quote-replace
    # corrupts exactly those rows into invalid JSON and crashes the whole load. literal_eval
    # (wrapped in try/except, matching notebooks/07_finbert_peft.ipynb's proven loader) parses
    # the actual Python-literal structure regardless of which quote style is used.
    log.info("Loading SEntFiN-v1.1...")
    raw = pd.read_csv(resolve_sentifin_path())

    texts, labels = [], []
    dropped_unparseable, dropped_ambiguous = 0, 0
    for _, row in raw.iterrows():
        try:
            decisions = ast.literal_eval(row["Decisions"])
        except (ValueError, SyntaxError):
            dropped_unparseable += 1
            continue
        if not decisions:
            continue
        unique_labels = set(decisions.values())
        if len(unique_labels) != 1:
            dropped_ambiguous += 1
            continue
        texts.append(row["Title"])
        labels.append(unique_labels.pop())

    df = pd.DataFrame({"text": texts, "label": labels})
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL2ID.keys())]
    df["label_id"] = df["label"].map(LABEL2ID)
    log.info(f"Loaded {len(df)} single-label headlines "
             f"(dropped {dropped_ambiguous} ambiguous, {dropped_unparseable} unparseable)")
    log.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


df = load_sentifin()

train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["label_id"], random_state=SEED)
train_df, val_df = train_test_split(
    train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=train_val_df["label_id"], random_state=SEED
)
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
""")

md("""\
## Stage 1 — Zero-shot benchmark across 9 models, log-probability extraction (no fine-tuning at all)

Following the CLaC@FinMMEval-2026 paper's approach: one forward pass through the frozen
instruct model, extract the logits at the final token position, restrict the softmax to the
token-ID variants for {positive, negative, neutral} (both with/without a leading space, both
cases -- avoids brittleness to a single tokenization), and take the argmax. No text generation,
no fine-tuning -- purely testing what the model already knows from pretraining.
""")

code("""\
ZERO_SHOT_PROMPT = (
    "You are a financial sentiment classifier for Indian stock market headlines. "
    "Classify the sentiment of the following headline as exactly one word: "
    "positive, negative, or neutral.\\n\\nHeadline: {headline}\\nSentiment:"
)


def get_label_token_ids(tokenizer):
    \"\"\"All BPE variants of each label word (leading-space / no-space, lower/title case).\"\"\"
    label_variants = {
        "positive": ["positive", " positive", "Positive", " Positive"],
        "negative": ["negative", " negative", "Negative", " Negative"],
        "neutral": ["neutral", " neutral", "Neutral", " Neutral"],
    }
    label_token_ids = {}
    for label, variants in label_variants.items():
        ids = set()
        for v in variants:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if len(toks) >= 1:
                ids.add(toks[0])   # first token of the label word
        label_token_ids[label] = list(ids)
    return label_token_ids


@torch.no_grad()
def zero_shot_predict_batch(model, tokenizer, texts, label_token_ids, batch_size=16):
    model.eval()
    all_preds = []
    batch_starts = range(0, len(texts), batch_size)
    for i in tqdm(batch_starts, desc="  batches", total=len(batch_starts), leave=False):
        batch = texts[i:i + batch_size]
        prompts = [ZERO_SHOT_PROMPT.format(headline=t) for t in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
        logits = model(**enc).logits   # (batch, seq_len, vocab)

        # last REAL (non-pad) token position per sequence
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        last_logits = logits[torch.arange(len(batch)), seq_lens]   # (batch, vocab)

        for row_logits in last_logits:
            scores = {}
            for label, ids in label_token_ids.items():
                scores[label] = torch.logsumexp(row_logits[ids], dim=0).item()
            all_preds.append(max(scores, key=scores.get))
    return all_preds


def evaluate_predictions(y_true_labels, y_pred_labels, run_name):
    y_true = [LABEL2ID[l] for l in y_true_labels]
    y_pred = [LABEL2ID[l] for l in y_pred_labels]
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    result = {
        "run_name": run_name, "accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "per_class": {ID2LABEL[i]: {"precision": precision[i], "recall": recall[i], "f1": f1_per_class[i]} for i in range(3)},
        "confusion_matrix": cm.tolist(),
    }
    print(f"{run_name}: accuracy={acc:.4f} f1_macro={f1_macro:.4f} f1_weighted={f1_weighted:.4f}")
    return result


def hard_cuda_cleanup():
    \"\"\"More thorough than a bare empty_cache(): synchronize first so pending kernels actually
    finish (an in-flight op can keep a tensor alive past when del/gc.collect run), and reset
    the allocator's fragmentation stats so we can see whether it's actually clearing.\"\"\"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_causal_lm_with_fallback(hf_id: str, quantize: bool, attn_implementation: str = "sdpa"):
    \"\"\"Tries GPU first (quantized if requested); on CUDA OOM, falls back to CPU in float32
    (bitsandbytes 4-bit doesn't run on CPU, so quantization is dropped for the fallback --
    slower, but it means a model that can't fit on a shared/fragmented GPU still gets included
    in the benchmark rather than silently skipped).\"\"\"
    common_kwargs = {"trust_remote_code": True, "attn_implementation": attn_implementation}
    if torch.cuda.is_available():
        try:
            load_kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}, **common_kwargs}
            if quantize:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                )
            model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
            return model, "cuda"
        except torch.cuda.OutOfMemoryError as e:
            print(f"  GPU load OOM'd ({str(e)[:120]}) -- falling back to CPU (slower, but keeps this model in the benchmark)")
            hard_cuda_cleanup()
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32, device_map="cpu", **common_kwargs)
    return model, "cpu"


zero_shot_results = {}
skipped_candidates = []

ZERO_SHOT_BATCH_SIZE = 8     # GPU batch size
CPU_FALLBACK_BATCH_SIZE = 2  # CPU inference doesn't benefit from large batches the way GPU does

candidates_progress = tqdm(ZERO_SHOT_CANDIDATES.items(), desc="Stage 1: zero-shot models", unit="model")
for name, candidate in candidates_progress:
    candidates_progress.set_postfix_str(name)
    hf_id, quantize = candidate["hf_id"], candidate["quantize"]
    attn_impl = candidate.get("attn_implementation", "sdpa")
    stage1_start = time.time()
    print(f"\\n{'='*70}\\nZero-shot: {name}{' (4-bit)' if quantize else ''} -- started {time.strftime('%H:%M:%S')}\\n{'='*70}")

    # Defensive: clear any fragmentation left over from a previous candidate before this one
    # even starts, not just after it finishes.
    hard_cuda_cleanup()

    model = None
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"   # so the last real token is always at the same relative position

        print(f"  downloading/loading weights... (a first-time download of a multi-GB model can "
              f"take several minutes with no output in between -- that's normal, not stuck)")
        model, device_used = load_causal_lm_with_fallback(hf_id, quantize, attn_impl)
        print(f"  loaded on {device_used} in {time.time() - stage1_start:.0f}s, running inference on {len(test_df)} test headlines...")

        batch_size = ZERO_SHOT_BATCH_SIZE if device_used == "cuda" else CPU_FALLBACK_BATCH_SIZE
        label_token_ids = get_label_token_ids(tok)
        preds = zero_shot_predict_batch(model, tok, test_df["text"].tolist(), label_token_ids, batch_size=batch_size)
        result = evaluate_predictions(test_df["label"].tolist(), preds, f"{name}_zero_shot")
        zero_shot_results[name] = result
        print(f"  done in {time.time() - stage1_start:.0f}s total")

        with mlflow.start_run(run_name=f"{name}_zero_shot"):
            mlflow.log_params({"model": name, "method": "zero_shot", "quantized": quantize, "device": device_used})
            mlflow.log_metrics({"test_accuracy": result["accuracy"], "test_f1_macro": result["f1_macro"],
                                 "test_f1_weighted": result["f1_weighted"]})

        # Persisted after EVERY model, not just once at the end of Stage 1 -- Stage 1 alone
        # downloads and runs 9 models and can itself take a while, so this is also a real
        # timeout window, not just Stage 2's training loop.
        with open(os.path.join(RESULTS_DIR, "zero_shot_results_partial.json"), "w") as f:
            json.dump(zero_shot_results, f, indent=2, default=str)
        package_outputs(f"stage1_after_{name.replace(' ', '_')}")

    except Exception as e:
        # Gated models (Llama, Gemma-2) fail to LOAD without HF_TOKEN + accepted license.
        print(f"SKIPPING {name}: {type(e).__name__}: {str(e)[:200]}")
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print(f"  -> looks like a gated model. Accept the license at https://huggingface.co/{hf_id} "
                  f"and set HF_TOKEN, then re-run this cell if you want it included.")
        elif "out of memory" in str(e).lower():
            print(f"  -> CUDA OOM even after the CPU fallback attempt -- this model may just not fit "
                  f"in this session's available RAM either. Try lowering CPU_FALLBACK_BATCH_SIZE above.")
        skipped_candidates.append(name)

    finally:
        # Runs whether the trial succeeded, was skipped, or raised -- a model is never left
        # dangling on the GPU going into the next candidate.
        if model is not None:
            del model
        hard_cuda_cleanup()

if skipped_candidates:
    print(f"\\nSkipped {len(skipped_candidates)} candidate(s): {skipped_candidates}")

package_outputs("stage1_complete")
""")

md("""\
## Zero-shot leaderboard -- picking which models advance to fine-tuning

Ranked by test F1 (weighted). Only the top `TOP_K_FOR_FINETUNING` advance to the expensive
LoRA/QLoRA search below -- this is a proxy, not a guarantee (a model that zero-shots poorly
could in principle fine-tune better than one that zero-shots well), but it's the standard,
defensible way to keep a multi-model benchmark's compute bounded: screen cheap, invest
expensive compute only where the cheap screen says it's worth it.
""")

code("""\
zero_shot_leaderboard = pd.DataFrame([
    {"model": name, "accuracy": r["accuracy"], "f1_macro": r["f1_macro"], "f1_weighted": r["f1_weighted"]}
    for name, r in zero_shot_results.items()
]).sort_values("f1_weighted", ascending=False).reset_index(drop=True)

print(zero_shot_leaderboard.to_string(index=True))
zero_shot_leaderboard.to_csv(os.path.join(RESULTS_DIR, "zero_shot_leaderboard.csv"), index=False)

FINETUNE_CANDIDATES = {
    name: ZERO_SHOT_CANDIDATES[name]["hf_id"]
    for name in zero_shot_leaderboard["model"].head(TOP_K_FOR_FINETUNING).tolist()
}
print(f"\\nAdvancing to LoRA/QLoRA fine-tuning: {list(FINETUNE_CANDIDATES.keys())}")
""")

md("""\
## Stage 2 — LoRA and QLoRA fine-tuning (top `TOP_K_FOR_FINETUNING` zero-shot models only)

Both use `AutoModelForSequenceClassification` on the same decoder backbones (adds a linear
classification head reading the last non-padding token's hidden state -- the standard way to
fine-tune a causal LM for classification, and directly analogous to what FinBERT+LoRA did on
an encoder). QLoRA is identical except the frozen base weights are loaded 4-bit NF4-quantized
(`BitsAndBytesConfig`) before LoRA adapters are attached -- same adapters, much smaller memory
footprint, at some numerical precision cost.

Both get the same small hyperparameter search (rank, alpha, dropout, learning rate) before the
best configuration trains to convergence with early stopping on validation F1.
""")

code("""\
def tokenize_split(tokenizer, split_df):
    enc = tokenizer(split_df["text"].tolist(), padding="max_length", truncation=True,
                     max_length=MAX_LENGTH, return_tensors="pt")
    labels = torch.tensor(split_df["label_id"].tolist(), dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


TARGET_MODULE_SETS = {
    "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_and_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def build_classifier(hf_id, quantize: bool, config: dict, attn_implementation: str = "sdpa"):
    quant_kwargs = {}
    if quantize:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        dtype=torch.bfloat16, device_map=({"": 0} if torch.cuda.is_available() else None),
        trust_remote_code=True, attn_implementation=attn_implementation, **quant_kwargs,
    )
    base.config.pad_token_id = base.config.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["r"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
        target_modules=TARGET_MODULE_SETS[config["target_modules"]], bias="none",
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    return model


def run_epoch(model, loader, optimizer=None, desc=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    loss_fn = nn.CrossEntropyLoss()

    iterator = tqdm(loader, desc=desc or ("train" if training else "eval"), leave=False) if desc is not False else loader
    for input_ids, attention_mask, labels in iterator:
        input_ids, attention_mask, labels = input_ids.to(model.device), attention_mask.to(model.device), labels.to(model.device)
        with torch.set_grad_enabled(training):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


def train_with_early_stopping(model, train_loader, val_loader, lr, weight_decay, max_epochs, patience,
                               optuna_trial=None):
    \"\"\"optuna_trial, if passed, gets an intermediate report()+prune check after every epoch --
    this is what lets Optuna kill a clearly-losing config after 1-2 epochs instead of wasting
    the full per-model search-epoch budget on it (see get_search_budget).\"\"\"
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    best_val_f1, best_state, epochs_no_improve = -1, None, 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, desc=f"  epoch {epoch}/{max_epochs} train")
        val_loss, val_f1 = run_epoch(model, val_loader, desc=f"  epoch {epoch}/{max_epochs} eval")
        print(f"  epoch {epoch}/{max_epochs} ({time.time() - epoch_start:.0f}s): train_loss={train_loss:.4f} "
              f"train_f1={train_f1:.4f} | val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if optuna_trial is not None:
            optuna_trial.report(val_f1, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k or "classifier" in k or "score" in k}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return model, best_val_f1
""")

md("""\
### The search space (7 dimensions, TPE-sampled, median-pruned)

| Hyperparameter | Range | Why it's worth tuning |
|---|---|---|
| `r` (LoRA rank) | 4 - 64 (log) | Adapter capacity -- too low underfits, too high risks overfitting a 7.5k-row train set |
| `alpha` | 8 - 128 (log) | Scales the adapter's effective update magnitude relative to `r` |
| `dropout` | 0.0 - 0.3 | Regularization on the adapter path specifically |
| `learning_rate` | 1e-5 - 1e-3 (log) | The single highest-leverage knob for adapter fine-tuning |
| `weight_decay` | 1e-4 - 0.1 (log) | L2 regularization on the trainable (adapter + head) params |
| `batch_size` | {8, 16, 32} | Affects both gradient noise and throughput |
| `target_modules` | attention-only vs. attention+MLP | Whether the adapter touches only attention projections or MLP blocks too -- a real architectural choice, not just a training knob |

25 trials per (model, method) combination x 4 combinations (2 models x {LoRA, QLoRA}) = 100
trials total. `MedianPruner` stops a trial the moment its validation F1 falls behind the
median of previously completed trials at the same epoch, so most of the 100 never run to
completion -- this is what makes 25 real trials per combo affordable rather than nominal.
""")

code("""\
def make_objective(hf_id, quantize: bool, search_train_ds, val_ds, tag: str, search_epochs: int, attn_implementation: str):
    def objective(trial):
        config = {
            "r": trial.suggest_int("r", 4, 64, log=True),
            "alpha": trial.suggest_int("alpha", 8, 128, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "target_modules": trial.suggest_categorical("target_modules", list(TARGET_MODULE_SETS.keys())),
        }
        trial_start = time.time()
        print(f"\\n[{tag}] trial {trial.number} started {time.strftime('%H:%M:%S')}: {config}")

        train_loader = DataLoader(search_train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

        # model construction (loading + attaching LoRA) is INSIDE the try too -- an OOM there
        # is just as real as one during training, and previously wasn't covered by the cleanup.
        model = None
        try:
            model = build_classifier(hf_id, quantize, config, attn_implementation)
            _, val_f1 = train_with_early_stopping(
                model, train_loader, val_loader, config["lr"], config["weight_decay"],
                search_epochs, patience=99, optuna_trial=trial,
            )
        finally:
            if model is not None:
                del model
            hard_cuda_cleanup()
        print(f"[{tag}] trial {trial.number} done in {time.time() - trial_start:.0f}s, val_f1={val_f1:.4f}")
        return val_f1
    return objective


def hyperparameter_search(hf_id, full_train_ds, search_train_ds, val_ds, quantize: bool, tag: str,
                           budget: dict, attn_implementation: str):
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=budget["n_startup_trials"], n_warmup_steps=1),
        study_name=tag,
    )
    # catch=(...) means a trial that genuinely crashes (e.g. CUDA OOM on a large batch_size
    # draw) is recorded as FAILED and the study moves on, instead of the whole search dying.
    # show_progress_bar gives a live N/search_trials bar -- this search is the single longest-
    # running part of the whole notebook, so it's the place a progress bar matters most.
    # Trials train on search_train_ds (a subsample -- see SEARCH_DATA_FRAC), which is about
    # RANKING configs relative to each other, not squeezing out the best absolute number yet.
    study.optimize(
        make_objective(hf_id, quantize, search_train_ds, val_ds, tag, budget["search_epochs"], attn_implementation),
        n_trials=budget["search_trials"], catch=(RuntimeError,), show_progress_bar=True,
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(RESULTS_DIR, f"{tag}_optuna_trials.csv"), index=False)
    n_pruned = (trials_df["state"] == "PRUNED").sum()
    n_complete = (trials_df["state"] == "COMPLETE").sum()
    n_failed = (trials_df["state"] == "FAIL").sum()
    print(f"\\n[{tag}] search done: {n_complete} completed, {n_pruned} pruned, {n_failed} failed (of {budget['search_trials']})")
    if n_complete == 0:
        raise RuntimeError(
            f"[{tag}] every trial failed or was pruned -- likely every config OOM'd (check "
            f"{tag}_optuna_trials.csv for the actual errors). Try a smaller batch_size range "
            f"in the search space, or set quantize=True for this model if it isn't already."
        )
    print(f"[{tag}] best config: {study.best_params} (val_f1={study.best_value:.4f})")

    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"[{tag}] parameter importances: {importances}")
    except Exception as e:
        print(f"[{tag}] importance computation skipped: {e}")

    # The FINAL training run on the winning config uses the full training set, not the search
    # subsample -- the subsample is only meant to speed up RANKING configs during the search.
    best_config = study.best_params
    train_loader = DataLoader(full_train_ds, batch_size=best_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False)
    print(f"\\n[{tag}] training best config on the FULL training set to convergence "
          f"({budget['full_epochs']} epochs, patience={PATIENCE})...")
    final_model = build_classifier(hf_id, quantize, best_config, attn_implementation)
    final_model, final_val_f1 = train_with_early_stopping(
        final_model, train_loader, val_loader, best_config["lr"], best_config["weight_decay"],
        budget["full_epochs"], PATIENCE,
    )
    return final_model, best_config, final_val_f1, study
""")

code("""\
finetune_results = {}
finetune_artifacts = {}   # tag -> {hf_id, quantize, adapter_path, tokenizer_path} for reloading later

EVAL_BATCH_SIZE = 32   # only used for the held-out test set -- training batch size is tuned per trial

for name, hf_id in FINETUNE_CANDIDATES.items():
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = tokenize_split(tok, train_df)
    val_ds = tokenize_split(tok, val_df)
    test_ds = tokenize_split(tok, test_df)

    # Search trials train on a subsample -- ranking configs relative to each other doesn't need
    # the full ~7.6k rows, and this is the single biggest lever on search wall-clock time short
    # of cutting trial/epoch counts. The FINAL winning-config run below still uses the full set.
    search_train_df = train_df.sample(frac=SEARCH_DATA_FRAC, random_state=SEED)
    search_train_ds = tokenize_split(tok, search_train_df)

    attn_impl = ZERO_SHOT_CANDIDATES[name].get("attn_implementation", "sdpa")
    budget = get_search_budget(ZERO_SHOT_CANDIDATES[name]["params_billions"])
    print(f"\\n[{name}] search budget for this size ({ZERO_SHOT_CANDIDATES[name]['params_billions']}B params): {budget}")

    # Both methods run for every remaining candidate -- this restriction used to skip plain
    # LoRA for anything flagged quantize=True (every one of Qwen2.5-7B's 25 "lora" trials OOM'd,
    # since its ~14GB bf16 weights alone left no room for gradients/activations). With 7B
    # dropped from the candidate pool, everything left tops out at 3.8B (Phi-3.5), whose bf16
    # weights are ~7.6GB -- comfortably fits plain LoRA fine-tuning on a 14.56GB GPU alongside
    # gradients/activations, so a real LoRA-vs-QLoRA comparison is worth actually running now.
    # A trial that still OOMs on an unlucky large-batch draw just fails that one trial (caught
    # by hyperparameter_search's existing catch=(RuntimeError,)), not the whole combo.
    methods_to_run = [("lora", False), ("qlora", True)]

    for method, quantize in methods_to_run:
        tag = f"{name}_{method}"
        combo_start = time.time()
        print(f"\\n{'#'*70}\\n{tag} -- started {time.strftime('%H:%M:%S')} "
              f"({budget['search_trials']} search trials x up to {budget['search_epochs']} epochs on a "
              f"{int(SEARCH_DATA_FRAC*100)}% data subsample, then up to {budget['full_epochs']} final "
              f"epochs on the full training set)\\n{'#'*70}")

        # A combo that still fails entirely (every trial OOM'd or otherwise crashed) is now
        # skipped, not fatal -- previously an all-trials-failed RuntimeError from
        # hyperparameter_search() propagated all the way up and killed every OTHER model/method
        # combo still waiting to run, which is how one bad model (7B) took down the whole Stage 2
        # loop instead of just itself.
        try:
            model, best_config, val_f1, study = hyperparameter_search(
                hf_id, train_ds, search_train_ds, val_ds, quantize, tag, budget, attn_impl)
        except RuntimeError as e:
            print(f"SKIPPING {tag}: {e}")
            hard_cuda_cleanup()
            continue
        print(f"\\n[{tag}] entire combo finished in {(time.time() - combo_start)/60:.1f} min")

        model.eval()
        test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                logits = model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device)).logits
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.tolist())

        pred_labels = [ID2LABEL[p] for p in all_preds]
        true_labels = [ID2LABEL[l] for l in all_labels]
        result = evaluate_predictions(true_labels, pred_labels, tag)
        result["best_config"] = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) if not isinstance(v, str) else v
                                  for k, v in best_config.items()}
        result["n_trials_run"] = len(study.trials)
        result["n_trials_pruned"] = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        finetune_results[tag] = result

        adapter_path = os.path.join(MODELS_DIR, f"{tag}_adapter")
        tokenizer_path = os.path.join(MODELS_DIR, f"{tag}_tokenizer")
        model.save_pretrained(adapter_path)
        tok.save_pretrained(tokenizer_path)
        finetune_artifacts[tag] = {"hf_id": hf_id, "quantize": quantize, "attn_implementation": attn_impl,
                                    "adapter_path": adapter_path, "tokenizer_path": tokenizer_path}

        with mlflow.start_run(run_name=tag):
            mlflow.log_params({"model": name, "method": method, **result["best_config"]})
            mlflow.log_metrics({"val_f1": val_f1, "test_accuracy": result["accuracy"],
                                 "test_f1_macro": result["f1_macro"], "test_f1_weighted": result["f1_weighted"],
                                 "n_trials_run": result["n_trials_run"], "n_trials_pruned": result["n_trials_pruned"]})
            mlflow.log_artifacts(adapter_path, artifact_path="adapter")

        # Persisted after EVERY combo (up to 4: 2 candidates x {lora, qlora}), each of which can
        # itself take a long time (25-trial search + full training) -- this is the single most
        # likely place a Kaggle timeout actually lands, so it's the most important checkpoint.
        with open(os.path.join(RESULTS_DIR, "finetune_results_partial.json"), "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in finetune_results.items()},
                      f, indent=2, default=str)
        package_outputs(f"stage2_after_{tag}")

        # Drop the live model entirely rather than keep it (or a CPU copy of it) resident for
        # the rest of the loop -- up to 4 combos run here (2 candidates x {lora, qlora}), and
        # anything still referenced by a Python variable doesn't actually get freed by
        # empty_cache(). The adapter's already saved to disk; whichever combo turns out to be
        # the best just gets reloaded fresh from that saved adapter for the sanity-check re-test
        # below, once this whole loop is done and there's no competing model in memory.
        del model
        hard_cuda_cleanup()
""")

md("""\
## Final comparison

Every zero-shot candidate that loaded successfully, plus LoRA and QLoRA results for whichever
`TOP_K_FOR_FINETUNING` models won that screen, plus FinBERT+LoRA's known 86.5% F1 for
reference -- all on the identical test split.
""")

code("""\
comparison_rows = []
for name, result in zero_shot_results.items():
    comparison_rows.append({"model": name, "method": "zero_shot", "accuracy": result["accuracy"],
                             "f1_macro": result["f1_macro"], "f1_weighted": result["f1_weighted"]})
for tag, result in finetune_results.items():
    model_name, method = tag.rsplit("_", 1)
    comparison_rows.append({"model": model_name, "method": method, "accuracy": result["accuracy"],
                             "f1_macro": result["f1_macro"], "f1_weighted": result["f1_weighted"]})
comparison_rows.append({"model": "FinBERT+LoRA (Part A reference)", "method": "lora",
                         "accuracy": 0.8654, "f1_macro": None, "f1_weighted": 0.8652})

comparison_df = pd.DataFrame(comparison_rows).sort_values("f1_weighted", ascending=False)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(RESULTS_DIR, "llm_lora_qlora_comparison.csv"), index=False)

def reload_finetuned_model(tag: str):
    \"\"\"Rebuilds the base model fresh and attaches the TRAINED adapter weights saved to disk
    (PeftModel.from_pretrained loads the actual trained LoRA weights; get_peft_model/
    build_classifier would instead attach a new, randomly-initialized, untrained adapter).\"\"\"
    artifact = finetune_artifacts[tag]
    quant_kwargs = {}
    if artifact["quantize"]:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForSequenceClassification.from_pretrained(
        artifact["hf_id"], num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        dtype=torch.bfloat16, device_map=({"": 0} if torch.cuda.is_available() else None),
        trust_remote_code=True, attn_implementation=artifact.get("attn_implementation", "sdpa"), **quant_kwargs,
    )
    base.config.pad_token_id = base.config.eos_token_id
    model = PeftModel.from_pretrained(base, artifact["adapter_path"])
    tok = AutoTokenizer.from_pretrained(artifact["tokenizer_path"])
    return model, tok


best_tag = max(finetune_results, key=lambda k: finetune_results[k]["f1_weighted"])
best_result = finetune_results[best_tag]
best_model, best_tok = reload_finetuned_model(best_tag)
print(f"\\nBest fine-tuned variant: {best_tag} (f1_weighted={best_result['f1_weighted']:.4f})")
print(f"Confusion matrix ({best_tag}):")
print(pd.DataFrame(best_result["confusion_matrix"], index=["true_pos", "true_neu", "true_neg"],
                    columns=["pred_pos", "pred_neu", "pred_neg"]))
""")

md("""\
## Sanity check — does this fix FinBERT's known blind spot?

FinBERT+LoRA was 99%+ confidently WRONG on "corporate action" headlines that don't contain
classic profit/loss vocabulary (verified live against the dashboard's FastAPI service):
partnership announcements, new labs, contract wins, expansions. Testing the same failing
headlines against the best model found here.
""")

code("""\
BLIND_SPOT_HEADLINES = [
    ("TCS opens industrial AI lab in Bengaluru in partnership with Nvidia", "positive/neutral"),
    ("TCS partners with Nvidia", "positive/neutral"),
    ("Company launches new product line", "positive/neutral"),
    ("Company expands operations to new city", "positive/neutral"),
    ("Infosys wins new contract worth 500 crore", "positive"),
    ("Company invests in new manufacturing facility", "positive/neutral"),
    ("Reliance Industries posts record quarterly profit", "positive"),   # control: should stay positive
    ("Stock rallies after strong earnings", "positive"),                  # control
]

best_model.eval()   # best_model, best_tok already loaded above via reload_finetuned_model(best_tag)
print(f"{'Headline':<70} {'Expected':<18} {'Predicted'}")
print("-" * 105)
with torch.no_grad():
    for headline, expected in BLIND_SPOT_HEADLINES:
        enc = best_tok(headline, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(best_model.device)
        logits = best_model(**enc).logits
        pred = ID2LABEL[logits.argmax(dim=-1).item()]
        probs = torch.softmax(logits, dim=-1)[0]
        conf = probs.max().item()
        print(f"{headline[:68]:<70} {expected:<18} {pred} ({conf:.1%} confident)")
""")

code("""\
# ══════════════════════════════════════════════════════════════
# SAVE FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
summary = {
    "comparison": comparison_rows,
    "best_variant": best_tag,
    "best_metrics": {k: v for k, v in best_result.items() if k != "confusion_matrix"},
}
with open(os.path.join(RESULTS_DIR, "llm_lora_qlora_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\\nSaved comparison table + summary to {RESULTS_DIR}/")
print(f"Best adapter saved to {MODELS_DIR}/{best_tag}_adapter/")
print("\\nTo serve the fine-tuned model locally via Ollama afterward: merge the LoRA adapter into")
print("the base weights (peft's `merge_and_unload()`), export to GGUF with llama.cpp's converter,")
print("then `ollama create <name> -f Modelfile`.")

# Final comprehensive package -- everything from both stages, run to completion. If you're
# reading this printout, the notebook finished normally and this is just the last checkpoint,
# not the only one: the SAME zip-and-link step already ran after every zero-shot model and
# every fine-tuning combo above, specifically so a mid-run timeout still leaves a downloadable
# archive of whatever had completed up to that point.
package_outputs("final_complete")
""")

md("""\
## Download everything to your laptop now

Every checkpoint above already zipped its own progress and left a manual download link -- this
cell is the one that actually pushes a file into your **Downloads folder without you clicking
anything**. Run it whenever you want a fresh copy (it re-zips whatever exists on disk at that
moment, so it works mid-run too, not just at the end -- handy if you want a safety copy before
stepping away, not only once everything has finished).

If your browser blocks the automatic download (some do, especially after the first one or two
in a session), the manual link rendered just above the trigger still works -- click it directly.
""")

code("""\
package_outputs("manual_download", auto_download=True)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = "/Users/vijay/Desktop/sentiment-drl-trading/notebooks/13_llm_lora_qlora_sentiment.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(cells)} cells)")
