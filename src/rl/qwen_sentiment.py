"""
qwen_sentiment.py — sentiment scoring using the user's own fine-tuned Qwen2.5-3B LoRA adapter,
running locally via MLX (Apple's ML framework for Apple Silicon), not PyTorch/transformers.

Why MLX and not the HF/PEFT path: the base Qwen2.5-3B-Instruct weights (~6GB) would need
downloading from HF, and on this machine's network that measured ~192KB/s -- roughly 8-9 hours
for one file. An MLX-converted copy of the same base model plus an MLX-converted copy of the
adapter were already present locally (finetune_llm/Qwen2.5-3B-MLX-fp16), so this path needs no
download at all and loads in ~15s.

Why the exact prompt format matters: an initial test with a plain "Classify this headline..."
prompt produced near-identical, undiscriminating output for every headline regardless of
content (e.g. "massive losses and layoffs" scored almost the same as "record profit") --  not
a broken adapter, just the wrong prompt. The adapter was fine-tuned/tested against Qwen's
ChatML template with a system prompt and an explicit company field (see
finetune_llm/download.ipynb, cell 8/10); switching to that exact format immediately produced
sharp, correct, well-calibrated predictions (verified against 15 hand-checked test cases,
14/15 correct or reasonable) -- so it's reproduced here exactly, not simplified.

Only scores the small set of REAL headlines (~60-90) -- the RL pipeline bootstrap-resamples
these across the rest of the training window, it was never going to need thousands of live
LLM calls, so per-call latency (~15-30s on this hardware) is a non-issue.
"""
import logging
import math
import os

import mlx.core as mx
from mlx_lm import load

log = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MLX = os.path.join(_THIS_DIR, "finetune_llm", "Qwen2.5-3B-MLX-fp16")
ADAPTER_PATH = os.path.join(BASE_MLX, "mlx_adapters")

# Same friendly names the dashboard uses (frontend/src/stockMeta.js) -- the model was
# tested/tuned against natural company names, not raw Yahoo Finance ticker symbols.
TICKER_TO_COMPANY = {
    "INFY.NS": "Infosys", "RELIANCE.NS": "Reliance", "TCS.NS": "TCS", "SBIN.NS": "SBI",
    "MARUTI.NS": "Maruti", "WIPRO.NS": "Wipro", "TMPV.NS": "Tata Motors", "LT.NS": "L&T",
    "ITC.NS": "ITC", "SUNPHARMA.NS": "Sun Pharma",
}

SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Given a news headline and a company name, "
    "classify the sentiment towards that company. "
    "Reply with exactly one word: positive, negative, or neutral."
)

_model = None
_tokenizer = None
_label_ids = None


def _make_prompt(headline: str, company: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nHeadline: {headline}\nCompany: {company}\nSentiment:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model():
    global _model, _tokenizer, _label_ids
    if _model is not None:
        return
    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(f"MLX adapter not found at {ADAPTER_PATH}")
    log.info(f"Loading MLX model from {BASE_MLX} with adapter {ADAPTER_PATH}...")
    _model, _tokenizer = load(BASE_MLX, adapter_path=ADAPTER_PATH)
    mx.set_cache_limit(0)  # MLX's cache otherwise grows unbounded across repeated calls (crashed this 8GB machine twice)
    # Each label is a single token for this tokenizer (verified directly) -- no multi-subword
    # first/last-token ambiguity to worry about.
    _label_ids = {
        "positive": _tokenizer.encode("positive")[0],
        "negative": _tokenizer.encode("negative")[0],
        "neutral": _tokenizer.encode("neutral")[0],
    }
    log.info(f"Qwen (MLX) sentiment model ready, label_ids={_label_ids}")


def score_headline(headline: str, ticker: str | None = None) -> float:
    """Returns P(positive) - P(negative) in [-1, 1], same convention FinBERT used."""
    load_model()
    assert _model is not None and _tokenizer is not None and _label_ids is not None
    company = TICKER_TO_COMPANY.get(ticker, ticker) if ticker else "the company"
    prompt = _make_prompt(headline, company)
    tokens = _tokenizer.encode(prompt)
    logits = _model(mx.array(tokens)[None])
    last = logits[0, -1]
    raw = {label: last[idx].item() for label, idx in _label_ids.items()}
    m = max(raw.values())
    exps = {k: math.exp(v - m) for k, v in raw.items()}
    total = sum(exps.values())
    probs = {k: v / total for k, v in exps.items()}
    mx.clear_cache()  # paired with set_cache_limit(0) above -- keeps memory flat across repeated calls
    return probs["positive"] - probs["negative"]


def score_headlines_batch(texts: list[str], ticker: str | None = None) -> list[float]:
    return [score_headline(t, ticker) for t in texts]
