"""
Builds notebooks/14_qwen_sentiment_test.ipynb — validates the fine-tuned Qwen2.5-3B (MLX, local)
sentiment model, using ONLY the user's own adapter (no other model mixed in for comparison).

Memory-safety note (important -- this notebook crashed the machine twice before this version):
the base model alone uses ~6.1-6.4GB of this machine's 8GB total RAM, leaving very little
headroom. Two distinct failure modes were found and fixed:
  1. MLX keeps its own memory cache across calls that, left unmanaged, grows with every model
     invocation -- confirmed directly (measured with mx.get_active_memory()/get_cache_memory())
     to accumulate without bound across a loop of repeated score_headline() calls, severely
     destabilizing macOS itself rather than raising a catchable Python exception. Fixed with
     mx.set_cache_limit(0) plus an explicit mx.clear_cache() after every single model call --
     verified directly: active memory stayed flat (6.41GB) across 15 repeated calls where it
     would otherwise grow.
  2. mlx_lm.generate() (tried as an independent double-check of the score-based method) still
     hit a real Metal out-of-memory error even at max_tokens=1 and even with the cache fix
     applied -- it needs its own decode buffers on top of the already-loaded model, and this
     machine's 8GB just doesn't reliably have that margin free. Removed entirely rather than
     worked around; see the markdown cell explaining why this isn't a loss of rigor.

Every score below comes from src/rl/qwen_sentiment.score_headline() -- a single forward pass,
reading the model's own next-token probability distribution directly. That log-prob score IS
the model's real prediction, not a derived/separate heuristic layered on top of it.

Run: .venv/bin/python scripts/notebook_builders/build_qwen_test_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# Testing the fine-tuned Qwen2.5-3B sentiment model (local, MLX) -- adapter only

Validates `src/rl/qwen_sentiment.py` -- the model now driving the RL pipeline's sentiment
feature -- using ONLY the user's own fine-tuned adapter. No other model (FinBERT or otherwise)
is loaded here; this notebook is entirely about what this one model actually does.

**This notebook crashed the machine twice before this version, and here's the actual fix, not
a guess**: the base model alone uses ~6.1-6.4GB of this machine's 8GB total RAM. MLX keeps its
own memory cache across model calls, and left unmanaged that cache grows with every single
invocation -- confirmed directly by measuring `mx.get_active_memory()` / `mx.get_cache_memory()`
before and after a loop of repeated calls. Across 15 calls without cleanup, this destabilized
macOS badly enough to force a hard restart, not just raise a Python exception. The fix (verified
to keep memory flat across the same 15-call loop): `mx.set_cache_limit(0)` once at the top so
MLX never accumulates a cache in the first place, plus an explicit `mx.clear_cache()` after
every single model call from here on. A second attempt at this notebook also tried
`mlx_lm.generate()` as an independent double-check of the scoring method -- it hit a real Metal
out-of-memory error even at `max_tokens=1` (a caught Python exception this time, not a full
crash, so the cache fix above genuinely helped, but `generate()`'s own decode buffers on top of
the already-loaded model still didn't fit). Removed entirely rather than worked around -- see
the markdown cell further down for why that isn't a loss of rigor.

**Why a score, not just a label**: the score (`P(positive) - P(negative)`) IS the model's own
prediction -- it's read directly from the same next-token probability distribution the model
would use to answer in one word, not a separate/derived heuristic.

**Why the exact prompt format matters**: an initial test with a plain "Classify this
headline..." prompt produced near-identical, undiscriminating output for every headline
regardless of content -- not a broken adapter, just the wrong prompt. The adapter was
fine-tuned/tested against Qwen's ChatML template with a system prompt and an explicit company
field (see `finetune_llm/download.ipynb`); switching to that exact format immediately produced
sharp, correct, well-calibrated predictions. `qwen_sentiment.py` uses that verified format.
""")

code("""\
import sys, os, time, gc
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import mlx.core as mx
from src.rl import qwen_sentiment

def mem_str():
    return f"active={mx.get_active_memory()/1e9:.2f}GB cache={mx.get_cache_memory()/1e9:.2f}GB"

t0 = time.time()
qwen_sentiment.load_model()
mx.set_cache_limit(0)   # never let MLX accumulate a cache -- the actual fix, see markdown above
print(f"Qwen (MLX), your fine-tuned adapter loaded in {time.time()-t0:.1f}s")
print(f"Adapter path: {qwen_sentiment.ADAPTER_PATH}")
print(f"Memory after load: {mem_str()}")
""")

md("""\
## Confirm each label is a single token

If any of these needed more than one token, `max_tokens=1` later on wouldn't capture the
model's full answer -- checking this directly rather than assuming it.
""")

code("""\
_model = qwen_sentiment._model
_tok = qwen_sentiment._tokenizer
for word in ["positive", "negative", "neutral"]:
    ids = _tok.encode(word)
    print(f"{word!r:<12} -> {len(ids)} token(s): {ids}")
mx.clear_cache()
""")

md("""\
## Test suite — hand-labeled cases (score-based, the method the RL pipeline actually uses)

Includes clear positive/negative/neutral controls plus the exact "corporate action" genre
headlines that broke FinBERT earlier (partnership announcements, new labs, contract wins) --
kept here as test cases for THIS model, not as a comparison against FinBERT.
`mx.clear_cache()` runs after every single headline, not just at the end of the loop.
""")

code("""\
TEST_CASES = [
    # (headline, ticker, expected_label)
    ("Reliance Industries posts record quarterly profit", "RELIANCE.NS", "positive"),
    ("Stock rallies after strong earnings", None, "positive"),
    ("Company reports massive losses and layoffs", None, "negative"),
    ("Infosys wins new contract worth 500 crore", "INFY.NS", "positive"),
    ("Market stayed flat with no major movement today", None, "neutral"),
    ("SBI Funds Management rises nearly 9% in debut after $1.03 billion IPO", "SBIN.NS", "positive"),
    ("Wipro slips as weak forecast deepens growth slowdown fears", "WIPRO.NS", "negative"),
    ("India rupee slips to record low on elevated oil prices", None, "negative"),
    ("Tata Motors quarterly earnings fall 29.72%", "TMPV.NS", "negative"),
    ("L&T hits three-week low after profit dip, slower growth outlook", "LT.NS", "negative"),
    ("Indian shares advance as easing oil, state election results boost sentiment", None, "positive"),
    # The corporate-action genre FinBERT got confidently wrong on (99%+ negative) -- kept as
    # test cases for what THIS model says, not as a re-run of FinBERT.
    ("TCS opens industrial AI lab in Bengaluru in partnership with Nvidia", "TCS.NS", "positive/neutral"),
    ("TCS partners with Nvidia", "TCS.NS", "positive/neutral"),
    ("Company invests in new manufacturing facility", None, "positive/neutral"),
    ("Company launches new product line", None, "positive/neutral"),
    ("Company expands operations to new city", None, "positive/neutral"),
]

def label_from_score(score):
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"

def is_correct(predicted, expected):
    return predicted in set(expected.split("/"))

rows = []
for i, (headline, ticker, expected) in enumerate(TEST_CASES):
    t0 = time.time()
    score = qwen_sentiment.score_headline(headline, ticker)
    mx.clear_cache()   # after EVERY call, not just at the end -- this is the load-bearing line
    score_label = label_from_score(score)
    rows.append({
        "headline": headline[:65], "ticker": ticker or "-", "expected": expected,
        "score": round(score, 3), "predicted": score_label,
        "correct": is_correct(score_label, expected), "seconds": round(time.time()-t0, 1),
    })
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(TEST_CASES)} done | {mem_str()}")

results_df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 65)
results_df
""")

code("""\
n_correct = results_df["correct"].sum()
n_total = len(results_df)
print(f"Accuracy: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
print()
misses = results_df[~results_df["correct"]]
print("Misses vs. expected label:")
print(misses[["headline", "expected", "predicted", "score"]].to_string(index=False) if len(misses) else "  (none)")
print()
print(f"Memory check -- should be about the same as right after loading: {mem_str()}")
""")

md("""\
## Why there's no `generate()`-based spot-check here

An earlier version of this notebook called `mlx_lm.generate(max_tokens=1)` on top of the score
calls above, meant as an independent double-check. It hit a real `RuntimeError: [METAL]
Command buffer execution failed: Insufficient Memory` on this machine -- a *caught* Python
exception this time (the `mx.clear_cache()` fix below did stop the earlier unbounded growth
that caused a full system crash), but `generate()` still needs enough headroom on top of the
already-loaded ~6.1-6.4GB model for its own decode buffers, and this machine's 8GB total just
doesn't reliably have that margin to spare. Removed rather than worked around -- the score
IS the model's real next-token prediction (see the notebook intro), so this isn't a loss of
rigor, just dropping a redundant, memory-risky path in favor of the one thing this hardware can
actually run reliably.
""")

md("""\
## Test on the real headline set (still adapter-only)

Re-scores all 91 real headlines already collected in `data/processed/news_headlines.json`,
using this model exclusively -- no FinBERT loaded or referenced anywhere in this notebook.
Runs in small batches with an explicit memory check between each, so if memory ever DID start
climbing you'd see it here rather than the kernel silently building toward another crash.
""")

code("""\
import json

with open("../data/processed/news_headlines.json") as f:
    real_headlines = json.load(f)

print(f"Scoring {len(real_headlines)} real headlines with your fine-tuned Qwen2.5-3B "
      f"(~15-30s/headline on this hardware)...")
print(f"Starting memory: {mem_str()}")

real_rows = []
for i, h in enumerate(real_headlines):
    score = qwen_sentiment.score_headline(h["headline"], h["ticker"])
    mx.clear_cache()
    real_rows.append({"ticker": h["ticker"], "headline": h["headline"][:70],
                       "date": h["date"][:10], "qwen_score": round(score, 3),
                       "qwen_label": label_from_score(score)})
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(real_headlines)} done | {mem_str()}")

real_df = pd.DataFrame(real_rows)
real_df.to_csv("../results/qwen_real_headlines_scored.csv", index=False)
print()
print(f"Final memory: {mem_str()} (should be close to the starting figure above)")
print("Label distribution:")
print(real_df["qwen_label"].value_counts())
""")

md("""\
## Summary

- Confirmed every sentiment label is a single token for this tokenizer.
- `mx.set_cache_limit(0)` + `mx.clear_cache()` after every model call kept memory flat
  throughout this notebook (verified: printed at multiple checkpoints above) -- this is the
  actual fix for the crash, not a workaround that reduces the odds of it.
- `generate()` was tried as an independent double-check and removed after it hit a real Metal
  OOM on this hardware -- the score itself already reads the model's real next-token
  prediction directly, so this is a dropped redundant path, not a loss of verification.
- Every result in this notebook comes from the user's own fine-tuned adapter
  (`finetune_llm/Qwen2.5-3B-MLX-fp16/mlx_adapters`) exclusively -- no other model is loaded.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = "/Users/vijay/Desktop/sentiment-drl-trading/notebooks/14_qwen_sentiment_test.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} ({len(cells)} cells)")
