"""
Daily sentiment scoring for the paper replication, using the user's fine-tuned Qwen2.5-3B LoRA
adapter (src/rl/qwen_sentiment.py's model/tokenizer) in place of the paper's LLaMA-3.2-1B.

The scoring mechanism (single forward pass, softmax restricted to {positive,negative,neutral}
token IDs, score = P(positive) - P(negative)) is identical to the paper's Section 4.3 -- only
the backbone model differs, which is exactly the requested swap. Articles are truncated to the
first 300 words to match the paper's truncation choice (focus on headline + lede).
"""
import math

import mlx.core as mx

from src.rl import qwen_sentiment as qs

SYSTEM_PROMPT_TMPL = (
    "You are a financial sentiment classifier for {asset}. Respond with exactly one word: "
    "positive, negative, or neutral."
)


def _make_prompt(asset: str, article_text: str) -> str:
    truncated = " ".join(article_text.split()[:300])
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT_TMPL.format(asset=asset)}<|im_end|>\n"
        f"<|im_start|>user\n{truncated}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def score_article(asset: str, article_text: str) -> float:
    qs.load_model()
    assert qs._model is not None and qs._tokenizer is not None and qs._label_ids is not None
    prompt = _make_prompt(asset, article_text)
    tokens = qs._tokenizer.encode(prompt)
    logits = qs._model(mx.array(tokens)[None])
    last = logits[0, -1]
    raw = {label: last[idx].item() for label, idx in qs._label_ids.items()}
    m = max(raw.values())
    exps = {k: math.exp(v - m) for k, v in raw.items()}
    total = sum(exps.values())
    probs = {k: v / total for k, v in exps.items()}
    mx.clear_cache()
    return probs["positive"] - probs["negative"]
