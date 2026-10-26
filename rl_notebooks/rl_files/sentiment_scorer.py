# sentiment_scorer.py — Entity-aware sentiment scoring with fine-tuned Qwen2.5-3B
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import json

from config import SENTIMENT_REPO, SENTIMENT_BASE, MAX_ARTICLE_WORDS

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


class SEntFiNScorer:
    """
    Drop-in replacement for the paper's LLaMA 3.2 1B zero-shot scorer.
    Uses fine-tuned Qwen2.5-3B (90.42% F1 vs ~75% zero-shot).

    Key advantage: entity-aware scoring via (headline, company) pairs,
    fixing systematic errors on corporate actions that FinBERT/LLaMA miss.

    Matches the paper's logit-extraction formula: score = P(+) - P(-) ∈ [-1, 1]
    """

    def __init__(self, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("Loading SEntFiN scorer (Qwen2.5-3B fine-tuned)...")
        self.device_str = device

        self.tok = AutoTokenizer.from_pretrained(SENTIMENT_REPO)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            SENTIMENT_BASE,
            dtype=torch.float16,
            device_map=device
        )
        self.model = PeftModel.from_pretrained(base, SENTIMENT_REPO)
        self.model.eval()

        # Device for tensor ops
        self.device = next(self.model.parameters()).device

        # Pre-compute token IDs for each label (paper's approach)
        # Enumerate BPE variants: leading space, lower, title-case
        self.label_token_ids = {}
        for label in ["positive", "negative", "neutral"]:
            variants = [label, " " + label, label.capitalize(), " " + label.capitalize()]
            ids = []
            for v in variants:
                tids = self.tok.encode(v, add_special_tokens=False)
                ids.extend(tids)
            self.label_token_ids[label] = list(set(ids))

        self._n_pos = len(self.label_token_ids["positive"])
        self._n_neg = len(self.label_token_ids["negative"])
        self._all_ids = (
            self.label_token_ids["positive"] +
            self.label_token_ids["negative"] +
            self.label_token_ids["neutral"]
        )
        print("✅ SEntFiN scorer ready")

    def score_article(self, article: str, asset: str) -> float:
        """
        Score one article for given asset.
        Returns float ∈ [-1, 1]: positive=bullish, negative=bearish.
        Matches paper's Equation (2): score = P(+) - P(-)
        """
        # Truncate to first MAX_ARTICLE_WORDS words (matching paper)
        words = article.split()[:MAX_ARTICLE_WORDS]
        text  = " ".join(words)

        # Asset name for entity-aware prompt
        asset_name = {"TSLA": "Tesla (TSLA)", "BTC-USD": "Bitcoin (BTC)"}.get(
            asset.upper(), asset
        )

        SYSTEM = (
            f"You are a financial sentiment analyst. Given a news headline and a company name, "
            f"classify the sentiment towards {asset_name}. "
            f"Reply with exactly one word: positive, negative, or neutral."
        )

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"Headline: {text}\nCompany: {asset_name}\nSentiment:"}
        ]

        try:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models that don't support system role
            combined = f"{SYSTEM}\n\nHeadline: {text}\nCompany: {asset_name}\nSentiment:"
            messages_fb = [{"role": "user", "content": combined}]
            prompt = self.tok.apply_chat_template(
                messages_fb, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=400
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits at final (prediction) token — same as paper
        last_logits = outputs.logits[0, -1, :]   # shape: [vocab_size]

        # Softmax over label token IDs only
        label_logits = last_logits[self._all_ids]
        label_probs  = torch.softmax(label_logits, dim=-1).cpu().float()

        p_pos = label_probs[:self._n_pos].sum().item()
        p_neg = label_probs[self._n_pos:self._n_pos + self._n_neg].sum().item()
        # p_neutral implicitly reduces p_pos/p_neg toward 0 (paper's insight)

        return float(p_pos - p_neg)

    def score_daily(self, articles: List[str], asset: str) -> float:
        """
        Score a list of articles and return mean daily sentiment.
        Returns 0.0 on days with no articles.
        """
        if not articles:
            return 0.0
        scores = [self.score_article(a, asset) for a in articles]
        return float(np.mean(scores))

    def score_dataset(self, df: pd.DataFrame, asset: str,
                      cache_name: Optional[str] = None) -> pd.Series:
        """
        Score all days in a DataFrame.
        df must have columns: [date, articles]
        Returns Series indexed by date.

        Uses disk cache to avoid re-scoring.
        """
        cache_path = CACHE_DIR / f"sentiment_{cache_name or asset}.json" if cache_name else None

        # Load from cache
        cached = {}
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"  Loaded {len(cached)} cached sentiment scores")

        scores = {}
        n_total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            date_str = str(row["date"].date())

            if date_str in cached:
                scores[row["date"]] = cached[date_str]
                continue

            score = self.score_daily(row["articles"], asset)
            scores[row["date"]] = score
            cached[date_str] = score

            if i % 50 == 0:
                print(f"  Scoring sentiment: {i}/{n_total} days")
                # Save cache checkpoint
                if cache_path:
                    with open(cache_path, "w") as f:
                        json.dump(cached, f)

        # Final cache save
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(cached, f)
            print(f"  ✅ Sentiment cache saved: {cache_path}")

        return pd.Series(scores)


class MockScorer:
    """
    Fast mock scorer for testing without GPU.
    Returns random scores in [-1, 1].
    """
    def score_dataset(self, df, asset, cache_name=None):
        np.random.seed(42)
        scores = {}
        for _, row in df.iterrows():
            scores[row["date"]] = np.random.uniform(-0.3, 0.3)
        return pd.Series(scores)
