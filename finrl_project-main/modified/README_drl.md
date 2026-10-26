# Sentiment-Aware Deep RL Stock Trading

This folder is the working, fixed-up version of a Deep Reinforcement Learning stock-trading project:
three RL algorithms (A2C, PPO, TD3) are trained to trade a 10-stock portfolio, first using only price/technical
data, then using price data plus a daily sentiment score derived from real financial news headlines — scored by
a **Qwen2.5-3B model that was fine-tuned specifically for this task**. The goal was to answer one question:
*does knowing whether the news was positive or negative make an RL trading agent better?*

For a from-scratch explanation of the RL theory, the math behind each algorithm, and a cell-by-cell code
walkthrough (written for interview prep, assumes no RL background), see the companion guide:
**[Sentiment-Aware Deep RL Trading, From First Principles](https://claude.ai/code/artifact/a8b34551-6d38-43d0-909d-a57d71aad0c5)**.

---

## 1. What's in this folder

| Path | What it is |
|---|---|
| `Kaggle_0_SentimentScoring.ipynb` | Runs on Kaggle (GPU). Streams FNSPID news, scores every headline with the fine-tuned Qwen adapter, saves `news.csv` + `scored.csv`. Run **once**, not per-config. |
| `Kaggle_1_NoSA/` | Baseline: no sentiment. Trains A2C/PPO/TD3 on price + technical indicators only. |
| `Kaggle_2_RawSA/` | Adds sentiment as a raw daily average — no smoothing, no filtering. |
| `Kaggle_3_OptSA/` | Adds sentiment filtered to days with ≥2 headlines, EMA-smoothed, standardized. |
| `Kaggle_4_OptSA_Vol/` | Same as OptSA, plus a second feature: 3-day rolling sentiment volatility. |
| `multiseed_ablation_results.csv` / `_summary.csv` | 5-seed validation of TD3 and A2C, with vs. without sentiment — the results that actually get trusted (§5). |
| `news.csv` / `scored.csv` | Real output of `Kaggle_0` — 60,343 headlines matched to the 10 tickers, 2020–2023, with Qwen sentiment scores. |

Each `Kaggle_N_*/` folder is self-contained: its own notebook, its own `processed_full*.csv` input, its own
`result.csv` / `evaluation_metrics_table.csv` output — so results never overwrite each other.

---

## 2. The RL models trained

Every one of the four configs above trains the **same three algorithms**, independently, on the same
50,000-timestep budget, so results are directly comparable:

| Algorithm | Type | Why it's here |
|---|---|---|
| **A2C** (Advantage Actor-Critic) | On-policy, stochastic | Simplest baseline, trains fastest (~35s) |
| **PPO** (Proximal Policy Optimization) | On-policy, stochastic, clipped updates | Standard, widely-used, more conservative than A2C |
| **TD3** (Twin Delayed DDPG) | Off-policy, deterministic | Built for continuous control; reuses experience via a replay buffer, ends up the strongest performer |

**The environment (FinRL's `StockTradingEnv`), the same in all four configs:**

- **Universe**: AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, TSLA, BA, DIS
- **State**: `1 + 2n + mn` — cash balance, current price × 10 stocks, shares held × 10 stocks, plus every
  technical/sentiment indicator × 10 stocks. State size grows with `m` (indicator count): **121** (no sentiment,
  10 indicators) → **131** (+1 sentiment feature) → **141** (+2 sentiment features, OptSA_Vol).
- **Action**: continuous, ±100 shares max per stock per day (`hmax=100`)
- **Reward**: change in total portfolio value between days, scaled by `1e-4`
- **Transaction cost**: 0.1% per trade (buy and sell)
- **Technical indicators**: MACD, Bollinger Bands, RSI(30), CCI(30), DX(30), SMA(30), SMA(60), VIX, turbulence index
- **Train / test split**: train on 2020-01-01 → 2023-01-01, backtest (trade) on all of 2023 — chronological,
  never random, to avoid lookahead bias
- **Starting capital**: $1,000,000

That's **12 trained policies from single runs** (3 algorithms × 4 configs), plus a further **20 policies**
from the 5-seed validation (TD3 and A2C, with/without sentiment) described in §5.

---

## 3. The sentiment model journey — from classical ML to fine-tuned Qwen

The Qwen adapter used throughout this project (§3.6) wasn't the first thing tried. It's the end point of a
deliberate progression through five stages of modeling, all on the same underlying task — classify a
financial headline's sentiment as positive/negative/neutral — tracked end-to-end in MLflow (§3.5). This
section walks through what was tried, in order, and compares the results honestly.

> **One methodology note before the numbers**: stages 3.1–3.3 below trained on **9,462 headlines**, each
> reduced to a single label (`data/processed/{train,val,test}.csv`, 7,095 / 947 / 1,420 split). The final Qwen
> stage (3.6) instead trained on **14,409 `(headline, company, label)` triples** — the same underlying SEntFiN
> headlines, but *exploded per company mentioned*, so a headline naming two companies with different sentiment
> becomes two training examples instead of one collapsed label. That's a genuine methodological improvement
> (it's also exactly how sentiment gets used downstream — per ticker, not per headline) but it means the
> accuracy numbers below aren't perfectly apples-to-apples across every stage. The relative story — classical
> ML plateaus, from-scratch deep learning doesn't clearly beat it, pretrained transformers jump well past
> both — is real regardless.

### 3.1 Stage 1 — Classical ML baselines

Bag-of-Words and TF-IDF (`bow_vs_tfidf.ipynb`) were swept across `max_features` (5K–30K) and n-gram ranges
(1–3) as the feature representation, feeding six classical classifiers, each with a real hyperparameter search
(`*_with_hpt.ipynb` notebooks — grid search, cross-validation, and for two of them an incremental/SGD variant):

| Model | Best test accuracy | Best F1 (weighted) |
|---|---:|---:|
| Logistic Regression | 81.3% | 0.813 |
| Linear SVM | 81.1% | 0.811 |
| XGBoost | 79.3% | 0.793 |
| Random Forest | 79.0% | 0.790 |
| LightGBM | 78.1% | 0.781 |
| KNN (TF-IDF + SVD) | 73.2% | 0.730 |

Linear models (Logistic Regression, SVM) edged out every tree ensemble here — a common pattern for sparse,
high-dimensional TF-IDF features, where linear decision boundaries tend to generalize better than tree splits.

### 3.2 Stage 2 — Deep learning from scratch

Four neural architectures, each implemented and trained from random initialization (no pretraining), each
with its own hyperparameter-tuned run (`03_textcnn.ipynb` → `06_transformer_scratch.ipynb`):

| Model | Best test accuracy | Best F1 (weighted) |
|---|---:|---:|
| Transformer (from scratch) | 77.7% | 0.777 |
| BiLSTM | 77.3% | 0.773 |
| CNN-BiLSTM-Attention | 76.5% | 0.764 |
| TextCNN | 76.4% | 0.764 |

Notably, **none of these clearly beat the best classical ML baseline** (Logistic Regression, 81.3%). This is
the expected result, not a bug: architectures like BiLSTM and Transformer have far more parameters to learn
from scratch, and ~9,500 labeled headlines is a small dataset for that — without pretraining on a much larger
text corpus first, a from-scratch deep net doesn't have enough signal to out-learn a well-regularized linear
model on sparse TF-IDF features.

### 3.3 Stage 3 — FinBERT + parameter-efficient fine-tuning

The next stage switched to **pretrained** transformers — FinBERT, already pretrained on financial text —
fine-tuned with three different PEFT (parameter-efficient fine-tuning) methods rather than full fine-tuning
(`07_finbert_peft_.ipynb`, `finbert_peft_v2.ipynb`, `finbert_peft_v3.ipynb`):

| Method | Test accuracy | Test F1 (weighted) | Trainable params |
|---|---:|---:|---:|
| LoRA (r=16, q+k+v) | 86.5% | 0.865 | 0.80% |
| QLoRA (r=8, q+v, 4-bit) | 86.3% | 0.863 | 0.44% |
| Prefix Tuning (20 tokens) | 70.8% | 0.706 | 0.34% |

(Best numbers shown are from the most refined iteration, v3; earlier iterations of the same LoRA/QLoRA setup
landed lower, ~76–77%, before further tuning.) This was the first clear jump past the classical ML ceiling —
starting from a model that already understands financial language, rather than learning language structure
from scratch on 9,500 examples, was worth roughly +5 points of accuracy over the best classical baseline.
Prefix tuning (which only trains a small set of virtual prompt tokens, no attention weights) clearly
underperformed LoRA/QLoRA here — too few trainable parameters for this task.

### 3.4 Full comparison, all five stages

| Stage | Model | Accuracy | F1 (weighted) |
|---|---|---:|---:|
| Classical ML | Logistic Regression (best) | 81.3% | 0.813 |
| Classical ML | Linear SVM (best) | 81.1% | 0.811 |
| From-scratch DL | Transformer (from scratch) | 77.7% | 0.777 |
| From-scratch DL | BiLSTM | 77.3% | 0.773 |
| Pretrained + PEFT | **FinBERT + LoRA** | **86.5%** | **0.865** |
| Pretrained + PEFT | FinBERT + QLoRA | 86.3% | 0.863 |
| **Final, production** | **Qwen2.5-3B-Instruct + LoRA** | **90.4%** | **0.905** |

The trend is a clean staircase: **classical ML (~81%) → from-scratch deep learning (~76–78%, no clear win) →
pretrained + PEFT fine-tuning (~86%) → a larger pretrained instruction-tuned model, also PEFT fine-tuned
(~90%)**. Model scale and pretraining mattered far more here than architecture novelty — the biggest single
jump in the whole table is "start from a model that already knows language" (stage 2 → stage 3), not any
particular architecture choice within a stage.

### 3.5 MLflow experiment tracking

Every run above (and every hyperparameter trial behind it) was logged to a local MLflow instance backed by
SQLite (`notebooks/mlflow.db`), under two experiments — `IndianFinSentiment` (models) and
`IndianFinSentiment_ClassicalML` (BoW/TF-IDF vectorizer sweeps). In total: **877 logged runs** for the
sentiment-modeling work alone (a separate `mlflow.db` at the project root tracks the unrelated RL trading
experiments under `DRLTrading`).

What actually got logged per run: hyperparameters (`params` table — e.g. every `C`/`kernel` combination for
SVM, every `num_leaves`/`learning_rate` combination for LightGBM), and metrics (`metrics` table — accuracy,
per-class F1, weighted F1, cross-validation mean/std, train time, trainable parameter count). Each model
family's exhaustive sweep (dozens to ~100+ trials per model) is findable by its verbose hyperparameter-encoded
run name (e.g. `LightGBM_subsample=0.8_reg_lambda=0.1_...`), with the winning configuration re-logged under a
clean `_BEST` or `_HPT_tuned` name for fast lookup — that naming convention is what made pulling the
comparison tables above possible directly from the tracking database, without re-running anything or digging
through old notebook cell outputs.

### 3.6 The final model: fine-tuning Qwen2.5-3B

The sentiment score used everywhere in this project — the one actually feeding the RL environment — isn't
FinBERT (despite what some leftover markdown text in the trading notebooks still says) or any of the models
above. It's a **LoRA-fine-tuned Qwen2.5-3B-Instruct**, trained specifically for this task, one further step
past FinBERT-LoRA. The fine-tuning notebook lives at `finetune_llm/qwen_final_with_hp.ipynb` (outside this
folder, in the main project). Summary:

#### Dataset
[**SEntFiN-v1.1**](https://data.mendeley.com/datasets/b7fghyxvcs/3) — financial news headlines, each labeled
per company mentioned (one headline can name several companies with different sentiment for each). Exploded
into `(headline, company, label)` triples:

| Split | Rows | Notes |
|---|---:|---|
| Train | 11,527 | 80%, stratified by label |
| Validation | 1,441 | 10% — used during hyperparameter search |
| Test | 1,441 | 10% — **touched only once**, after the best hyperparameters were already locked in |

Class balance (before splitting): neutral 5,515 (38.3%) · positive 5,075 (35.2%) · negative 3,819 (26.5%).

#### Prompt format
Trained as instruction-following chat, not raw text classification:

```
System:    You are a financial sentiment analyst. Given a news headline and a company
           name, classify the sentiment towards that company. Reply with exactly one
           word: positive, negative, or neutral.
User:      Headline: {headline}
           Company: {company}
           Sentiment:
Assistant: {label}
```

At inference time (used throughout `Kaggle_0_SentimentScoring.ipynb`), the model does **one forward pass**
(no text generation) and reads the softmax probability at just the `positive`/`negative`/`neutral` token
positions: `score = P(positive) − P(negative)`, a continuous value in `[-1, 1]` — cheaper and more
information-dense than a hard 3-way label.

#### LoRA configuration (final)

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` (frozen) |
| Rank (`r`) | 32 |
| Alpha | 128 |
| Dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all attention + MLP projections) |
| Trainable params | 59,867,136 / 3,145,805,824 total → **1.90%** |

#### Hyperparameter tuning process

The rank/learning-rate/alpha combination above was selected in earlier sweep rounds; this notebook runs the
final round — **epoch count** — holding everything else fixed:

| Config | Epochs | Val Macro F1 |
|---|---|---|
| Known baseline (typical defaults: r=16, lr=2e-4, α=32) | 3 | 0.8894 |
| r=32, lr=5e-4, α=128 | 2 | tested, epochs=3 selected as best |
| r=32, lr=5e-4, α=128 | **3 (chosen)** | 0.9039 |
| r=32, lr=5e-4, α=128 | 5 | tested (overfitting check) |

**Training setup**: batch size 4 × gradient accumulation 8 (effective batch 32), learning rate 5e-4 with cosine
schedule + 50 warmup steps, fp16, `adamw_torch`, gradient checkpointing, max sequence length 128 tokens.

#### Final result (test set, evaluated once)

| Metric | Baseline HPs | **HP-tuned (final)** |
|---|---:|---:|
| Test Accuracy | — | **0.9042** |
| Test Macro F1 | 0.9006 | **0.9047** |
| Positive F1 | — | 0.9166 |
| Negative F1 | — | 0.9043 |
| Neutral F1 | — | 0.8932 |
| Inference speed | — | ~59ms/sample |

The tuned adapter improved test Macro F1 by +0.0041 over reasonable defaults, and is hosted privately at
[`vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned`](https://huggingface.co/vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned)
on Hugging Face — loaded via `PeftModel.from_pretrained(base_model, adapter_repo)` everywhere it's used in this
project.

> **Security note**: while reviewing `finetune_llm/qwen_final_with_hp.ipynb`, a live-looking Hugging Face token
> was found hardcoded in plaintext in one cell (`login(token="hf_...")`). It was not reproduced here or anywhere
> in this repo — **rotate that token** on huggingface.co/settings/tokens and remove the hardcoded line from that
> notebook.

---

## 4. The sentiment scoring → training pipeline

News scoring is expensive (GPU, ~60K headlines through a 3B model) and only needs to happen **once** — all
three sentiment configs read from the same scored output rather than each re-streaming and re-scoring
independently:

```
Kaggle_0_SentimentScoring.ipynb  (Kaggle, GPU)
  1. Stream FNSPID from Hugging Face (never fully downloaded — ~60K of ~50M+ rows match our 10 tickers)
  2. Score every matched headline with the fine-tuned Qwen adapter  →  scored.csv
        ↓  (brought back locally)
Local merge  (no GPU needed — plain pandas)
  3. Aggregate scored.csv 3 ways: raw daily mean / EMA-smoothed+scaled / EMA-smoothed+volatility+scaled
  4. Left-join each onto the cached technical-indicator grid (processed_full_noSA.csv) on (date, ticker)
        ↓
Local RL training  (no GPU needed — small networks, fast on CPU)
  5. Build StockTradingEnv per config, train A2C/PPO/TD3 (50K timesteps each)
  6. Backtest on 2023, compute Sharpe/return/drawdown vs. DJI
```

---

## 5. Results

### Single run per configuration (2023 backtest, $1M start)

| Config | Agent | Final value | Return | Sharpe |
|---|---|---:|---:|---:|
| NoSA | A2C | $1,508,228 | 50.8% | 1.67 |
| NoSA | PPO | $1,439,990 | 44.0% | 2.30 |
| NoSA | TD3 | $1,601,465 | 60.1% | 2.02 |
| RawSA | A2C | $1,681,882 | 68.2% | 2.23 |
| RawSA | PPO | $1,298,558 | 29.9% | 2.04 |
| RawSA | TD3 | $1,600,178 | 60.0% | 2.39 |
| OptSA | A2C | $1,480,485 | 48.0% | 1.68 |
| OptSA | PPO | $1,158,544 | 15.9% | 1.58 |
| OptSA | TD3 | $1,442,041 | 44.2% | 2.25 |
| OptSA_Vol | A2C | $1,500,860 | 50.1% | 2.01 |
| OptSA_Vol | PPO | $1,214,290 | 21.4% | 1.67 |
| OptSA_Vol | **TD3** | **$1,774,085** | **77.4%** | **2.53** |
| — | DJI (benchmark) | $1,138,028 | 13.8% | 1.21 |

### What actually holds up: 5-seed validation

A single run per config isn't enough to trust — RL has real run-to-run variance from initialization and
exploration noise alone. TD3 and A2C were each re-run 5× (different seeds), with and without the best
sentiment config (OptSA_Vol):

| Agent | Config | Sharpe (mean ± std) | Seeds improved | Paired t-test |
|---|---|---|---|---|
| TD3 | No sentiment | 2.10 ± 0.31 | — | — |
| TD3 | + Optimized sentiment & volatility | **2.29 ± 0.18** | **4 / 5** | one-tailed p = 0.085 |
| A2C | No sentiment | 2.08 ± 0.23 | — | — |
| A2C | + Optimized sentiment & volatility | 1.90 ± 0.47 | 1 / 5 | one-tailed p = 0.88 |

**Conclusion**: sentiment (specifically EMA-smoothed + volatility) gives TD3 a real, moderate, *not fully
statistically proven* edge (p=0.085, just short of the conventional 0.05 bar with only 5 seeds). It makes A2C
**worse**, reliably — the opposite of what the single-run table above would suggest, which is exactly why the
multi-seed check mattered.

### The benchmark that actually matters: buy-and-hold

Comparing only against the DJI index is misleading here — DJI is a diversified, industrial-heavy index with
no relationship to this specific, tech-heavy 10-stock portfolio. The fairer benchmark is doing nothing with
the *same* 10 stocks:

> **Equal-weight buy-and-hold, same 10 stocks, no trading, 2023: $1,882,283 (+88.2%)**
> — driven by NVDA (+246%), META (+187%), TSLA (+134%) that year.

That beats **every single trained agent**, including the best one (TD3 + OptSA_Vol, mean $1,699,573 / +70%
across the 5 validated seeds). The honest takeaway: the trading/timing decisions the agents learned were net
value-*destructive* relative to simply holding, in a year when a few of the underlying names had extraordinary
gains that active timing tends to partially miss.

---

## 6. Known limitations

- **META has zero sentiment coverage.** FNSPID tags older Facebook articles as `FB`; the streaming filter
  only matched literal `META`, so one of the ten stocks contributes flat/zero sentiment throughout.
- **One test year.** All results are a single chronological backtest on 2023 — a strong year for this
  specific stock basket. No rolling/walk-forward validation across multiple market regimes was done.
- **5 seeds is a minimum, not a generous sample**, for the statistical validation in §5.
- **PPO was not re-validated with multiple seeds** — it was the weakest agent in every single-run config and
  there's no reason to expect that direction to flip, but it wasn't formally re-tested.

---

## 7. Reproducing this

1. Run `Kaggle_0_SentimentScoring.ipynb` on Kaggle (GPU, `HF_TOKEN` secret attached) → download `news.csv` +
   `scored.csv`.
2. Locally: merge sentiment onto the cached technical-indicator grid (see the merge step in §4) to produce
   `processed_full.csv` / `processed_full_opt.csv` / `processed_full_filtered_scaled.csv` for each config.
3. Run each `Kaggle_N_*/Kaggle_N_*.ipynb` locally via `jupyter nbconvert --execute` — no GPU required, each
   completes in a few minutes on a standard laptop CPU.
