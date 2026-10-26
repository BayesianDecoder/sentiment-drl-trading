# Sentiment-Augmented DRL Stock Trading

Two things built end-to-end: a financial sentiment model fine-tuned from a 3B LLM, and RL trading agents that use those scores to trade a US stock portfolio. A dashboard connects both into a live visualization.

The core question: does knowing whether the news is positive or negative actually help a trading agent make better decisions?

---

## Part A — Sentiment Model

### What was built

A full progression from the simplest possible baseline to a fine-tuned large language model, all trained on **SEntFiN-v1.1** and tracked across **877 MLflow runs**.

The dataset was restructured for the task: instead of collapsing multi-company headlines to a single label, each headline was exploded into `(headline, company, label)` triples — 14,409 triples total. This matches how sentiment actually gets used downstream (per ticker, not per article) and was a genuine methodological improvement over the original paper.

### Results

| Stage | Model | Accuracy | F1 (macro) |
|---|---|---:|---:|
| Classical ML | Logistic Regression (TF-IDF, tuned) | 81.3% | 0.813 |
| Classical ML | Linear SVM (tuned) | 81.1% | 0.811 |
| Classical ML | RF / LightGBM / XGBoost | 78–79% | — |
| From-scratch DL | TextCNN | 76.4% | 0.764 |
| From-scratch DL | BiLSTM | 77.3% | 0.773 |
| From-scratch DL | CNN-BiLSTM-Attention | 76.5% | 0.764 |
| From-scratch DL | Transformer | 77.7% | 0.777 |
| Pretrained + PEFT | FinBERT + LoRA (r=16) | 86.5% | 0.865 |
| Pretrained + PEFT | FinBERT + QLoRA (r=8, 4-bit) | 86.3% | 0.863 |
| Pretrained + PEFT | FinBERT + Prefix Tuning | 70.8% | 0.706 |
| **Final** | **Qwen2.5-3B-Instruct + LoRA (HP-tuned)** | **90.4%** | **0.905** |

**What the numbers say**: from-scratch deep learning never beat classical ML — the dataset is too small. Starting from a pretrained model (FinBERT) was worth +5 points over the best classical baseline. Scaling to a larger pretrained model (Qwen 3B) added another +4. Model scale and pretraining quality mattered far more than architecture choices.

### The final model

The Qwen adapter (LoRA rank=32, α=128, all attention + MLP projections, 1.9% trainable params) is hosted at [`vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned`](https://huggingface.co/vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned).

At inference it does a single forward pass and reads `P(positive) − P(negative)` at the three token positions — a continuous score in `[-1, 1]` instead of a hard label. More information-dense as a state feature for the RL environment.

---

## Part B — RL Trading

### Setup

Replication and extension of [hab5510/finrl_project](https://github.com/hab5510/finrl_project). The original used FinBERT for sentiment scoring — this project swaps it for the fine-tuned Qwen adapter.

- **Framework**: FinRL + Stable-Baselines3
- **Agents**: A2C, PPO, TD3
- **Universe**: 10 US stocks — AAPL, MSFT, NVDA, META, AMZN, GOOGL, JPM, TSLA, BA, DIS
- **Portfolio**: $1,000,000 starting capital
- **Train**: 2020-01-01 → 2022-12-31
- **Test (backtest)**: all of 2023
- **Technical indicators**: MACD, Bollinger Bands, RSI(30), CCI(30), DX(30), SMA(30), SMA(60), VIX, turbulence index

### 4 sentiment configurations

| Config | Description |
|---|---|
| No SA | Price + technical indicators only (baseline) |
| Raw SA | Daily mean Qwen score per stock, no preprocessing |
| Optimized SA | Filtered (≥2 headlines/day), EMA span=3, StandardScaler |
| Optimized SA + Volatility | Same + rolling 3-day sentiment std dev as a second feature |

### Results (single run, 2023 backtest)

| Config | Agent | Final value | Return | Sharpe |
|---|---|---:|---:|---:|
| No SA | TD3 | $1,601,465 | +60.1% | 2.02 |
| Raw SA | A2C | $1,681,882 | +68.2% | 2.23 |
| Optimized SA | TD3 | $1,442,041 | +44.2% | 2.25 |
| **Opt SA + Volatility** | **TD3** | **$1,774,085** | **+77.4%** | **2.53** |
| Buy & hold (same 10 stocks) | — | $1,882,283 | +88.2% | — |

### 5-seed validation

Single runs aren't enough to trust — RL has real run-to-run variance. TD3 and A2C were each re-run 5 times with and without the best sentiment config:

| Agent | Config | Sharpe mean ± std | Seeds improved | p-value |
|---|---|---|---|---|
| TD3 | No sentiment | 2.10 ± 0.31 | — | — |
| **TD3** | **Opt SA + Volatility** | **2.29 ± 0.18** | **4 / 5** | **0.085** |
| A2C | No sentiment | 2.08 ± 0.23 | — | — |
| A2C | Opt SA + Volatility | 1.90 ± 0.47 | 1 / 5 | 0.88 |

**Honest findings**:
- Sentiment gives TD3 a consistent, moderate edge (tighter variance, higher mean) — doesn't quite clear p=0.05 at 5 seeds but the direction is real
- It made A2C reliably *worse* — the single-run gain was noise
- Buy-and-hold beat every agent (NVDA +246%, META +187%, TSLA +134% in 2023 — passive holding wins when the basket has outliers like that)

---

## Part C — Dashboard

Three services that connect Parts A and B into a live UI.

| Service | Port | What it does |
|---|---|---|
| FastAPI (`backend/fastapi_ml/`) | 8000 | Loads the trained FinBERT-LoRA adapter, scores headlines on request |
| Express + MongoDB (`backend/express_api/`) | 4000 | Serves RL model metrics, A2C strategy timeline, news feed. Uses `mongodb-memory-server` — no system MongoDB install needed |
| React / Vite (`frontend/`) | 5173 | The UI |

**What you see on the dashboard**:
- **Sidebar**: all 10 stocks with price sparklines and A2C's latest buy / hold / sell action
- **Chart**: A2C portfolio value vs buy-and-hold over the full test period. Hover any point to see exactly which stocks the agent was holding that day
- **News feed**: real headlines scored live by FinBERT-LoRA, each matched to A2C's action on that stock around that date

---

## Project structure

```
sentiment-drl-trading/
│
├── finrl_project-main/
│   ├── README.md                # Original paper (Hana Ben Slama, THI)
│   ├── report/                  # PDF report + comparison charts
│   └── modified/                # Part B — this project's RL work
│       ├── Kaggle_0_SentimentScoring.ipynb   # Score ~60K FNSPID headlines with Qwen
│       ├── Kaggle_1_NoSA/                    # No sentiment config + results
│       ├── Kaggle_2_RawSA/                   # Raw sentiment config + results
│       ├── Kaggle_3_OptSA/                   # Optimized sentiment config + results
│       ├── Kaggle_4_OptSA_Vol/               # Opt + volatility config + results
│       ├── multiseed_ablation_results.csv    # Raw 5-seed data
│       ├── multiseed_ablation_summary.csv    # Aggregated per-agent summary
│       ├── news.csv                          # ~60K matched FNSPID headlines
│       ├── scored.csv                        # Same, with Qwen sentiment scores
│       └── README_drl.md                     # Full method write-up + all numbers
│
├── notebooks/                   # Part A — sentiment model pipeline
│   ├── data_eda.ipynb           # EDA, cleaning, train/val/test split
│   ├── bow_vs_tfidf.ipynb       # BoW vs TF-IDF feature sweep
│   ├── logistic_regression_with_hpt.ipynb
│   ├── svm_with_hpt.ipynb
│   ├── lightgbm_with_hpt.ipynb
│   ├── xgboost_with_hpt.ipynb
│   ├── knn_with_hpt.ipynb
│   ├── random_forest_with_hpt.ipynb
│   ├── 03_textcnn.ipynb                     # Deep learning from scratch
│   ├── 04_bilstm.ipynb
│   ├── 05_cnn_bilstm_attention.ipynb
│   ├── 06_transformer_scratch.ipynb
│   ├── 07_finbert_peft.ipynb                # FinBERT + LoRA / QLoRA / Prefix Tuning
│   ├── 13_llm_lora_qlora_sentiment.ipynb    # LLM zero-shot benchmark
│   └── 14_qwen_sentiment_test.ipynb         # Qwen adapter inference test
│
├── finetune_llm/
│   ├── qwen_final_with_hp.ipynb    # Qwen2.5-3B LoRA HP-tuned training (Kaggle, GPU)
│   └── download.ipynb              # Local inference + MLX adapter utilities
│
├── backend/
│   ├── fastapi_ml/                 # Part C — live sentiment inference API
│   └── express_api/                # Part C — RL data + news API
│
├── frontend/                       # Part C — React/Vite dashboard
│
├── scripts/
│   ├── collect_rl_results.py       # Push backtest results to dashboard DB
│   ├── generate_a2c_strategy.py    # Generate per-day A2C strategy timeline
│   └── fetch_news_headlines.py     # Fetch + score live headlines
│
├── data/
│   └── raw/SEntFiN-v1.1.csv       # Source dataset (download separately)
│
├── mlflow.db / mlruns/             # 877 logged experiment runs
├── start_dashboard.sh
├── stop_dashboard.sh
├── requirements.txt
└── setup.sh
```

---

## How to run it

### Setup

```bash
git clone https://github.com/BayesianDecoder/sentiment-drl-trading
cd sentiment-drl-trading
bash setup.sh
```

`setup.sh` creates a `.venv` and installs everything from `requirements.txt`. Node.js is bundled at `.tools/` — no system install needed.

---

### Part A — Sentiment model

**Step 1** — EDA + splits (run this first, everything else reads the CSVs it saves):

```bash
jupyter notebook notebooks/data_eda.ipynb
```

**Step 2** — Classical ML (all local, a few minutes each):

```bash
jupyter notebook notebooks/logistic_regression_with_hpt.ipynb
jupyter notebook notebooks/svm_with_hpt.ipynb
jupyter notebook notebooks/lightgbm_with_hpt.ipynb
jupyter notebook notebooks/xgboost_with_hpt.ipynb
jupyter notebook notebooks/random_forest_with_hpt.ipynb
```

**Step 3** — Deep learning + FinBERT (Kaggle recommended, GPU):

Run in order: `03_textcnn.ipynb` → `04_bilstm.ipynb` → `05_cnn_bilstm_attention.ipynb` → `06_transformer_scratch.ipynb` → `07_finbert_peft.ipynb`

**Step 4** — View all runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# open http://localhost:5000
```

**Qwen fine-tuning** (optional — adapter already trained):

Run `finetune_llm/qwen_final_with_hp.ipynb` on Kaggle (T4 GPU or better). Add `HF_TOKEN` as a Kaggle secret. The notebook covers the full HP search and final training run.

---

### Part B — RL trading

**Step 1** — Score the headlines (Kaggle, GPU, run once):

Open `finrl_project-main/modified/Kaggle_0_SentimentScoring.ipynb` on Kaggle. Add `HF_TOKEN` as a secret. Run it — it streams ~60K FNSPID headlines, scores each with the Qwen adapter, and saves `news.csv` and `scored.csv`. Download both and place them in `finrl_project-main/modified/`.

> **Kaggle fix**: if FinRL import errors appear, the notebook patches `finrl/__init__.py` to remove eager imports of `finrl.test`, `finrl.trade` and `finrl.train`, clears stale `sys.modules` entries, and mocks `wrds`, `ccxt` and `jqdatasdk`. This is already included in `Kaggle_0`.

**Step 2** — Run the 4 configs locally (no GPU needed, a few minutes each):

```bash
cd finrl_project-main/modified

jupyter nbconvert --to notebook --execute Kaggle_1_NoSA/Kaggle_1_NoSA.ipynb
jupyter nbconvert --to notebook --execute Kaggle_2_RawSA/Kaggle_2_RawSA.ipynb
jupyter nbconvert --to notebook --execute Kaggle_3_OptSA/Kaggle_3_OptSA.ipynb
jupyter nbconvert --to notebook --execute Kaggle_4_OptSA_Vol/Kaggle_4_OptSA_Vol.ipynb
```

Results land in each folder's `result.csv` and `evaluation_metrics_table.csv`. Full method and all numbers in `finrl_project-main/modified/README_drl.md`.

---

### Part C — Dashboard

```bash
# Generate the data the dashboard needs
.venv/bin/python scripts/collect_rl_results.py
.venv/bin/python scripts/generate_a2c_strategy.py
.venv/bin/python scripts/fetch_news_headlines.py

# Start all three services
./start_dashboard.sh
# open http://localhost:5173

# Stop
./stop_dashboard.sh
```

To refresh dashboard data after rerunning the RL configs, re-run the three scripts above and restart with `./stop_dashboard.sh && ./start_dashboard.sh`.

---

## Data

**SEntFiN-v1.1** — download from [Mendeley Data](https://data.mendeley.com/datasets/b7fghyxvcs/3), place at `data/raw/SEntFiN-v1.1.csv`. 10,753 Indian financial headlines with per-entity labels. Headlines where multiple entities disagree on sentiment (~11.5%) are dropped, leaving 9,459 clean samples, exploded to 14,409 triples for training.

**FNSPID** — streamed directly from Hugging Face during `Kaggle_0`, never fully downloaded. Only the ~60K rows matching the 10 tickers are kept. Pre-scored output is already at `finrl_project-main/modified/scored.csv`.

---

## Limitations

- 5 seeds is a minimal sample — TD3's p=0.085 doesn't clear the conventional bar
- A2C's single-run gain from sentiment doesn't survive multi-seed testing
- META has zero sentiment coverage throughout (FNSPID uses the old `FB` ticker tag)
- 2023 was an outlier year for this basket — hard to beat buy-and-hold when several names double or triple
- All results are a single chronological backtest window, no rolling/walk-forward validation
