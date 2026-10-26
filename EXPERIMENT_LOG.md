# Experiment Log — Project 1 (10-stock Indian portfolio RL)

Plain-language record of every experiment: what was tried, why, what happened, and what we
decided as a result. Written *as we go*, not reconstructed afterward — the reason this file
exists is that a prior "v2" experiment (see below) left no surviving numbers anywhere, which
made it impossible to learn from later. Every entry here should have real numbers attached
before we move on from it.

---

## Baseline — v1 (current dashboard-backing results)

**What it is**: multi-asset continuous-action portfolio env (`src/rl/trading_env.py`), 10 Indian
stocks, raw portfolio log-return reward (not benchmark-relative), technical indicators MACD/
RSI-14/CCI-20/ADX-14 (not price-normalized), real Qwen sentiment (bootstrap-resampled for
train/val, sparse-real for test). 5 agents: PPO, A2C, DDPG, SAC, TD3.

**Result** (`results/rl/final_all_results.csv`, MLflow experiment `DRLTrading`):

| Agent | Variant | Return | Sharpe | Max DD | Beats Buy&Hold |
|---|---|---|---|---|---|
| SAC | sentiment | -3.12% | -0.014 | -25.98% | Yes (best) |
| DDPG | sentiment | -8.95% | -0.621 | -18.76% | Yes (narrow) |
| PPO | nosentiment | -1.14% | +0.035 | -17.26% | Yes |
| A2C | sentiment | -18.02% | -1.019 | -33.12% | No |
| TD3 | sentiment | -22.10% | -1.391 | -32.37% | No |
| PPO | sentiment | -23.16% | -1.037 | -40.50% | No |
| SAC | nosentiment | -23.42% | -1.410 | -32.48% | No |
| Buy&Hold | — | -9.46% | -0.822 | -18.49% | — |

**Verdict**: SAC-with-sentiment is the current best result. Sentiment helped SAC enormously
(-23.4% → -3.1%) but hurt DDPG/PPO relative to their own no-sentiment baselines — a real,
agent-dependent effect, not noise.

---

## "v2" — alpha-reward + richer features (found, NOT run by this session — historical, unverifiable)

**What we know**: `src/rl/config.py` carries a comment saying a "CLaC@FinMMEval-2026 alpha-reward"
experiment was tried and reverted for underperforming, prior to this session. Investigated on
2026-08-07: no metrics, logs, or results files survive anywhere for it. Only evidence is 5
orphaned SB3 model files in `mlruns/1/*/artifacts/` (`DDPG_v2_alpha.zip`, `PPO_v2_alpha.zip` x2,
`A2C_v2_alpha.zip` x2, dated 2025-07-24) with no accompanying params/metrics/tags.

Inspecting `DDPG_v2_alpha.zip` directly: action space `Box(10,)` (same multi-asset framing as
v1), observation space `Box(107,)` (vs. v1's 71) — consistent with 6 calendar cyclical features
+ cash (7 global) + 10 features/asset x 10 assets, i.e. something architecturally close to what
this session was about to build fresh (normalized indicators + calendar encodings + alpha
reward). Timesteps=200,000, seed=42, matching current `config.py` off-policy budget.

**Verdict**: real signal that this specific combination was tried before and abandoned, but the
magnitude/fairness of "underperformed" is **unverifiable** — no numbers survive, and we can't
rule out a bug or unfair comparison in that attempt. Could not reload the old models for a fresh
backtest because the exact 107-dim feature ordering isn't recoverable without its source code.
Decision: proceed with a fresh, carefully-built, **fully logged** attempt rather than treat an
undocumented result as final.

---

## v3 — full comparison vs. v1 baseline (2026-08-07)

All 5 agents finished. Buy&Hold benchmark on this test split: return=-9.46%, sharpe=-0.822
(identical to v1's, same dates/stocks, confirms apples-to-apples comparison).

| Agent | v1 Return | v1 Sharpe | v3 Return | v3 Sharpe | v3 Max DD | Verdict |
|---|---|---|---|---|---|---|
| PPO | -23.16% (sent) | -1.037 | -18.75% | -1.197 | -31.48% | Better return, worse Sharpe; still loses to benchmark |
| A2C | -18.02% | -1.019 | -12.84% | -0.931 | -26.29% | Improved on both, still loses to benchmark |
| DDPG | -8.95% | -0.621 | -8.97% | -0.679 | -16.09% | Essentially unchanged return/Sharpe; drawdown improved a lot (-18.76%→-16.09%) |
| **SAC** | **-3.12%** | **-0.014** | -8.37% | -0.506 | -25.01% | **Regression** — v1 remains clearly better |
| TD3 | -22.10% | -1.391 | -10.35% | -0.800 | -20.48% | Improved substantially on both, still loses to benchmark |

**Verdict**: v3 (alpha reward + normalized features + calendar encodings + cross-sectional
features + volatility regime + sentiment smoothing + random episode starts) does **not** beat
v1 overall. It improved TD3, A2C, and DDPG's drawdown meaningfully, but **regressed SAC** —
the single best result in the whole project — from beating the benchmark clearly to barely
beating it. v1's SAC-with-sentiment (-3.12%/-0.014) remains the best result found to date.

**Working hypothesis for the SAC regression** (not yet tested): v3's random episode starts cap
training episodes at a maximum length (`n_steps - MIN_DAYS_REMAINING_FOR_START + 1`, well short
of the full ~850-day training period), fragmenting the experience SAC's replay buffer sees.
SAC is off-policy and benefited most from v1's uninterrupted full-length episodes; the other
four agents (especially TD3, which improved the most) may be more robust to shorter, varied
episodes, or may have been actively hurt by v1's lack of that variety. This needs an ablation
(same v3 setup, SAC only, full-length training episodes) to confirm before concluding anything.

## SAC regression, fully decomposed (2026-08-07)

Three isolating ablations, holding SAC hyperparameters constant throughout:

| Variant | Features | Reward | Random starts | Return | Sharpe |
|---|---|---|---|---|---|
| v1 baseline | v1 (unnormalized) | raw portfolio return | No (full episodes) | **-3.12%** | **-0.014** |
| v3 full | v3 (normalized+calendar+cross-sectional+vol) | alpha | Yes | -8.37% | -0.506 |
| Ablation 1 | v3 | alpha | **No** (full episodes) | -11.96% | -0.775 |
| Ablation 2 | **v1 (unnormalized)** | alpha | No | -5.21% | -0.217 |

**Conclusion**: the new *feature set* is the dominant cause of the regression, not the alpha
reward. Swapping back to v1's original unnormalized features while keeping alpha reward
(Ablation 2) recovered most of the gap (-8.37%→-5.21%, -0.506→-0.217). Alpha reward alone still
costs SAC a smaller residual amount (-5.21%/-0.217 vs. v1's -3.12%/-0.014) — real but much
smaller than the feature-set effect. Random starts, ruled out earlier, made things *worse* when
removed (Ablation 1 is the worst of the three), confirming they weren't the problem either.

**Practical reading**: v3's richer 118-dim feature set (vs. v1's 71-dim) likely gives SAC more
to learn from than its training budget can extract signal from -- more noisy/redundant
dimensions (cross-sectional return, volatility regime, sentiment smoothing, 6 calendar
features) without a corresponding increase in training data or timesteps. TD3, A2C, and DDPG
improved under the same richer features, suggesting this is an interaction between the new
features and SAC's specific learning dynamics (entropy-regularized off-policy), not a flaw in
the features themselves. v1's SAC-with-sentiment (-3.12%/-0.014) remains the best result in the
project. Next: consider a feature-reduced v3 variant (alpha reward + calendar/normalized
indicators only, dropping cross-sectional/vol-regime/sentiment-smoothing) to see if a lighter
version recovers v1's SAC performance while keeping TD3/A2C/DDPG's gains.

## (next entries added below as work happens)

## v3 — PPO_v3 (2026-08-07 00:39)

**Config**: `{'learning_rate': 0.0003, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 10, 'gamma': 0.95, 'gae_lambda': 0.9, 'clip_range': 0.3, 'ent_coef': 0.001}`
**Best validation Sharpe during search/training**: 1.758
**Test result**: return=-18.75% | sharpe=-1.197 | max_dd=-31.48% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 279s

## v3 — A2C_v3 (2026-08-07 00:43)

**Config**: `{'learning_rate': 0.0007, 'n_steps': 5, 'gamma': 0.95, 'gae_lambda': 0.95, 'ent_coef': 0.0, 'vf_coef': 0.25}`
**Best validation Sharpe during search/training**: 1.328
**Test result**: return=-12.84% | sharpe=-0.931 | max_dd=-26.29% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 205s

## v3 — DDPG_v3 (2026-08-07 01:20)

**Config**: `{'learning_rate': 0.003, 'batch_size': 256, 'tau': 0.001, 'gamma': 0.995, 'action_noise_sigma': 0.2}`
**Best validation Sharpe during search/training**: 2.381
**Test result**: return=-8.97% | sharpe=-0.679 | max_dd=-16.09% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 2242s

## v3 — SAC_v3 (2026-08-07 02:34)

**Config**: `{'learning_rate': 0.0001, 'batch_size': 512, 'tau': 0.001, 'gamma': 0.99}`
**Best validation Sharpe during search/training**: 1.326
**Test result**: return=-8.37% | sharpe=-0.506 | max_dd=-25.01% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 4460s

## v3 — TD3_v3 (2026-08-07 03:19)

**Config**: `{'learning_rate': 0.003, 'batch_size': 256, 'tau': 0.001, 'gamma': 0.99, 'policy_delay': 2, 'action_noise_sigma': 0.3}`
**Best validation Sharpe during search/training**: 1.305
**Test result**: return=-10.35% | sharpe=-0.800 | max_dd=-20.48% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 2695s

## v3 — SAC_v3_ablation_fullep (2026-08-07 04:45)

**Config**: `{'learning_rate': 0.0001, 'batch_size': 512, 'tau': 0.001, 'gamma': 0.99, 'random_start': False}`
**Best validation Sharpe during search/training**: 1.545
**Test result**: return=-11.96% | sharpe=-0.775 | max_dd=-19.45% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 2264s

## v3 — SAC_ablation_alphareward_v1features (2026-08-07 14:55)

**Config**: `{'learning_rate': 0.0001, 'batch_size': 512, 'tau': 0.001, 'gamma': 0.99, 'features': 'v1 (unnormalized)', 'reward': 'alpha'}`
**Best validation Sharpe during search/training**: 1.657
**Test result**: return=-5.21% | sharpe=-0.217 | max_dd=-22.28% | benchmark_return=-9.46%
**Beats benchmark**: False
**Training time**: 36522s
