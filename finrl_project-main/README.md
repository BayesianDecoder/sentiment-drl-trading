# Enhancing Deep Reinforcement Learning for Stock Trading Using Financial News Sentiment and Volatility


---

## Overview

This project investigates whether integrating financial news sentiment — including both raw scores and volatility — into deep reinforcement learning (DRL) agents can improve trading performance.

We evaluate the impact of structured sentiment signals on three DRL agents:
- **A2C** (Advantage Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **TD3** (Twin Delayed DDPG)

Using four different experimental setups, we compare agent performance across:
- Raw price and technical data (baseline)
- Added raw sentiment
- Optimized sentiment (EMA, normalization)
- Optimized sentiment + volatility

---

## Research Question

> *Can the integration of news sentiment and sentiment volatility into RL agent state representations lead to improved profitability and risk-adjusted performance?*

---

## Dataset & Tools

- **Market Data**: Yahoo Finance (2020–2023)
- **News Data**: [fnspid dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset.git)
- **Sentiment Scoring**: [FinBERT](https://arxiv.org/abs/2006.08097)
- **Processing**: Filtered by headline count, 3-day EMA smoothing, standard deviation (volatility)
- **Framework**: [FinRL](https://github.com/AI4Finance-Foundation/FinRL), Google Colab
- **Initial Portfolio**: $1,000,000  
- **Training Time**: TD3 (~20 mins), A2C/PPO (~2 mins)

---

## Experimental Setups

| Experiment | Sentiment Input | Preprocessing |
|-----------|------------------|---------------|
| 1         | No sentiment   | —             |
| 2         | Raw sentiment  | None          |
| 3         | Sentiment      | Filtered, EMA, normalized |
| 4         | Sentiment + Volatility | Filtered, EMA, normalized, Std Dev |

---

## Results Summary

### Table 1 – Final Portfolio Value (USD)

| Model | No SA | Raw SA | Optimized SA | Opt SA + Volatility |
|-------|-------|--------|--------------|----------------------|
| A2C   | $1,528,590 | $2,040,000 | $1,640,656 | $1,554,927 |
| PPO   | $1,196,072 | $1,344,424 | $1,172,948 | $996,754 |
| TD3   | $1,457,936 | $1,730,000 | **$1,791,986** | $1,384,668 |
| DJI   | $1,138,028 | $1,138,028 | $1,138,028 | $1,138,028 |

### Table 2 – Cumulative Return / Sharpe Ratio

| Agent | No SA | Raw SA | Opt SA | Opt SA + Vol |
|-------|-------|--------|--------|--------------|
| A2C   | 0.53 / 1.52 | 0.67 / 2.38 | 0.64 / 2.36 | 0.55 / 1.71 |
| PPO   | 0.20 / 1.42 | 0.34 / 1.56 | 0.17 / 1.69 | 0.00 / –0.11 |
| TD3   | 0.46 / 1.69 | 0.72 / 2.42 | **0.79 / 2.82** | 0.38 / 1.88 |
| DJI   | 0.14 / 1.20 | 0.14 / 1.20 | 0.14 / 1.20 | 0.14 / 1.20 |

---

## Key Takeaways

- **TD3** showed the most robust performance, especially with optimized sentiment.
- **A2C** also benefited from sentiment, though less from volatility.
- **PPO** was unstable with complex or noisy features.
- **Sentiment volatility** introduced noise and degraded performance across all agents.

---

## Repository Contents


```
README.md

/without_sentiment/
    FinRL_without_SentimentAnalysis.ipynb  
    profile_noSA.png  
    evaluation_metrics_noSA.png  
    result_noSA.csv  
    evaluation_metrics_table_noSA.csv  
    processed_full_noSA.csv  

/with_sentiment/
    FinRL_with_SentimentAnalysis.ipynb  
    profile.png  
    metrics.png  
    result.csv  
    evaluation_metrics_table.csv  
    processed_full.csv  

/with_sentiment_optimized/
    FinRL_with_SentimentAnalysis_optimized.ipynb  
    profile.png  
    evaluation_metrics_comaprison.png  
    result.csv  
    evaluation_metrics_table.csv  
    processed_full_smoothing_scaling.csv  

/with_sentiment_optimized_volatility/
    FinRL_with_SentimentAnalysis_optimized_volatility.ipynb  
    profile.png  
    evaluation_metrics.png  
    result.csv  
    evaluation_metrics_table.csv  
    processed_full_smoothing_scaling_volatility.csv  

/report/
    FinRL_SentimentAnalysis.pdf 
    cumulative_return_sharpe_ratio_all.png  
    cumulative_return_sharpe_ratio_all.csv  
    profile_value_all.csv  
```

---

## Full Report

  [FinRL_SentimentAnalysis.pdf](./report/FinRL_SentimentAnalysis.pdf)

---

## References

- Tetlock, P. C. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market. *Journal of Finance*, 62(3), 1139–1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x

- Bollen, J., Mao, H., & Zeng, X. (2011). *Twitter mood predicts the stock market*. Journal of Computational Science, 2(1), 1–8. https://doi.org/10.1016/j.jocs.2010.12.007
- Huang, B., et al. (2020). “FinBERT: A Pretrained Language Model for Financial Communications.” Hugging Face: https://huggingface.co/yiyanghkust/finbert-tone  
- FinRL GitHub: https://github.com/AI4Finance-Foundation/FinRL
- FNSPID https://huggingface.co/papers/2402.06698
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. https://stable-baselines3.readthedocs.io/
- Ye, H., & Zhang, Z. (2020). “Deep Reinforcement Learning for Automated Stock Trading.” 
- Yahoo Finance API. Historical Market Data. https://finance.yahoo.com  
- pandas Documentation – ewm: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

---

