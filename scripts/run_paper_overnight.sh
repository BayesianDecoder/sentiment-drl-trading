#!/bin/bash
# Runs TSLA then BTC sequentially (avoids resource contention from running both at once),
# writing each asset's log separately. Wrapped in `caffeinate` by the caller so it survives
# lid-close / display sleep overnight.
set -e
cd /Users/vijay/Desktop/sentiment-drl-trading

echo "=== TSLA starting $(date) ==="
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -u -m src.paper_replication.train_paper --asset TSLA --agents PG PPO DQL DDPG > /tmp/paper_tsla/run.log 2>&1
echo "=== TSLA done $(date) ==="

echo "=== BTC starting $(date) ==="
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -u -m src.paper_replication.train_paper --asset BTC --agents PG PPO DQL DDPG > /tmp/paper_btc/run.log 2>&1
echo "=== BTC done $(date) ==="

echo "=== ALL DONE $(date) ==="
