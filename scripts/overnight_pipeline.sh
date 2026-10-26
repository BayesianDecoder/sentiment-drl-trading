#!/bin/bash
# Runs unattended: rebuild the sentiment panel with the fine-tuned Qwen2.5-3B model (MLX, local,
# no download needed), retrain all 5 RL agents with the full two-stage HPO process, then push
# results to the dashboard. Each step logs to its own file and the whole run stops on the first
# real failure rather than silently continuing on bad data.
set -e
cd /Users/vijay/Desktop/sentiment-drl-trading
LOG_DIR=/tmp/overnight_pipeline
mkdir -p "$LOG_DIR"

echo "[$(date)] STEP 1/4: rebuilding rl_panel.csv with Qwen2.5-3B sentiment..." | tee -a "$LOG_DIR/progress.log"
.venv/bin/python -m src.rl.data_pipeline > "$LOG_DIR/01_data_pipeline.log" 2>&1
echo "[$(date)] STEP 1/4 done" | tee -a "$LOG_DIR/progress.log"

echo "[$(date)] STEP 2/4: retraining PPO/A2C/DDPG/SAC/TD3 (full HPO, with sentiment)..." | tee -a "$LOG_DIR/progress.log"
for agent in PPO A2C DDPG SAC TD3; do
  echo "[$(date)]   training $agent..." | tee -a "$LOG_DIR/progress.log"
  .venv/bin/python -m src.rl.train_v1 --agents $agent > "$LOG_DIR/02_train_${agent}.log" 2>&1
  echo "[$(date)]   $agent done" | tee -a "$LOG_DIR/progress.log"
done
echo "[$(date)] STEP 2/4 done" | tee -a "$LOG_DIR/progress.log"

echo "[$(date)] STEP 3/4: pushing results to Express + regenerating A2C strategy view..." | tee -a "$LOG_DIR/progress.log"
lsof -ti:4000 -sTCP:LISTEN | xargs -r kill 2>/dev/null || true
sleep 2
export PATH="/Users/vijay/Desktop/sentiment-drl-trading/.tools/node-v22.14.0-darwin-arm64/bin:$PATH"
cd backend/express_api
nohup node src/server.js > "$LOG_DIR/03_express.log" 2>&1 &
sleep 5
cd /Users/vijay/Desktop/sentiment-drl-trading
.venv/bin/python scripts/collect_rl_results.py > "$LOG_DIR/03_collect_results.log" 2>&1
.venv/bin/python scripts/generate_a2c_strategy.py > "$LOG_DIR/03_generate_strategy.log" 2>&1 || echo "[$(date)]   note: generate_a2c_strategy.py failed -- likely a different agent won this time, check 03_collect_results.log for the actual winner" | tee -a "$LOG_DIR/progress.log"
.venv/bin/python scripts/fetch_news_headlines.py > "$LOG_DIR/03_fetch_news.log" 2>&1 || true
lsof -ti:4000 -sTCP:LISTEN | xargs -r kill 2>/dev/null || true
sleep 2
cd backend/express_api
nohup node src/server.js > "$LOG_DIR/03_express_final.log" 2>&1 &
cd /Users/vijay/Desktop/sentiment-drl-trading
echo "[$(date)] STEP 3/4 done" | tee -a "$LOG_DIR/progress.log"

echo "[$(date)] STEP 4/4: starting FastAPI + frontend for morning review..." | tee -a "$LOG_DIR/progress.log"
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill 2>/dev/null || true
sleep 2
cd backend/fastapi_ml
nohup ../../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > "$LOG_DIR/04_fastapi.log" 2>&1 &
cd /Users/vijay/Desktop/sentiment-drl-trading/frontend
lsof -ti:5173 -sTCP:LISTEN | xargs -r kill 2>/dev/null || true
nohup npm run dev > "$LOG_DIR/04_frontend.log" 2>&1 &
cd /Users/vijay/Desktop/sentiment-drl-trading

echo "[$(date)] ALL DONE -- see $LOG_DIR/progress.log and results/rl/ for final numbers" | tee -a "$LOG_DIR/progress.log"
