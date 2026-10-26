#!/bin/bash
# Starts all 3 dashboard services: FastAPI ML service (8000), Express+MongoDB API (4000),
# React dev server (5173). Run from the project root: ./start_dashboard.sh
# Stop everything with: ./stop_dashboard.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_BIN="$ROOT/.tools/node-v22.14.0-darwin-arm64/bin"
export PATH="$NODE_BIN:$PATH"

mkdir -p /tmp/sentiment-drl-logs

echo "Starting FastAPI ML service on :8000..."
(cd "$ROOT/backend/fastapi_ml" && nohup "$ROOT/.venv/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 \
  > /tmp/sentiment-drl-logs/fastapi_ml.log 2>&1 &)

echo "Starting Express + MongoDB API on :4000..."
(cd "$ROOT/backend/express_api" && nohup node src/server.js \
  > /tmp/sentiment-drl-logs/express_api.log 2>&1 &)

echo "Starting React dashboard on :5173..."
(cd "$ROOT/frontend" && nohup npm run dev -- --host \
  > /tmp/sentiment-drl-logs/frontend.log 2>&1 &)

sleep 3
echo ""
echo "All 3 services launching. Logs in /tmp/sentiment-drl-logs/"
echo "  - FastAPI ML service:  http://localhost:8000/health"
echo "  - Express API:         http://localhost:4000/api/health"
echo "  - Dashboard:           http://localhost:5173"
echo ""
echo "Give the FastAPI service ~10-15s to load the FinBERT model before using the Live Sentiment Demo tab."
