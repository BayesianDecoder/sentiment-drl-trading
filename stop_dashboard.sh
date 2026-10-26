#!/bin/bash
# Stops all 3 dashboard services started by start_dashboard.sh.
for port in 8000 4000 5173; do
  pid=$(lsof -ti:$port -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill "$pid" && echo "Stopped service on :$port (pid $pid)"
  else
    echo "Nothing listening on :$port"
  fi
done
