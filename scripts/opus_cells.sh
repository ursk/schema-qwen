#!/bin/bash
# Opus joins the bake-off: waits for the local field to finish, then plays the
# same two 30-min cells (coached "easy" + plain original) via headless CC.
# Needs no :8084 backend — runs after teardown restored the default.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/bakeoff.log
exec >>"$LOG" 2>&1

until grep -q "BAKE-OFF done" "$LOG"; do sleep 60; done
echo "=== OPUS CELLS start $(date) ==="
MINUTES=${BAKE_MINUTES:-30}

for arm in coached plain; do
  name="bake-ccopus-$arm"
  echo "--- RUN $name ($(date)) — $MINUTES min ---"
  rm -rf "runs/ls20-$name"
  flag=""
  [ "$arm" = "coached" ] && flag="--coached"
  lsof -iTCP:8123 -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill 2>/dev/null
  sleep 1
  nohup $PY -m harness.viewer --run "runs/ls20-$name" --port 8123 >/dev/null 2>&1 &
  gtimeout $((MINUTES * 60)) $PY -m harness.run --game ls20 --model cc:opus \
    --samples 1 --max-tokens 8192 $flag --run-name "$name"
  echo "--- $name done rc=$? ($(date)) ---"
  pkill -f "harness.run --game ls20" 2>/dev/null
  sleep 5
done
echo "=== OPUS CELLS done $(date) ==="
