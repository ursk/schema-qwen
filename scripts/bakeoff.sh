#!/bin/bash
# The Schema Bake-off: 4 local models x 2 harness arms x 30 min wall clock.
# Fresh run each cell, samples=1 everywhere (fair across serialized engines),
# game ls20, arms: coached ("easy") first, then plain (original).
# Detached via nohup — the Claude session only observes (harness bg kills).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/bakeoff.log
exec >>"$LOG" 2>&1
echo "=== BAKE-OFF start $(date) ==="

MODELS="qwen36 gemma26 mistral4 gptoss120"
MINUTES=${BAKE_MINUTES:-30}

restart_viewer() {
  local run_dir="$1"
  lsof -iTCP:8123 -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill 2>/dev/null
  sleep 1
  nohup $PY -m harness.viewer --run "$run_dir" --port 8123 >/dev/null 2>&1 &
}

for m in $MODELS; do
  echo "--- switching backend to $m ($(date)) ---"
  bash ~/code/housekeeping/switch-backend.sh "$m" || { echo "switch to $m FAILED — skipping"; continue; }
  ep=$(cat ~/code/housekeeping/.active-endpoint 2>/dev/null || echo native)
  if [ "$ep" = "bridge" ]; then URL=http://127.0.0.1:8086/v1; else URL=http://127.0.0.1:8084/v1; fi
  # wait until the model answers a tiny completion (max 8 min for big loads)
  ok=0
  for i in $(seq 1 96); do
    if curl -s -m 30 "$URL/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":8}" \
        | grep -q '"content"'; then ok=1; break; fi
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "$m never became ready — skipping"; continue; }

  for arm in coached plain; do
    name="bake-$m-$arm"
    echo "--- RUN $name ($(date)) — $MINUTES min ---"
    rm -rf "runs/ls20-$name"
    flag=""
    [ "$arm" = "coached" ] && flag="--coached"
    restart_viewer "runs/ls20-$name" || true
    gtimeout $((MINUTES * 60)) $PY -m harness.run --game ls20 --model "$m" \
      --base-url "$URL" --samples 1 --max-tokens 6144 $flag --run-name "$name"
    rc=$?
    echo "--- $name done rc=$rc ($(date)) ---"
    pkill -f "harness.run --game ls20" 2>/dev/null
    sleep 5
  done
done

echo "=== teardown: restore default backend + moss heartbeat ($(date)) ==="
bash ~/code/housekeeping/switch-backend.sh default
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.moss.heartbeat.plist 2>/dev/null
bash ~/code/housekeeping/note.sh "schema-qwen bake-off complete; default backend + moss heartbeat restored"
echo "=== BAKE-OFF done $(date) ==="
