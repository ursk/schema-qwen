"""Run worker.py in a subprocess with a wall-clock timeout."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TIMEOUTS = {"backtest": 90, "bfs": 150, "predict": 60}


def run_worker(cmd, model_path, timeline_path, args=None):
    argv = [
        sys.executable, "-m", "harness.worker",
        cmd, str(model_path), str(timeline_path),
    ]
    if args is not None:
        argv.append(json.dumps(args))
    try:
        proc = subprocess.run(
            argv, cwd=ROOT, capture_output=True, text=True,
            timeout=TIMEOUTS.get(cmd, 90),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{cmd} timed out — your code is too slow or loops forever"}
    if proc.returncode != 0:
        return {"ok": False, "error": f"worker crashed: {proc.stderr[-1500:]}"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": f"worker produced no JSON: {proc.stdout[-500:]} {proc.stderr[-500:]}"}
