"""Outer loop: observe -> deliberate -> execute -> record, until win or budget.

Usage:
  .venv/bin/python -m harness.run --game ls20 [--model qwen36-lens]
      [--base-url http://127.0.0.1:8086/v1] [--max-actions 1500]
      [--max-deliberations 200]
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import Agent, LLM
from .envio import BudgetExceeded, Env
from .timeline import Timeline

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--model", default="qwen36-lens")
    ap.add_argument("--base-url", default="http://127.0.0.1:8086/v1")
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--max-deliberations", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=4,
                    help="candidate replies sampled per turn (best-of-N via backtest)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    stamp = args.run_name or datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{args.game}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"

    def log(kind, data):
        with open(events_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "data": data}) + "\n")

    timeline = Timeline(run_dir / "timeline.jsonl")
    env = Env(args.game, timeline, max_actions=args.max_actions)
    llm = LLM(args.base_url, args.model, max_tokens=args.max_tokens)
    agent = Agent(env, timeline, llm, run_dir, log, samples=args.samples)

    env.reset()
    print(f"[{args.game}] started · {env.win_levels} levels · run dir {run_dir}")
    log("start", {"game": args.game, "model": args.model, "win_levels": env.win_levels})

    extra = ""
    for d in range(args.max_deliberations):
        try:
            result = agent.deliberate(extra)
        except BudgetExceeded as e:
            print(f"STOP: {e}")
            log("stop", {"reason": str(e)})
            break
        if env.state == "WIN":
            print(f"WIN after {timeline.action_count} actions, {llm.calls} llm calls")
            log("win", {"actions": timeline.action_count, "llm_calls": llm.calls})
            break
        if env.state == "GAME_OVER":
            env.act("RESET")
            extra = "The game hit GAME_OVER and was RESET. The level restarted."
        elif result is None:
            extra = ("Your previous deliberation ended without any COMMIT. You MUST take real "
                     "actions to learn. Commit a short probe now.")
        else:
            extra = ""
        print(f"[deliberation {d}] level {env.level}/{env.win_levels} · "
              f"{timeline.action_count} actions · {llm.calls} llm calls")
        log("progress", {"deliberation": d, "level": env.level,
                         "actions": timeline.action_count, "llm_calls": llm.calls})
    else:
        print("STOP: deliberation budget exhausted")
        log("stop", {"reason": "max deliberations"})


if __name__ == "__main__":
    main()
