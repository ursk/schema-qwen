"""Outer loop: observe -> deliberate -> execute -> record, until win or budget.

Usage:
  .venv/bin/python -m harness.run --game ls20 [--model qwen36-lens]
      [--base-url http://127.0.0.1:8086/v1] [--max-actions 1500]
      [--max-deliberations 200]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import Agent, ClaudeCLI, HumanCLI, LLM
from .envio import BudgetExceeded, Env
from .notify import abort_run
from .timeline import Timeline

ROOT = Path(__file__).resolve().parent.parent

# Fail-loud guard: this many consecutive deliberations without a single
# committed action aborts the run (Telegram + ABORTED.md via notify).
# Sized against real work: healthy runs show isolated single no-commit
# turns; three in a row has only ever meant a degenerate loop (vis2
# 2026-07-19: three straight turns burning the full token budget on
# repetition, zero moves).
MAX_NONE_STREAK = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--model", default="qwen36-lens")
    ap.add_argument("--base-url", default="http://127.0.0.1:8086/v1")
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--max-deliberations", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--samples", type=int, default=4,
                    help="candidate replies sampled per turn (best-of-N via backtest)")
    ap.add_argument("--stop-at-level", type=int, default=None,
                    help="stop the run once this many levels are cleared")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--coached", action="store_true",
                    help="use the genre-coached system prompt (A/B vs the plain one)")
    ap.add_argument("--vision", action="store_true",
                    help="sighted harness: object decomposition of the grid + "
                         "optic-flow transition rendering (documented deviation)")
    ap.add_argument("--from-ckpt", default=None,
                    help="seed the run dir from a named checkpoint (any model "
                         "may continue any player's checkpoint)")
    ap.add_argument("--resume", action="store_true",
                    help="deliberately continue an existing run dir, carrying "
                         "over world_model/notes/timeline. Without this flag a "
                         "non-empty run dir is an error: fresh start is the "
                         "default (carried-over state can poison a new/fixed "
                         "model and dirties comparisons).")
    args = ap.parse_args()

    stamp = args.run_name or datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{args.game}-{stamp}"
    if args.from_ckpt:
        from .checkpoint import seed
        seed(args.from_ckpt, run_dir)
    prior = [f for f in ("timeline.jsonl", "events.jsonl", "world_model.py", "notes.md")
             if (run_dir / f).exists()] if not args.from_ckpt else []
    if prior and not args.resume:
        sys.exit(f"{run_dir} already holds run state ({', '.join(prior)}).\n"
                 "Fresh start is the default: pick a new --run-name, or pass "
                 "--resume to deliberately continue this run, or snapshot it "
                 "first (python -m harness.checkpoint save <run_dir> <name>) "
                 "and seed a fresh dir with --from-ckpt.")
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"

    def log(kind, data):
        with open(events_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "data": data}) + "\n")

    timeline = Timeline(run_dir / "timeline.jsonl")
    env = Env(args.game, timeline, max_actions=args.max_actions)
    if args.model == "human":
        # blind-play adapter: a human at the terminal, same protocol as the models
        llm = HumanCLI()
        args.samples = 1
    elif args.model.startswith("cc:"):
        # cc:opus etc — headless Claude Code, like the nightly Opus jobs
        llm = ClaudeCLI(args.model[3:], max_tokens=args.max_tokens)
    else:
        llm = LLM(args.base_url, args.model, max_tokens=args.max_tokens)
    from .prompts import SYSTEM, SYSTEM_COACHED, VISION_NOTE
    system = SYSTEM_COACHED if args.coached else SYSTEM
    if args.vision:
        system += VISION_NOTE
    agent = Agent(env, timeline, llm, run_dir, log, samples=args.samples,
                  system=system, vision=args.vision)

    env.reset()
    print(f"[{args.game}] started · {env.win_levels} levels · run dir {run_dir}")
    log("start", {"game": args.game, "model": args.model, "win_levels": env.win_levels,
                  "coached": args.coached, "vision": args.vision,
                  # lineage marker: True = carried-over run dir, not a clean start
                  "resumed": bool(args.resume and prior)})

    extra = ""
    none_streak = 0
    for d in range(args.max_deliberations):
        try:
            result = agent.deliberate(extra)
        except BudgetExceeded as e:
            print(f"STOP: {e}")
            log("stop", {"reason": str(e)})
            break
        except Exception as e:
            # anything unexpected must reach the human, not die silently
            # in a nohup file
            abort_run(run_dir, log, f"unhandled exception in deliberation {d}: {e!r}")
            raise
        none_streak = none_streak + 1 if result is None else 0
        if none_streak >= MAX_NONE_STREAK:
            reason = (f"{none_streak} consecutive deliberations with no committed "
                      f"action (deliberation {d}, level {env.level}, "
                      f"{timeline.action_count} actions) — degenerate loop, "
                      f"check sampling params / backend")
            print(f"ABORT: {reason}")
            abort_run(run_dir, log, reason)
            sys.exit(2)
        if env.state == "WIN":
            print(f"WIN after {timeline.action_count} actions, {llm.calls} llm calls")
            log("win", {"actions": timeline.action_count, "llm_calls": llm.calls})
            break
        if args.stop_at_level is not None and env.level >= args.stop_at_level:
            print(f"STOP: reached level {env.level} (--stop-at-level) in "
                  f"{timeline.action_count} actions, {llm.calls} llm calls")
            log("stop", {"reason": f"reached level {env.level} "
                         f"({timeline.action_count} actions, {llm.calls} llm calls)"})
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
