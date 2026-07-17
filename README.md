# schema-qwen

Reproduction of [Schema](https://schema-harness.github.io/) (Impossible Research,
Jul 2026) on ARC-AGI-3, with a local Qwen instead of a frontier model.

Schema's idea: the agent plays like a physicist. Its theory of the game is an
executable `step(state, action)` program; the harness backtests that program
against an append-only log of every real transition, plans with BFS *inside*
the certified program, and executes plans with a per-step misprediction abort.

Goal here is deliberately modest: **fully clear one public game with a green
backtest** using a ~30B local model. The harness carries as much of the load
as possible (exact diffing, mismatch localization, backtest, search); the
model only writes the world-model code and chooses probes.

## Run

```bash
.venv/bin/python -m harness.run --game ls20            # defaults: qwen36-lens via :8086
.venv/bin/python -m harness.run --game ft09 --model qwen36-mlxlm
```

Artifacts land in `runs/<game>-<stamp>/`: `timeline.jsonl` (ground truth),
`world_model.py` + `notes.md` (the agent's durable state), `world_model_best.py`
(lowest-error model seen), `events.jsonl` (structured log), `trace.log` (raw stream).

## Watching a run

**Live dashboard** (game canvas + play-by-play + agent notes + world model):

```bash
.venv/bin/python -m harness.viewer --run runs/ls20-run1 --port 8123
# open http://127.0.0.1:8123/
```

The canvas follows the game live (uncheck *follow live* to scrub/replay any
step). The play-by-play panel narrates the run: model turns, backtest verdicts
with wrong-cell breakdowns, BFS plans, executed actions, surprises, level-ups.

**Raw token stream** — every token the model generates, interleaved with
harness responses, as it happens:

```bash
tail -f runs/ls20-run1/trace.log
```

**Ground truth** — `timeline.jsonl` has one JSON line per real transition
(action, full 64×64 grid, level, state); `events.jsonl` has the structured
run log (`start/llm/backtest/bfs/commit/progress/win/stop`).

## Diagnostics that shaped the harness (local-model failure modes)

Observed with Qwen3.6-35B-A3B and fixed in the harness — likely relevant to
anyone running Schema-style loops on small models:

1. **Repetition runaway.** In long contexts the model can lock into repeating
   a line until the token cap ("Wait, the provided grid shows: …" ×50). The
   streaming client detects periodic repetition mid-stream, aborts the
   generation, and tells the model why (`[TRUNCATED BY HARNESS…]`).
2. **Unchanged-model resubmit loop.** The model narrates the right next probe
   but pastes its current world model back instead of committing. Fix:
   byte-identical resubmissions are rejected with pointed feedback, and an
   explicit `COMMIT` outranks an unchanged pasted-back code block.
3. **Theorizing on thin data.** With 2 recorded transitions it will rewrite
   the model forever instead of acting. When the backtest is RED and fewer
   than ~12 real transitions exist, the report nudges toward a short probe.
4. **Regression blindness.** The model can't remember which of its models was
   best. Every submission is scored (total wrong cells); regressions are
   called out ("WORSE than your best (900 vs 72) — say REVERT"), and `REVERT`
   restores the best-scoring model.
5. **Mismatch reports need before-values.** "predicted 9, real 3" is much
   weaker than "was 3: predicted 9, real 3" plus the aggregate breakdown
   ("897 cells your model changed but reality did NOT") — the latter
   distinguishes over-firing rules from missing mechanisms at a glance.

## Layout

- `harness/timeline.py` — append-only transition log, level segmentation
- `harness/worker.py` — sandboxed backtest / BFS / predict over model-written code
- `harness/sandbox.py` — subprocess + timeouts around the worker
- `harness/agent.py` — deliberation loop, text protocol (no tool-calls: local
  models are fragile there), context compaction
- `harness/prompts.py` — the physicist system prompt + world-model contract
- `harness/envio.py` — ARC-AGI-3 env wrapper (`arc-agi` toolkit, runs games locally)
- `harness/run.py` — outer observe→deliberate→execute→record loop

`environment_files/` (downloaded game source) is gitignored and must never be
shown to the agent — it contains the answers.

## Agent protocol

The model ends each reply with one command:

| Command | Effect |
|---|---|
| ```` ```python ```` block | replace `world_model.py`, auto-backtest, get mismatch report |
| `PLAN` | BFS inside the model (only when backtest is green) |
| `COMMIT PLAN` | execute the last found plan, abort on first misprediction |
| `COMMIT: 1 3 6@12,40` | execute explicit probe actions |
| `NOTE: ...` | append to persistent `notes.md` |

World-model contract: `init_state(grid)`, `step(state, action)`,
`render(state)`, `is_goal(state)`, optional `candidate_clicks(state)`.
