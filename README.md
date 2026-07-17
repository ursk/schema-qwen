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
`world_model.py` + `notes.md` (the agent's durable state), `events.jsonl` (log).

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
