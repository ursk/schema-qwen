# The Schema Bake-off — LS20, 30 minutes per cell (2026-07-18)

Rules set by Urs: four local models plus Opus, each playing ARC-AGI-3 game
LS20 through the Schema-reproduction harness, 30 minutes of wall clock per
cell, two arms each — **coached** (the "easy" harness: system prompt carries
video-game genre knowledge and explicit counters to previously observed
failure modes) and **plain** (the original knowledge-free prompt). Fresh runs,
`--samples 1`, 6144-token budget (8192 for Opus), strict certification
(no near-green tolerance). Judged from `events.jsonl` after teardown.

## Results

| cell | deliberations | actions | levels | greens | BFS | best wrong/transition |
|---|---:|---:|---:|---:|---:|---:|
| **opus coached** | **73** | **440** | **2** | **2** | **3** | **0.0** |
| opus plain | 0 | 0 | — | — | — | (void: CC quota exhausted at start) |
| gptoss120 plain | 15 | 16 | 0 | 0 | 0 | 21.0 |
| qwen36 plain | 8 | 9 | 0 | 0 | 0 | 24.1 |
| qwen36 coached | 12 | 14 | 0 | 0 | 0 | 26.0 |
| gptoss120 coached | 13 | 16 | 0 | 0 | 0 | 29.6 |
| gemma26 plain | 7 | 7 | 0 | 0 | 0 | 33.6 |
| gemma26 coached | 7 | 6 | 0 | 0 | 0 | 41.4 |
| mistral4 plain | 12 | 11 | 0 | 0 | 0 | 48.7 |
| mistral4 coached | 14 | 16 | 0 | 0 | 0 | 53.3 |

("greens" = fully green backtests over ≥5 real transitions; "best
wrong/transition" = lowest total mispredicted cells per recorded transition
across the cell's backtests.)

## Findings

1. **Opus-coached closed the full Schema loop — twice.** Green backtest →
   BFS plan → COMMIT PLAN → LEVEL UP at 19:46, again ~20:09, then methodical
   level-2 play (ring pads, item pickups, routes planned in notes) until the
   clock. The first level clears of this project. 440 actions in 30 minutes —
   the per-step misprediction abort never had to fire on a certified model.
2. **No local model cleared anything or certified a single green** in 30
   minutes. Best modeling: gptoss120-plain at 21 wrong cells/transition.
3. **Coaching did not help the locals.** In all four, plain matched or beat
   coached on best score. The genre knowledge that unlocked Opus (which has
   the reasoning to *use* an avatar concept) mostly cost the locals prompt
   budget. Capability, not knowledge, is the binding constraint at this scale
   — with one exception: coaching cured gemma26's click fixation (its
   original run spent 13 straight actions clicking; both bake-off cells took
   real directional actions immediately, though the plain cell also did,
   suggesting that pathology is partly stochastic).
4. **Model personalities were stable across arms.** Mistral: fastest cadence,
   sloppiest models (48–53 w/t). Gemma: slowest cadence (6–7 deliberations),
   middling accuracy. Qwen: balanced, still cell-level perception. gpt-oss:
   most accurate local, still never questioned its representation.
5. **The bottleneck is perception, not reasoning budget.** Every local model
   eyeballs hex grids in-head and misreads object translation as color
   change; none thought to analyze the grid programmatically despite writing
   Python fluently. This diagnosis (Urs's: "if I was given the world in that
   format I would only touch it programmatically") led directly to the
   vision-harness work that followed the bake-off.

## Epilogue: the vision harness (2026-07-19, separate session)

A follow-on harness adds an `ANALYZE` command (a read-only REPL over the
agent's own timeline) and a `--vision` mode (connected-block OBJECTS section
in the prompt; transitions rendered as sparse in-grid change maps plus
MOVEMENT lines when changed cells are consistent with a translating object).
First signal (`ls20-vis2`, qwen36 + coached + vision): **three green
backtests over real history within six deliberations**, and notes that read
like Opus's — "The c/9 block moves UP/DOWN. It seems to be the player. The
3s are footprints." The same model that spent 368 backtests in cell-diff
hell without ever forming an object concept.
