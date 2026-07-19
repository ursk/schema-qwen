# Plans & decisions

Working log of strategy decisions for schema-qwen. Newest at the bottom.

## Goal (2026-07-16)

Reproduce the Schema harness (world-model-as-code on ARC-AGI-3) with a local
Qwen. Success bar: **fully clear one of the 25 public games**. A 25-game RHAE
score is explicitly out of scope — a ~35B local model is not a frontier model,
and the point is to measure how much of Schema's gain the *harness* transfers
downmarket.

## Decision: escalation ladder before/toward RL (2026-07-16)

Context: asked "if this doesn't go anywhere, do we RL the Qwen?" The plan is a
ladder, cheapest first, with an explicit decision rule.

**Rung 0 — harness engineering (running now).** Text protocol, exact
counterexamples with before-values and over/missed/wrong breakdowns,
best-model anchoring + REVERT, probe nudges, repetition abort. These fix
weak-model *pathologies*, not capability.

**Rung 1 — spend inference, not gradients (implemented: best-of-N).** The
backtest is a perfect, instant verifier — total mispredicted cells. Exploit it
at inference: sample N candidate replies per deliberation turn (concurrent
requests batch on vllm-mlx, so wall-clock ≈ 1–1.5× a single sample), backtest
every candidate world model, adopt the best, and log all scores. Variants if
needed: evolutionary refinement of the top-2 models, decomposed prompts
("write the rule for region X only"). Also cheap: swap the backend to
gptoss120 (117B MoE already installed) or an API-served Qwen3-Coder for a
capability read.

**Rung 2 — SFT on self-generated successes (STaR-style).** Run the harness
across all 25 games, harvest every deliberation fragment that reduced backtest
error or cleared a level, LoRA-SFT on (context → successful revision) pairs.
No reward model, no instability. Serving fits the existing single-backend +
per-request-LoRA pattern, so a "schema adapter" doesn't disturb the daily
driver.

**Rung 3 — RL proper (GRPO-style), only if 1–2 plateau.** Frame it as
*single-turn* world-model revision: given (situation, current model, mismatch
report) → emit code; reward = backtest-error delta, group-normalized over N
samples/prompt. Single-turn verifiable-reward RL is the tractable corner;
full multi-turn agentic RL on one Mac is a research project we don't start
lightly. mlx-rl has partial tooling.

**Decision rule.** Measure pass@N from rung 1 telemetry (the `bestofn` events
log every candidate's score):
- pass@16 ≫ pass@1 → the capability exists but is unreliable → rungs 2–3 are
  the right investment (they convert pass@N into pass@1).
- pass@16 ≈ 0 → sampling never finds the right revision → RL cannot conjure
  the capability; escalate the base model instead and revisit.

**What RL can't buy:** the representational leap (Schema's "the counterexample
indicts the representation itself"). That's where even Opus vs Fable diverged
in the original post. Rungs 1–2 are about reliability, not new capability.

## Decision: validate the harness with a strong model (2026-07-16)

If a frontier-class model can't clear a level through our harness, the harness
is the bottleneck, not Qwen. Implemented `cc:` models: `--model cc:opus` shells
out per turn to headless Claude Code (`claude -p --model opus`), the same
mechanism the nightly Opus jobs use — no API key required. Tool use disabled
and cwd sandboxed so the model can't read the downloaded game sources.
First validation run: LS20, `--stop-at-level 1`, 40-deliberation cap.
(Gemini via the bridge was tried first and returned empty streamed replies —
abandoned; the user wants Claude for this anyway.)

## Parked ideas: priming the weak model (2026-07-16, not yet implemented)

Two suggestions to revisit once the current ladder is measured:

1. **Demonstration priming.** Show the model a worked example of successful
   navigation — e.g. a condensed transcript of one solved level (observations,
   probes, model revisions, the final green backtest + plan) from a strong
   model's run or a hand-authored one. Few-shot in the system prompt, or
   retrieval per game. Risk to manage: context budget (a full deliberation is
   thousands of tokens) and overfitting to the demo game's mechanics.
   The cc:opus validation runs will generate exactly this material.

2. **Richer conceptual preamble.** The current system prompt explains the
   protocol but assumes the model knows what "a computer game" is like.
   Spell out the folk physics of grid games: avatars move under directional
   actions, walls block, counters/budgets tick, touching special tiles
   triggers effects, levels end on reaching a goal configuration. Cheap to
   try; measurable as fewer wasted probes in the first two deliberations.

## Validation result: Opus through the harness (2026-07-17)

Run `ls20-ccopus-val`: before the CC plan quota died mid-run, Opus reached a
world model with **2 wrong cells over 27 transitions** — block-push physics and
the bar-budget rule solved and experimentally confirmed; residual = an
unmodeled counter-digit font. Notes read like a lab notebook. No level clear
yet (`is_goal` still unknown) — run resumed after quota reset. Preliminary
verdict: **the harness is functional**; the last 10 deliberations of the
original run were an artifact (the adapter looped on the CLI's "session limit"
message — now detected, sleeps 20 min and retries instead).

## Pressure test: tiny-model (≈1.5B) imitation → RL (2026-07-17, analysis)

Proposal: math-tuned ~1.5B Qwen, imitation-learn from Opus chains, then RL.

**For it:**
1. **Dense verifiable reward.** Wrong-cell count is a continuous signal, not a
   sparse win bit — the single biggest enabler for small-model RL.
2. **Throughput is the tiny model's superpower.** ~1.5B runs 300+ tok/s and
   full-finetunes on this machine: thousands of GRPO rollouts/hour, exactly
   where 35B makes RL impractical.
3. **Precedent.** R1-Distill-Qwen-1.5B: distilled chains + RL polish produced
   outsized math gains; the recipe (imitate strong chains, then RL against a
   verifier) is the user's proposal almost verbatim.
4. The harness already carries memory, verification, and search — the model
   only needs *local revision competence*, the narrowest slice of the task.

**Against it (honest):**
1. **The task is three capabilities bundled**: grid perception over ~4k tokens
   of hex (weakest axis of tiny models), writing multi-mechanism Python
   simulators (marginal at 1.5B), causal induction. Math tuning covers only
   the third.
2. **Context.** Our prompts are ~10k tokens; tiny-model attention quality
   degrades well before that. Mitigation: compressed observations (RLE grid,
   diff-only, harness-extracted object lists) — legitimate, but note it moves
   part of *state grounding* into the harness and changes the claim.
3. **Imitation data volume.** Opus runs yield maybe a few hundred useful
   (situation → revision) pairs per game. Thin for teaching code-writing;
   fine for protocol/style. **Amplifier: procedural curriculum** — arcengine
   is installed; generate unlimited tiny games with known mechanisms, gold
   world models, and synthesized revision chains. Infinite SFT/RL data,
   controllable difficulty, clean provenance.
4. **Provenance note.** Bulk-training on Claude outputs runs into Anthropic
   usage-policy territory; the procedural-gold route avoids depending on it
   (keep Opus chains as eval reference / small protocol-priming set).
5. **The pass@N gate applies down-scale too.** After SFT, if pass@64 ≈ 0 on
   *easy synthetic* games, RL cannot rescue it — abort there, cheaply.

**Experiment E1 (when we commit to this):** Qwen3-1.7B base → procedural game
generator + gold traces → SFT (synthetic chains + protocol examples) →
Gate 1: pass@16 > 10% on held-out synthetic games → single-turn GRPO
(reward = wrong-cell delta), curriculum over mechanism count → Gate 2: clears
unseen synthetic games end-to-end → only then attempt the easiest real public
game. Success at Gate 2 is already a publishable-shaped result ("a 1.5B model
can drive a Schema-style harness"), independent of ARC-AGI-3 scores.

## Decision: observability on the dashboard (2026-07-16)

To tell at a glance whether the model is progressing or stuck: per-turn
rollout stats logged as `turn` events and plotted on the viewer —
live tokens/sec while generating, tokens per turn, world-model/notes lines
added/removed per turn, and an anomaly feed (repetition-loop truncation,
token-budget exhaustion, worker timeouts, no-command turns, LLM retries).
Rationale: every stall we've debugged so far (runaway loop, resubmit loop,
theorize-on-thin-data) was visible in exactly these signals before it was
visible in scores.

## Post-mortem: the 2026-07-17 "everything is broken" morning

What the user saw: Opus stopped playing and never finished the level; Qwen
kept playing but with no sensible moves and no hypothesis. Diagnosis:

1. **Opus didn't stop — it hit the resume's 30-deliberation cap** (11:52),
   after losing 4.4 h to a second quota window (the guard slept correctly and
   resumed at 11:29). Relaunched with `--max-deliberations 100`.
2. **A 2-cell cosmetic mismatch disabled planning all run.** LS20's live
   move-counter font is effectively unmodelable; strict certify-then-plan
   meant PLAN was never available and every multi-step COMMIT aborted on those
   cells. Fix: NEAR-GREEN tolerance (≤12 confined cells, no goal misses ⇒
   PLAN allowed, those cells excluded from execution checks).
3. **Champion scores went stale as history grew** ("WORSE than your best
   (X vs Y)" compared scores from different timeline lengths). Fix: re-score
   the champion whenever the timeline has grown.
4. **The backend was swapped out from under the Qwen run at 00:05** by a
   nightly wake restoring the resting default (qwen36-lens, single-flight
   mlx-lm). Consequences: 4-way concurrent sampling serialized; the
   wedge-watchdog mistook the long queues for hangs and kill-restarted the
   backend every ~10 min for 12 h; 51 HTTP retries; and all best-of-N samples
   came back byte-identical (mlx-lm seeds deterministically) — best-of-4 was
   silently pass@1. Fixes: backend switched back to qwen36 (vllm-mlx),
   per-sample seeds + temperature ladder, loud ops-journal note telling
   nightly wakes not to restore the default while a run is live.
5. **Qwen's commandless enumeration spiral** (newline-free "predicts 3->c for
   y=105 … y=106 …" prose) evaded the line-based loop detector and burned
   whole deliberations. Fixes: shingle-based repetition detector; end the
   deliberation after 3 commandless turns and rebuild context from durable
   state.

Meta-lesson for the writeup: most of what looked like "the model is too weak"
was harness/infra pathology. The model-capability read on Qwen is only valid
from runs after these fixes.

## Ruling 2026-07-17: strict certification, no near-green (Urs)

The near-green tolerance shipped in the morning fixes was reverted the same
day on Urs's ruling: "our goal is to clear one level fair and square. No
cheating." The original Schema is strict — its published LS20 trace (Opus 4.8)
reaches 36/36 exact transitions WITH the counter box modeled — so the counter
font is demonstrably modelable and the tolerance was lowering our bar, not
fixing an environment quirk. Kept from that work: the confined-mismatch HINT
(list the exact stubborn cells, point out they're deterministic and decodable
from history) — better counterexamples are fair; waived cells are not.

Also ruled: no more open-ended Opus runs (token cost). The cc: validation
lane is reserved for harness debugging only, with strict token limits agreed
in advance. Task focus returns to Qwen clearing a level.

## Rethink after stopping run1 (2026-07-17): Qwen thrashes — why, and what next

Run stopped by Urs ("just thrashing"). The telemetry agrees and localizes it.

**What the data shows (runs/ls20-run1, 368 backtests, 300 best-of-N events):**
- 348 code-rewrite turns vs 39 commits; 259 repetition-loop aborts; 148
  no-command turns. The model rewrites instead of probing, and loops.
- Backtest scores sit at 27k–180k wrong cells. Best-of-N spreads are marginal
  (27,428 vs 27,718): sampling harder does not rescue it — the errors are not
  unlucky samples, they are a wrong theory sampled consistently.
- notes.md is the tell. Early notes are coherent observation; late notes are a
  self-reinforcing fantasy ("Region D is the goal", "Reverting to best model
  (21k errors)" repeated ~10×). The append-only notes became an attractor:
  every new context starts by re-reading the junk that produced the last junk.

**The root cause is perception, not search.** LS20 is a block-pushing game
(the published Opus trace: "the avatar is the 5×5 cyan+maroon block... the bar
is a move budget"). Qwen never formed that gestalt. A block moving left looks,
cell-wise, like paired bands of 3->c and c->3 — Qwen read those diffs as a
"color cycling" mechanic in static regions, then invented region-goals to
match. Opus's first deliberation produced objects (avatar, budget bar, HUD);
Qwen reasons at raw cell granularity for the whole run and drowns.

**Proposals, ordered by leverage:**
1. **Kill junk accumulation (cheap, no fidelity cost).** Short deliberations
   (max 3–4 turns; the 14-turn conversations compound garbage). Notes become
   bounded: the model must periodically REWRITE them to ≤25 lines (forced
   consolidation turn) instead of appending forever. Add a periodic
   "fresh eyes" deliberation that sees only harness facts (timeline-derived
   experiment log: action → observed diff), not the model's own past prose —
   an escape hatch from wrong gestalts.
2. **Object-level observation reporting (the big lever; documented deviation).**
   Harness-side connected components: report each grid and each transition as
   objects ("5×5 block color c at (19,45)"; "block DISAPPEARED at (19,45),
   identical block APPEARED at (24,45)" — i.e. mechanical movement detection).
   Purely deterministic preprocessing of the agent's own observations — no
   game internals touched — but it moves part of Schema's Level-1 state
   grounding into the harness, so it must be reported as a deviation. It is
   precisely the capability Qwen lacks and Opus supplies itself.
3. **Gestalt priming (cheap).** One forced early deliberation: "What is the
   avatar? What is HUD/score? Which archetype fits (pusher/maze/navigation/
   budget/toggle)?" — answer stored in a dedicated GESTALT slot the model
   sees every turn and may revise but not bury. Generalizes the parked
   "explain what a computer game is" idea.
4. **Fresh run (run2).** run1's durable state is poisoned by the Region-D
   fantasy (and by 12 h of broken-backend chaos). Nothing worth resuming.
5. **Capability ladder, local-only.** qwen36 is 35B-A3B — ~3B ACTIVE params
   per token. gptoss120 (117B/5.1B-active, installed, ~66 tok/s) is the
   natural next rung: same harness, same rules, no API tokens. The best-of-N
   flatness suggests escalate-or-scaffold, not sample-harder.

Recommendation: 1+3+4 unconditionally (they cost nothing and remove the
failure mode we watched); then EITHER 2 (keep qwen36, add object scaffold,
document the deviation) OR 5 (keep pure Schema observations, bigger local
model) as the controlled experiment — running both, one at a time, tells us
whether the missing ingredient is perception scaffolding or raw capability.

## Result: gptoss120 on LS20 (2026-07-17/18, runs oss1+oss2)

After the serialized-route fix (empty streams retried, samples=1): 133
deliberations, 142 actions, ZERO green backtests, level 0/7. gpt-oss forms an
object-level ontology immediately (unlike qwen36) and probes with discipline
(~1 experiment per deliberation, near-zero over-firing), but it locked into a
"two paint-blocks with column offsets" gestalt in the first hour and absorbed
every subsequent counterexample as a coordinate tweak — per-transition error
flat (~70–115 wrong cells) for 100+ deliberations. The representational leap
("one avatar that moves") never came. Schema's central claim reproduced in
the negative: backtest+search transfer downmarket; paradigm breaks don't.
Next per Urs: try gemma26 (run gemma1), same game, same rules.

## Calibration from Urs's own playtest (2026-07-18)

Urs played the first few public games by hand: the avatar style stays the
same and difficulty only rises — so there is no softer entry game, and LS20
results generalize. The "try a different game" idea is dead. Opus through our
cc: adapter did somewhat better than the locals but no clear on a ~5h token
budget, with poor progress per wall-clock — note the published Schema cleared
LS20 with Opus 4.8 through a native tool-calling agent, so part of our gap is
the reproduction's flattened transport, not model capability. Fair paths
forward, stack-ranked: (1) finish the local capability map (gemma1 running),
(2) a patient quota-riding Opus lane (resume-across-sessions already works),
(3) the E1 synthetic-curriculum training ladder — under the no-cheating rule,
changing the MODEL via training on procedural games with clean provenance is
fair; inference-time scaffolds and priming are not.

## The perception turn (2026-07-18, post-bake-off)

Insight from reviewing the bake-off with Urs: the situation prompt already
contains everything (grid, exact diffs, mismatch before-values), but every
model — Qwen through Opus — eyeballs the hex in-head instead of computing over
it, which is exactly where "color cycling" gestalts come from. The models have
Python; the harness gave them nowhere to run it except inside the backtest,
whose stdout goes nowhere. Affordance gap, not capability gap (hypothesis).

Built (see README "Perception affordances", all documented deviations):
1. `ANALYZE` — read-only python over the recorded timeline, stdout returned.
   The fair-play REPL: vision is available but must be *chosen and built*.
2. `--vision` — sighted harness: OBJECTS decomposition in the prompt + every
   transition as a sparse in-grid change map + MOVEMENT translation detection
   (verified on the bake-off timelines: one transition suffices to read off
   the 5x5 two-tone avatar and the action→(dx,dy) mapping, e.g. 1 → dy=-5).
3. `--model human` — blind-play adapter (stdin/stdout, same protocol). The
   deal: Urs plays the blind game now that the sighted harness exists.

Experiment ladder this enables, cheapest-first, same game (LS20):
- qwen36 --vision (does perception alone unlock the representational leap
  that 133 gpt-oss deliberations never made?)
- qwen36 ANALYZE-only (does it think to build itself vision when told it can?)
- gptoss120 ANALYZE-only (ditto for the disciplined prober)
- human blind baseline (Urs, n=1, priceless)

## Result: qwen36 --coached --vision on LS20 (2026-07-18/19, run vis1)

Overnight: 27 deliberations, 85 actions, 908 LLM calls, ~2.8M generated
tokens, 7 green backtests (max exact on 42 transitions), 3 surprises, 0
GAME_OVERs — level 0/7. Viewer was live at /schema.

**Perception is solved — the sighted harness worked.** Turn 1 opened with a
correct object inventory (vs the bake-off's "color cycling"). By deliberation
13 the model had: the 5x5 two-tone avatar, the full action->(dx,dy) control
map (1:Up 2:Down 3:Left 4:Right, 5 cells/step), walls, and the budget bar's
erosion rule — all verified green on 37+ transitions. It even wrote its own
get_connected_components() INSIDE world_model.py and represented state as
objects. The representational leap that blind qwen36 and 133 gpt-oss
deliberations never made happened in under an hour with eyes. Best-of-N also
finally did real work (candidate scores like {2, 126, 2} -> adopt 2).

**The next bottleneck is now cleanly exposed: goal discovery + planning
initiation. Zero PLAN calls in 11 hours despite repeated greens.** Green never
converted into search because (a) the model treats green as license to keep
probing, and (b) is_goal is only ever constrained by level-up events, of
which there were none — chicken-and-egg: no goal evidence without clearing a
level, no clearing without a goal hypothesis it trusts. It committed hard to
one wrong goal theory ("erode the b bar to zero = win"), got a clean refutation
(bar emptied, nothing happened), and — unlike gpt-oss — revised honestly
("goal might be to reach the 8s"). No paradigm lock; it just ran out of night.

Harness warts observed: backtest on 0 transitions reports GREEN (vacuous,
unlocks PLAN inside fantasy); 171 red backtests = lots of churn the champion/
restore machinery absorbed correctly.

**Next scaffold iteration (quick-iteration mode, Urs 2026-07-19):**
1. AUTO-BFS on green — when a backtest goes green, the harness immediately
   runs BFS inside the certified model and reports the result unprompted
   ("plan found: ..." / "NO goal reachable — your is_goal is untested or
   wrong"). Pure automation of an existing tool, no knowledge injected; turns
   green into an actionable signal and makes is_goal falsifiable every time.
2. GOAL slot — like notes but dedicated and mandatory: current goal
   hypothesis + evidence for/against, shown every turn, must be revised when
   refuted. (Generalizes the parked gestalt-priming idea, minus the genre
   knowledge.)
3. Vacuous-green fix: 0-transition backtest reports "no data yet", not GREEN.
4. Consider: after N consecutive green deliberations with no PLAN, the
   harness nudges ("you have a verified model and N untested goal theories —
   PLAN or state why not").

## Cache-hit investigation → backend switch (2026-07-19)

vis2 paused mid-morning to chase abysmal prefix-cache hits (6.7%) on vllm-mlx
qwen36. Root cause chain: (a) qwen3.6 is hybrid (30/40 GDN layers, state
can't rewind) so only exact-prefix cache hits are possible; (b) the server's
think-suffix stripping computes S=2 but our default enable_thinking:false
renders a 4-token empty-think suffix — every stored key carries a 2-token
residue, so append-continuation (every deliberation turn) never matches and
re-prefills ~10k tokens; (c) latent: SSM layer state is silently untrimmed at
store time, so fixing (b) alone would serve subtly corrupted continuations.
An earlier "template re-renders history" hypothesis was tested and retracted
(dormant for our traffic shape — no reasoning_content in messages).

Resolution: schema runs moved to qwen36-gguf (llama.cpp slot cache +
recurrent checkpoints): measured 26/949 tokens processed on append-
continuation (~97% reuse), 75 tok/s single-stream decode. Plist gained
--parallel 2 / -c 131072; runs use --samples 2. vllm-mlx MLLM fix parked as
a follow-up (matters for Moss/digest). vis2 resumed on gguf, same run dir.
