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
