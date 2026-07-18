"""Live browser viewer for a run: replays timeline.jsonl on a canvas.

Usage:  .venv/bin/python -m harness.viewer --run runs/ls20-run1 [--port 8123]
Then open http://127.0.0.1:8123/ — follows the run live, scrub to replay.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RUN_DIR = None
METRICS_URL = "http://127.0.0.1:8084/metrics"
KV_HISTORY = []  # (t, kv_gb, metal_active_gb, entries) — in-process, viewer lifetime


def poll_backend_metrics():
    import urllib.request
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=0.8) as r:
            text = r.read().decode()
    except Exception:
        return None
    vals = {}
    for line in text.splitlines():
        if line.startswith("vllm_mlx_cache_memory_bytes "):
            vals["kv_bytes"] = float(line.split()[-1])
        elif line.startswith('vllm_mlx_metal_memory_bytes{kind="active"}'):
            vals["metal_active"] = float(line.split()[-1])
        elif line.startswith('vllm_mlx_metal_memory_bytes{kind="peak"}'):
            vals["metal_peak"] = float(line.split()[-1])
        elif line.startswith("vllm_mlx_cache_entry_count"):
            vals["entries"] = float(line.split()[-1])
        elif line.startswith("vllm_mlx_cache_hits"):
            vals["hits"] = float(line.split()[-1])
        elif line.startswith("vllm_mlx_cache_misses"):
            vals["misses"] = float(line.split()[-1])
    if not vals:
        return None
    import time as _t
    gb = 1024 ** 3
    m = {
        "kv_gb": round(vals.get("kv_bytes", 0) / gb, 2),
        "metal_active_gb": round(vals.get("metal_active", 0) / gb, 1),
        "metal_peak_gb": round(vals.get("metal_peak", 0) / gb, 1),
        "entries": int(vals.get("entries", 0)),
        "hit_rate": round(vals.get("hits", 0) /
                          max(vals.get("hits", 0) + vals.get("misses", 0), 1), 2),
    }
    KV_HISTORY.append((_t.time(), m["kv_gb"], m["metal_active_gb"], m["entries"]))
    del KV_HISTORY[:-240]
    m["history"] = [{"t": h[0], "kv_gb": h[1], "metal_gb": h[2]} for h in KV_HISTORY]
    return m


def humanize(e):
    """One play-by-play line per interesting events.jsonl entry."""
    import datetime
    k, d = e.get("kind"), e.get("data", {})
    ts = datetime.datetime.fromtimestamp(e.get("t", 0)).strftime("%H:%M:%S")
    if k == "start":
        return f"{ts} ▶ run started: {d.get('game')} · {d.get('win_levels')} levels · model {d.get('model')}"
    if k == "llm":
        reply = (d.get("reply") or "").strip().replace("\n", " ")
        return f"{ts} 🧠 model (turn {d.get('turn')}): {reply[:160]}…"
    if k == "backtest":
        if d.get("ok"):
            return f"{ts} ✅ backtest GREEN — {d.get('transitions_checked')} transitions reproduced exactly"
        mm = (d.get("mismatches") or [{}])[0]
        return (f"{ts} ❌ backtest RED — {d.get('total_wrong_cells', '?')} wrong cells "
                f"({mm.get('breakdown', mm.get('detail', ''))})")
    if k == "bfs":
        if d.get("ok") and d.get("plan") is not None:
            return f"{ts} 🔍 BFS found a {len(d['plan'])}-action plan ({d.get('expanded', '?')} nodes expanded)"
        return f"{ts} 🔍 BFS found no goal ({d.get('expanded', '?')} nodes, {d.get('distinct_states', '?')} states)"
    if k == "commit":
        acts = " ".join(d.get("actions", []))
        res = d.get("result", "")
        tail = ""
        if "SURPRISE" in res:
            tail = " — SURPRISE, plan aborted"
        elif "LEVEL UP" in res:
            tail = " — LEVEL UP!"
        elif "WIN" in res:
            tail = " — WIN!"
        elif "GAME OVER" in res:
            tail = " — game over"
        return f"{ts} 🎮 executed: {acts}{tail}"
    if k == "turn" and d.get("anomalies"):
        return f"{ts} ⚠ anomaly: {', '.join(d['anomalies'])} (turn {d.get('turn')}, {d.get('gen_tokens')} tok)"
    if k == "bestofn":
        sc = d.get("scores", {})
        return (f"{ts} 🎲 best-of-{d.get('n')}: {d.get('code_candidates')} code candidates, "
                f"scores {sc} → adopted #{d.get('adopted')}")
    if k == "progress":
        return (f"{ts} ── deliberation {d.get('deliberation')} done · level {d.get('level')} · "
                f"{d.get('actions')} actions · {d.get('llm_calls')} llm calls")
    if k == "win":
        return f"{ts} 🏆 GAME WON in {d.get('actions')} actions ({d.get('llm_calls')} llm calls)"
    if k == "stop":
        return f"{ts} ■ stopped: {d.get('reason')}"
    return None

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>schema-qwen · live run</title>
<style>
:root{
  --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,.10);
  --accent:#2a78d6; --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
}
@media (prefers-color-scheme: dark){
  :root{
    --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,.10);
    --accent:#3987e5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:12px 16px 48px}
h1{font-size:17px;margin:8px 0 2px}
h2{font-size:13px;font-weight:600;color:var(--ink2);margin:0 0 8px;
  text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--muted);font-size:12px;margin-bottom:12px}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px;margin-bottom:14px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.chip{font-size:12px;color:var(--ink2);border:1px solid var(--border);
  border-radius:6px;padding:2px 8px;background:var(--page)}
.chip b{color:var(--ink);font-weight:600}
.chip.green b{color:var(--good)}
.chip.red b{color:var(--critical)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--muted);margin-right:6px}
.dot.live{background:var(--good);animation:pulse 2s infinite}
@keyframes pulse{50%{opacity:.35}}
.cols{display:grid;grid-template-columns:552px 1fr;gap:14px}
canvas{image-rendering:pixelated;border:1px solid var(--grid);border-radius:6px;
  display:block;width:512px;height:512px}
.bar{margin-top:10px;display:flex;gap:8px;align-items:center}
.bar input[type=range]{flex:1}
.bar button{border:1px solid var(--border);background:var(--surface);color:var(--ink);
  border-radius:6px;padding:4px 12px;font-size:13px;cursor:pointer}
label{font-size:12px;color:var(--ink2);user-select:none}
.meta{font-size:12px;color:var(--muted);margin-top:6px;min-height:1.4em;
  font-variant-numeric:tabular-nums}
.log{white-space:pre-wrap;word-break:break-word;
  font:11px/1.55 ui-monospace,Menlo,monospace;color:var(--ink2);
  max-height:300px;overflow-y:auto;background:var(--page);
  border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin:0}
details.expl{margin-top:2px}
details.expl summary{cursor:pointer;font-size:13px;font-weight:600;
  color:var(--ink2);padding:2px 0}
details.expl summary:hover{color:var(--ink)}
.expl h3{font-size:13px;margin:16px 0 4px;color:var(--ink)}
.expl p,.expl li{font-size:13px;color:var(--ink2);line-height:1.55}
.expl p{margin:6px 0;max-width:86ch}
.expl ul,.expl ol{margin:4px 0;padding-left:20px;max-width:84ch}
.expl li{margin:4px 0}
.expl code{font:12px ui-monospace,Menlo,monospace;background:var(--page);
  border:1px solid var(--border);border-radius:4px;padding:0 4px}
.expl a{color:var(--accent)}
.expl b{color:var(--ink)}
footer{color:var(--muted);font-size:11px;margin-top:20px;max-width:76ch}
@media(max-width:1100px){.cols{grid-template-columns:1fr}}
</style></head><body>
<div class="wrap">
  <h1><span class="dot live" id="dot"></span><span id="title">schema-qwen</span></h1>
  <div class="sub">world-model-as-code agent on ARC-AGI-3 · <span id="substat">connecting…</span></div>
  <div class="chips" id="chips"></div>
  <div class="cols">
    <div>
      <div class="card">
        <h2>Game</h2>
        <canvas id="c" width="512" height="512"></canvas>
        <div class="bar">
          <button id="play">▶</button>
          <input type="range" id="scrub" min="0" max="0" value="0">
          <label><input type="checkbox" id="follow" checked> follow live</label>
        </div>
        <div class="meta" id="meta"></div>
      </div>
      <div class="card">
        <h2>Rollout stats</h2>
        <div style="display:flex;gap:18px;align-items:baseline">
          <div><span id="tps" style="font-size:28px;font-weight:700;font-variant-numeric:tabular-nums">–</span>
            <span style="font-size:12px;color:var(--muted)"> tok/s</span></div>
          <div class="meta" id="genstat" style="margin-top:0"></div>
        </div>
        <div class="meta" style="margin-top:8px">tokens per turn (green = code accepted, gray = other, red edge = anomaly)</div>
        <svg id="turnbars" viewBox="0 0 480 60" preserveAspectRatio="none"
             style="width:100%;height:60px;background:var(--page);border:1px solid var(--border);border-radius:6px"></svg>
        <div class="meta" style="margin-top:6px">world-model lines added/removed per turn</div>
        <svg id="wmbars" viewBox="0 0 480 40" preserveAspectRatio="none"
             style="width:100%;height:40px;background:var(--page);border:1px solid var(--border);border-radius:6px"></svg>
        <div class="meta" style="margin-top:8px">backend memory — KV cache (blue) · Metal active (gray), GB
          <span id="kvchips"></span></div>
        <svg id="kvspark" viewBox="0 0 480 40" preserveAspectRatio="none"
             style="width:100%;height:40px;background:var(--page);border:1px solid var(--border);border-radius:6px"></svg>
        <div class="meta" style="margin-top:8px">anomalies</div>
        <pre class="log" id="anoms" style="max-height:110px"></pre>
      </div>
    </div>
    <div>
      <div class="card"><h2>Model output <span id="genliv" style="color:var(--good);font-weight:400;text-transform:none;letter-spacing:0"></span></h2>
        <pre class="log" id="gentail" style="max-height:160px"></pre></div>
      <div class="card"><h2>Play-by-play</h2><pre class="log" id="feed"></pre></div>
      <div class="card"><h2>Agent notes (notes.md)</h2><pre class="log" id="notes" style="max-height:170px"></pre></div>
      <div class="card"><h2>World model (world_model.py)</h2><pre class="log" id="wm" style="max-height:240px"></pre></div>
    </div>
  </div>

  <div class="card">
    <details class="expl">
      <summary>What is this? — the method, our reproduction, and where we deviate</summary>

      <h3>The benchmark: games that don't tell you the rules</h3>
      <p>
        <a href="https://arcprize.org/arc-agi/3">ARC-AGI-3</a> (ARC Prize
        Foundation, March 2026) is a set of interactive games. The agent sees a
        64×64 grid of 16 colors — that's the canvas on the left — and a handful
        of unlabeled actions: <code>1</code>–<code>5</code>,
        sometimes a click <code>6@x,y</code>, and <code>RESET</code>. Nothing
        else. No rule sheet, no object list, no stated goal, no reward signal.
        The agent must discover what the pixels mean, what its actions do, and
        what counts as progress, purely by acting and watching. Each game has
        several levels; the official metric (RHAE) scores completion <i>and</i>
        action-efficiency against first-time human baselines, squaring the
        penalty for wasted actions. Frontier models scored 0.51% at launch;
        the best official result by July 2026 was 13.33% (GPT-5.6 Sol, Public
        set). You can play the games yourself at
        <a href="https://three.arcprize.org">three.arcprize.org</a> — five
        minutes with one game makes everything below concrete.
      </p>

      <h3>The method we're reproducing: Schema</h3>
      <p>
        <a href="https://schema-harness.github.io/">Schema</a> (Impossible
        Research, July 2026) self-reports ~99% RHAE on the 25 public games
        using Claude Opus 4.8 + Fable 5 — with the <em>same</em> models scoring
        42.8% under a generic coding harness. The claim is that the arrangement
        around the model, not the model, closes that gap. Schema makes the
        agent behave like a physicist:
      </p>
      <ul>
        <li><b>The world model is a program, not a vector.</b> The agent's
          entire theory of the game lives in one editable Python file (right
          panel, live). It must define what the state <i>is</i>
          (<code>init_state</code> — which pixels form objects, what hidden
          variables exist) and how it <i>moves</i>
          (<code>step(state, action)</code>), plus <code>render</code> (state →
          expected grid) and <code>is_goal</code> (what completes a level).
          Because the theory is code, it is readable, diffable, and — the key
          property — <em>executable</em>: it doubles as a simulator.</li>
        <li><b>Certify against all of history.</b> Every real transition ever
          observed is recorded in an append-only timeline. A
          <i>backtest</i> replays the candidate program over the entire record
          and demands exact, cell-perfect agreement. One wrong pixel = RED,
          with the counterexample. The agent cannot fool itself about how good
          its theory is.</li>
        <li><b>Plan inside the certified model.</b> Once the backtest is GREEN,
          breadth-first search runs thousands of simulated games inside the
          program to find a shortest action sequence to the goal — costing
          zero real actions. This is where the efficiency comes from: pay for
          discovery once, then plan for free.</li>
        <li><b>Reality outranks the model.</b> During execution every real
          frame is compared with the model's prediction; the first mismatch
          aborts the plan and becomes a counterexample the model must explain
          before planning resumes.</li>
        <li><b>Act to discover, not just to win.</b> When several rules fit
          the history, the right move is the experiment that best separates
          them — commit it, observe, revise.</li>
      </ul>
      <p>
        The loop you see in the play-by-play is exactly this cycle:
        <code>observe → deliberate (theorize / backtest / plan) → execute →
        record</code>, repeated per level.
      </p>

      <h3>Our reproduction</h3>
      <p>
        The harness (<a href="https://github.com/ursk/schema-qwen">ursk/schema-qwen</a>)
        reimplements the Schema loop from the blog post alone — no code was
        released. The games run locally via the official <code>arc-agi</code>
        toolkit; the agent talks to a local LLM served on this same Mac Studio.
        The harness owns everything deterministic: the append-only timeline,
        the sandboxed backtest with cell-level diff reports, BFS over the
        model's state space, per-step prediction checks during execution, and
        the persistent memory files (<code>notes.md</code>,
        <code>world_model.py</code>). The LLM owns exactly two things: writing
        the world-model code, and choosing which experiment to run next.
      </p>

      <h3>Where we deviate from the published method</h3>
      <ol>
        <li><b>A ~35B local model instead of frontier models.</b> Schema used
          Claude Opus 4.8 / Fable 5 and GPT-5.6 Sol at max reasoning, with a
          two-model fallback pairing per game. We run one model:
          Qwen3.6-35B-A3B, 4-bit, on a single Mac Studio. Accordingly the goal
          is deliberately modest — <em>fully clear one public game</em> — not
          a 25-game RHAE score, and there is no fallback pairing.</li>
        <li><b>Plain-text command protocol instead of native tool calls.</b>
          Small models are fragile at structured tool-calling (our own
          SWE-bench-style evals showed illegal-tool-format as the dominant
          failure axis), so the agent answers with a python code block or a
          bare command line (<code>PLAN</code>, <code>COMMIT …</code>,
          <code>NOTE:</code>), parsed with regexes and re-prompted on
          failure.</li>
        <li><b>The harness compensates for weak-model pathologies</b> that the
          Schema post never needed to mention. Each was added after watching
          this agent fail in a specific way:
          <ul>
            <li><i>Repetition runaway</i> — in long contexts the model can lock
              into repeating one line until the token cap; the streaming client
              detects periodicity mid-generation and truncates with an
              explanation.</li>
            <li><i>Unchanged resubmits</i> — the model narrates the right probe
              but pastes its old code back; identical resubmissions are
              rejected, and an explicit <code>COMMIT</code> outranks an
              unchanged code block.</li>
            <li><i>Theorizing on thin data</i> — with under ~12 recorded
              transitions, RED backtest reports nudge toward a real probe
              instead of another rewrite.</li>
            <li><i>Regression blindness</i> — every submission is scored (total
              mispredicted cells); the harness keeps the best-scoring model,
              announces regressions, offers <code>REVERT</code>, and starts
              each deliberation from the best-verified theory.</li>
            <li><i>Richer counterexamples</i> — mismatch reports include each
              cell's before-value and an aggregate breakdown ("N cells your
              model changed but reality did NOT / M cells reality changed but
              your model did NOT"), separating over-firing rules from missing
              mechanisms at a glance.</li>
          </ul>
          Whether these are "deviations" or just the price of running the
          method on a small model is, in a sense, the experiment.</li>
        <li><b>Strict certification, kept.</b> Like the original, PLAN is only
          available on a fully green backtest — no tolerance for "cosmetic"
          mismatches. We briefly tried relaxing this (a validation run saw a
          frontier model locked out of planning all run by a 2-cell unmodeled
          counter font) and reverted the same day: the goal here is a
          fair-and-square clear, and every pixel is deterministic and
          therefore modelable. What we kept instead is a sharper
          counterexample: when all mismatches are confined to a few cells,
          the harness lists them and points out that whatever lives there can
          be decoded from recorded history.</li>
        <li><b>Simplified planning.</b> Our BFS searches simple actions plus
          clicks the model explicitly proposes via
          <code>candidate_clicks(state)</code>, with node/depth caps, states
          keyed by canonical JSON. Schema describes richer deliberation-time
          search but not its exact machinery; ours is the minimal version.</li>
        <li><b>Backtest semantics per level segment.</b> We fold each level's
          transitions from that level's first observed frame; on level-up
          transitions we check <code>is_goal</code> instead of the (unseen next
          level) grid. The post doesn't specify its exact treatment; this is
          our reading.</li>
        <li><b>No RHAE pipeline.</b> We track raw action counts against the
          eventual goal of one cleared game. Any numbers here are self-measured
          and not comparable to official leaderboard scores.</li>
      </ol>

      <h3>How to read this dashboard</h3>
      <ul>
        <li><b>Game</b> — the real environment, live; uncheck <i>follow
          live</i> to scrub back through every recorded frame.</li>
        <li><b>Play-by-play</b> — the run narrated from the structured event
          log: 🧠 model turns, ❌/✅ backtest verdicts with wrong-cell
          breakdowns, 🔍 BFS results, 🎮 executed actions (with SURPRISE /
          LEVEL UP flags).</li>
        <li><b>Agent notes</b> — the model's own persistent scratchpad; its
          hypotheses in its own words.</li>
        <li><b>World model</b> — the agent's current theory of the game as
          runnable code. Watching this file evolve — objects appearing,
          rules generalizing, representations being torn up after a
          counterexample — is the most interpretable view of the agent's
          understanding.</li>
      </ul>

      <h3>Status & caveats</h3>
      <p>
        Work in progress. The agent has not yet cleared a level; it is
        currently inducing the first game's mechanism (watch the backtest
        wrong-cell count trend down in the play-by-play). Everything is
        self-reported; the agent never sees the game's source code (the
        toolkit downloads it to run locally, but that directory is walled off
        from the agent). Expect the run to take many hours: one deliberation
        turn ≈ one local 35B generation.
      </p>
    </details>
  </div>

  <footer>
    Single Mac Studio (M3 Ultra, 96 GB) serving both the game and the model —
    qwen36 via mlx/vllm on the local backend slot. Harness, dashboard, and run
    artifacts: <a href="https://github.com/ursk/schema-qwen" style="color:inherit">github.com/ursk/schema-qwen</a>.
    After <a href="https://schema-harness.github.io/" style="color:inherit">Schema</a> (Impossible Research, 2026).
  </footer>
</div>
<script>
const PALETTE = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA",
                 "#F012BE","#FF851B","#7FDBFF","#870C25","#4B0082","#8B4513",
                 "#00FFFF","#556B2F","#FFC0CB","#FFFFFF"];
let events = [], cur = 0, playing = false;
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const scrub = document.getElementById("scrub"), meta = document.getElementById("meta");
const follow = document.getElementById("follow");
function draw(i) {
  if (!events.length) return;
  cur = Math.max(0, Math.min(i, events.length - 1));
  const ev = events[cur], g = ev.grid, s = 8;
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
    ctx.fillStyle = PALETTE[g[y][x] & 15];
    ctx.fillRect(x * s, y * s, s, s);
  }
  scrub.value = cur;
  meta.textContent = `step ${cur}/${events.length - 1} · action ${ev.action ?? "(initial)"} · ` +
                     `level ${ev.level} · ${ev.state}`;
}
function chips(s) {
  const bt = s.backtest === null ? ["–", ""] : s.backtest ? ["GREEN", "green"] : ["RED", "red"];
  document.getElementById("chips").innerHTML =
    `<span class="chip">game <b>${s.game ?? "?"}</b></span>` +
    `<span class="chip">model <b>${s.model ?? "?"}</b></span>` +
    `<span class="chip">level <b>${s.level ?? 0}/${s.win_levels ?? "?"}</b></span>` +
    `<span class="chip">actions <b>${s.actions ?? 0}</b></span>` +
    `<span class="chip">llm calls <b>${s.llm_calls ?? 0}</b></span>` +
    `<span class="chip ${bt[1]}">backtest <b>${bt[0]}</b></span>` +
    `<span class="chip">deliberation <b>${s.deliberation ?? 0}</b></span>`;
  document.getElementById("substat").textContent =
    s.finished ? `run finished: ${s.finished}` : "run live · updates every 1.5s";
  document.getElementById("dot").className = "dot" + (s.finished ? "" : " live");
}
scrub.oninput = () => { follow.checked = false; draw(+scrub.value); };
document.getElementById("play").onclick = () => {
  playing = !playing; follow.checked = false;
  document.getElementById("play").textContent = playing ? "⏸" : "▶";
  if (playing) tick();
};
function tick() {
  if (!playing) return;
  if (cur < events.length - 1) draw(cur + 1); else playing = false;
  document.getElementById("play").textContent = playing ? "⏸" : "▶";
  if (playing) setTimeout(tick, 150);
}
let feedCount = 0;
// works at / locally and under a reverse-proxy mount like /schema
const BASE = location.pathname.replace(/\\/$/, "");
function renderKv(kv) {
  const chips = document.getElementById("kvchips"), svg = document.getElementById("kvspark");
  if (!kv) { chips.textContent = " · backend metrics unavailable"; svg.innerHTML = ""; return; }
  chips.textContent = ` · KV ${kv.kv_gb} GB · Metal ${kv.metal_active_gb}/${kv.metal_peak_gb} GB peak · ${kv.entries} entries · hit ${Math.round(kv.hit_rate * 100)}%`;
  const H = kv.history || [];
  if (H.length < 2) { svg.innerHTML = ""; return; }
  const W = 480, HT = 40, maxG = Math.max(8, ...H.map(h => h.metal_gb));
  const pts = key => H.map((h, i) =>
    `${(i / (H.length - 1) * W).toFixed(1)},${(HT - 2 - (h[key] / maxG) * (HT - 6)).toFixed(1)}`).join(" ");
  svg.innerHTML =
    `<polyline points="${pts("metal_gb")}" fill="none" stroke="var(--muted)" stroke-width="1.5" opacity="0.7"/>` +
    `<polyline points="${pts("kv_gb")}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>`;
}
function rollout(live, turns) {
  const tps = document.getElementById("tps"), gs = document.getElementById("genstat");
  const fresh = live && (Date.now() / 1000 - live.updated) < 6;
  if (fresh && live.generating && live.phase === "prefill") {
    tps.textContent = "…";
    gs.textContent = `PREFILL · ~${((live.prompt_tokens||0)/1000).toFixed(1)}k prompt tok · ` +
      `${live.prefill_s}s so far` +
      (live.prefill_tok_s ? ` · ~${live.prefill_tok_s} tok/s if it finishes now` : "");
  } else if (fresh && live.generating) {
    tps.textContent = live.tok_s;
    gs.textContent = `decoding (${live.samples}× sampled) · ${live.tokens} tok · ` +
      `${live.seconds}s · prefill was ~${((live.prompt_tokens||0)/1000).toFixed(1)}k tok in ${live.prefill_s}s`;
  } else if (turns && turns.length) {
    const t = turns[turns.length - 1];
    tps.textContent = t.tok_s;
    gs.textContent = `idle · last turn: ${t.gen_tokens} tok in ${t.seconds}s (${t.samples}× sampled)`;
  } else { tps.textContent = "–"; gs.textContent = "waiting for first turn"; }
  if (!turns) return;
  const T = turns.slice(-60), W = 480, H = 60;
  const bw = W / Math.max(T.length, 20);
  // sub-bar per sample, scaled to the token cap: a capped sample touches the top
  document.getElementById("turnbars").innerHTML = T.map((t, i) => {
    const cap = t.max_tokens || 4096;
    const toks = (t.sample_tokens && t.sample_tokens.length)
      ? t.sample_tokens : [t.gen_tokens];
    const sw = (bw - 2) / toks.length;
    const fill = t.kind === "code" ? "var(--good)" : "var(--muted)";
    const title = `<title>d${t.deliberation} t${t.turn}: [${toks.join(", ")}] / ${cap} tok, ${t.kind}${t.anomalies.length ? " ⚠" + t.anomalies.join(",") : ""}</title>`;
    return toks.map((tk, j) => {
      const capped = tk >= cap - 2;
      const h = Math.max(2, Math.min(1, tk / cap) * (H - 4));
      const f = capped ? "var(--critical)" : fill;
      return `<rect x="${(i * bw + 1 + j * sw).toFixed(1)}" y="${(H - h - 2).toFixed(1)}" width="${Math.max(sw - 1, 1).toFixed(1)}" height="${h.toFixed(1)}" fill="${f}" opacity="0.8">${title}</rect>`;
    }).join("");
  }).join("");
  const maxWm = Math.max(1, ...T.map(t => t.wm_added + t.wm_removed));
  document.getElementById("wmbars").innerHTML = T.map((t, i) => {
    const ha = t.wm_added / maxWm * 18, hr = t.wm_removed / maxWm * 18;
    return `<rect x="${(i * bw + 1).toFixed(1)}" y="${(20 - ha).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${ha.toFixed(1)}" fill="var(--good)" opacity="0.75"></rect>` +
           `<rect x="${(i * bw + 1).toFixed(1)}" y="20" width="${(bw - 2).toFixed(1)}" height="${hr.toFixed(1)}" fill="var(--critical)" opacity="0.6"></rect>`;
  }).join("");
  const an = turns.filter(t => t.anomalies.length).slice(-8);
  renderKv(window._kv);
  document.getElementById("anoms").textContent = an.length
    ? an.map(t => `d${t.deliberation} turn ${t.turn}: ${t.anomalies.join(", ")}`).join("\\n")
    : "none";
}
async function poll() {
  try {
    const r = await fetch(`${BASE}/data?since=${events.length}&feed_since=${feedCount}`);
    const d = await r.json();
    document.getElementById("title").textContent = d.run;
    if (d.events.length) {
      events.push(...d.events);
      scrub.max = events.length - 1;
      if (follow.checked) draw(events.length - 1);
    }
    if (d.feed.length) {
      const el = document.getElementById("feed");
      el.textContent += d.feed.join("\\n") + "\\n";
      el.scrollTop = el.scrollHeight;
    }
    feedCount = d.feed_count;
    chips(d.stats);
    window._kv = d.kv;
    const gt = document.getElementById("gentail"), gl = document.getElementById("genliv");
    const atBottom = gt.scrollHeight - gt.scrollTop - gt.clientHeight < 30;
    gt.textContent = d.gen_tail || "";
    if (atBottom) gt.scrollTop = gt.scrollHeight;
    const liveNow = d.live && (Date.now() / 1000 - d.live.updated) < 6 && d.live.generating;
    gl.textContent = liveNow ? "· streaming" : "· idle";
    rollout(d.live, d.turns);
    document.getElementById("notes").textContent = d.notes;
    document.getElementById("wm").textContent = d.world_model;
  } catch (e) {}
  setTimeout(poll, 1500);
}
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        # route by suffix so any reverse-proxy mount prefix works (/schema, /pro, …)
        path = "/data" if u.path.rstrip("/").endswith("/data") or u.path == "/data" else "/"
        if path == "/":
            self._send(PAGE.encode(), "text/html")
        elif path == "/data":
            since = int(parse_qs(u.query).get("since", ["0"])[0])
            events = []
            tl = RUN_DIR / "timeline.jsonl"
            if tl.exists():
                with open(tl) as f:
                    for i, line in enumerate(f):
                        if i >= since and line.strip():
                            e = json.loads(line)
                            events.append({k: e[k] for k in ("action", "grid", "level", "state")})
            feed_since = int(parse_qs(u.query).get("feed_since", ["0"])[0])
            feed, feed_count = [], 0
            stats = {"backtest": None, "llm_calls": 0, "actions": 0,
                     "level": 0, "finished": None}
            turns = []
            ej = RUN_DIR / "events.jsonl"
            if ej.exists():
                with open(ej) as f:
                    lines = [ln for ln in f if ln.strip()]
                feed_count = len(lines)
                for i, ln in enumerate(lines):
                    try:
                        e = json.loads(ln)
                    except Exception:
                        continue
                    k, d = e.get("kind"), e.get("data", {})
                    if k == "start":
                        stats["game"] = d.get("game")
                        stats["model"] = d.get("model")
                        stats["win_levels"] = d.get("win_levels")
                    elif k == "llm":
                        stats["llm_calls"] += 1
                    elif k == "backtest":
                        stats["backtest"] = bool(d.get("ok"))
                    elif k == "progress":
                        stats["deliberation"] = d.get("deliberation")
                        stats["level"] = d.get("level")
                        stats["actions"] = d.get("actions")
                    elif k == "win":
                        stats["finished"] = "WIN"
                    elif k == "stop":
                        stats["finished"] = d.get("reason")
                    if k == "turn":
                        turns.append(d)
                    if i >= feed_since:
                        h = humanize(e)
                        if h:
                            feed.append(h)
            # timeline is fresher than the last progress event
            tlp = RUN_DIR / "timeline.jsonl"
            if tlp.exists():
                n_actions, last = 0, None
                with open(tlp) as f:
                    for ln in f:
                        if ln.strip():
                            last = json.loads(ln)
                            if last["action"] is not None:
                                n_actions += 1
                if last is not None:
                    stats["actions"] = n_actions
                    stats["level"] = last["level"]
            live = None
            lp = RUN_DIR / "live.json"
            if lp.exists():
                try:
                    live = json.loads(lp.read_text())
                except Exception:
                    live = None
            gen_tail = ""
            tp = RUN_DIR / "trace.log"
            if tp.exists():
                with open(tp, "rb") as f:
                    f.seek(max(0, tp.stat().st_size - 2500))
                    gen_tail = f.read().decode("utf-8", "replace")
            def read(name):
                p = RUN_DIR / name
                return p.read_text() if p.exists() else "(none yet)"
            body = json.dumps({
                "run": RUN_DIR.name, "events": events,
                "feed": feed, "feed_count": feed_count, "stats": stats,
                "turns": turns[-80:], "live": live, "kv": poll_backend_metrics(),
                "gen_tail": gen_tail,
                "notes": read("notes.md"), "world_model": read("world_model.py"),
            }).encode()
            self._send(body, "application/json")
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global RUN_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    RUN_DIR = Path(args.run).resolve()
    print(f"viewer for {RUN_DIR} on http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
