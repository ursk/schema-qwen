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
    </div>
    <div>
      <div class="card"><h2>Play-by-play</h2><pre class="log" id="feed"></pre></div>
      <div class="card"><h2>Agent notes (notes.md)</h2><pre class="log" id="notes" style="max-height:170px"></pre></div>
      <div class="card"><h2>World model (world_model.py)</h2><pre class="log" id="wm" style="max-height:240px"></pre></div>
    </div>
  </div>
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
        # tolerate a reverse-proxy mount prefix (e.g. tailscale serve /schema)
        path = u.path
        if path.startswith("/schema"):
            path = path[len("/schema"):] or "/"
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
            def read(name):
                p = RUN_DIR / name
                return p.read_text() if p.exists() else "(none yet)"
            body = json.dumps({
                "run": RUN_DIR.name, "events": events,
                "feed": feed, "feed_count": feed_count, "stats": stats,
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
