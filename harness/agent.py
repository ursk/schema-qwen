"""Deliberation loop: talk to the LLM, parse protocol commands, drive tools."""

import json
import re
import time
from pathlib import Path

import httpx

from .grids import grid_to_text, diff_summary
from .prompts import SYSTEM, action_semantics
from .vision import describe as vision_describe, flow as vision_flow
from .sandbox import run_worker

CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.S)
COMMIT_RE = re.compile(r"^\s*COMMIT:\s*(.+?)\s*$", re.M)
COMMIT_PLAN_RE = re.compile(r"^\s*COMMIT PLAN\s*$", re.M)
PLAN_RE = re.compile(r"^\s*PLAN\s*$", re.M)
NOTE_RE = re.compile(r"^\s*NOTE:\s*(.+?)\s*$", re.M)
GOAL_RE = re.compile(r"^\s*GOAL:\s*(.+?)\s*$", re.M)
ANALYZE_RE = re.compile(r"^\s*ANALYZE\s*$", re.M)
ACTION_TOKEN_RE = re.compile(r"^(RESET|[123457]|6@\d{1,2},\d{1,2})$")


class LLM:
    def __init__(self, base_url, model, max_tokens=4096, temperature=0.6):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = httpx.Client(timeout=600)
        self.calls = 0

    def chat(self, messages, on_delta=None, seed=None, temperature=None):
        """Returns {"text", "chunks", "retries"}. chunks ~ generated tokens."""
        self.calls += 1
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            # Qwen3-family thinking-mode guidance: temp 0.6, and raise
            # presence_penalty toward 1.5 for quantized models when endless
            # repetition occurs. vis2 2026-07-19 hit exactly that (three
            # straight turns of full-budget repetition, loop guard firing,
            # zero commits) with every anti-repetition sampler at default 0.
            "presence_penalty": 1.5,
            "stream": True,
            # thinking ON (2026-07-19, Urs): it's a reasoning model — let it
            # reason. History renders think-stripped (template default), so
            # the cache re-prefills only turn N's stripped reply each turn.
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if seed is not None:
            payload["seed"] = seed
        last_err = None
        for attempt in range(5):
            try:
                text, chunks = self._stream_once(payload, on_delta)
                if chunks == 0:
                    # gptoss120's serialized route rejects overlapping requests
                    # with a clean-but-empty stream ("route is busy") — retryable,
                    # never a real reply
                    last_err = RuntimeError("empty stream (backend busy?)")
                    if on_delta:
                        on_delta(f"\n[empty stream, retry {attempt + 1}/5]\n")
                    time.sleep(20 * (attempt + 1))
                    continue
                # gpt-oss via vllm-mlx streaming leaks harmony channel markers
                # into content ("analysis...assistantfinal<reply>") — the
                # reasoning parser only runs on the non-stream path. Keep only
                # the final channel; a no-op for models without the marker.
                if "assistantfinal" in text:
                    text = text.split("assistantfinal")[-1]
                # backends without a reasoning parser leave <think> inline in
                # content — strip closed blocks; an unclosed block means the
                # whole tail is reasoning (budget hit), drop it too
                if "<think>" in text:
                    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
                    text = text.split("<think>")[0]
                return {"text": text, "chunks": chunks, "retries": attempt}
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                # a 400 with chat_template_kwargs set is usually a template that
                # doesn't know enable_thinking (mistral) — drop it and retry now
                if (isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code == 400
                        and "chat_template_kwargs" in payload):
                    payload = {k: v for k, v in payload.items()
                               if k != "chat_template_kwargs"}
                    continue
                # local backend may crash and get relaunched by launchd — wait it out
                last_err = e
                if on_delta:
                    on_delta(f"\n[llm error, retry {attempt + 1}/5: {type(e).__name__}]\n")
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"LLM unreachable after 5 attempts: {last_err}")

    def chat_n(self, messages, n, on_deltas=None):
        """n concurrent samples (the backend batches them). on_deltas(i, chunk).

        Returns a list of n result dicts (None where a sample failed hard).

        Distinct per-sample seeds + a small temperature ladder: the backend
        seeds identically per request otherwise, and n identical samples make
        best-of-N a no-op (observed: sample_tokens [5658, 5658, 5658, 5658]).

        Sample 0 launches alone and the rest wait for its first generated
        token: n simultaneous identical prompts each prefill their own KV
        (3/4 of requests were guaranteed prefix-cache misses); staggered,
        samples 1..n-1 hit the prefix sample 0 just computed.
        """
        import random
        import threading

        base_seed = random.randrange(1 << 30)
        results = [None] * n
        warmed = threading.Event()
        def work(i):
            def cb(chunk, i=i):
                if i == 0:
                    warmed.set()
                if on_deltas:
                    on_deltas(i, chunk)
            try:
                results[i] = self.chat(
                    messages, cb,
                    seed=base_seed + i,
                    temperature=min(1.0, self.temperature + 0.1 * i),
                )
            except Exception:
                results[i] = None
            finally:
                if i == 0:
                    warmed.set()
        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        threads[0].start()
        warmed.wait(timeout=180)  # prefill of a ~10k-token prompt, with margin
        for t in threads[1:]:
            t.start()
        for t in threads:
            t.join()
        return results

    def _stream_once(self, payload, on_delta):
        """Returns (reply_text, generated_token_count).

        Reasoning deltas (reasoning_content) stream to on_delta (trace
        visibility) and count toward generation/loop detection, but are NOT
        part of the returned reply text — a COMMIT inside a think trace must
        never execute. Backends that leave <think> inline in content are
        handled by the strip in chat()."""
        content = []
        n_total = 0
        combined = []  # content + reasoning, for loop detection
        with self.client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                reason = delta.get("reasoning_content") or ""
                text = delta.get("content") or ""
                if not reason and not text:
                    continue
                n_total += 1
                if reason:
                    combined.append(reason)
                    if on_delta:
                        on_delta(reason)
                if text:
                    content.append(text)
                    combined.append(text)
                    if on_delta:
                        on_delta(text)
                if n_total % 32 == 0 and _is_looping("".join(combined)):
                    msg = "\n[TRUNCATED BY HARNESS: your output was repeating in a loop]"
                    if on_delta:
                        on_delta(msg + "\n")
                    return "".join(content) + msg, n_total
        return "".join(content), n_total


class ClaudeCLI:
    """LLM adapter that shells out to headless Claude Code (`claude -p`),
    the same way the nightly Opus jobs fire — no API key needed.

    Tool use is disabled and cwd is an empty sandbox so the model cannot read
    the downloaded game sources. No streaming: on_delta gets the whole reply.
    """

    def __init__(self, model="opus", max_tokens=8192, temperature=None):
        import subprocess
        import tempfile
        self._subprocess = subprocess
        self.model = model
        self.max_tokens = max_tokens  # informational; CC manages its own budget
        self.calls = 0
        self.sandbox = tempfile.mkdtemp(prefix="schema-cc-")

    @staticmethod
    def _flatten(messages):
        parts = []
        for m in messages:
            role = {"system": "SYSTEM INSTRUCTIONS", "user": "HARNESS",
                    "assistant": "YOUR PREVIOUS REPLY"}[m["role"]]
            parts.append(f"=== {role} ===\n{m['content']}")
        parts.append("Answer directly in plain text following the system instructions. "
                     "Do not use any tools.")
        return "\n\n".join(parts)

    QUOTA_RE = re.compile(r"session limit|usage limit|rate limit", re.I)

    def chat(self, messages, on_delta=None):
        self.calls += 1
        for attempt in range(24):
            try:
                proc = self._subprocess.run(
                    ["claude", "-p", "--model", self.model,
                     "--output-format", "text", "--max-turns", "1",
                     "--disallowedTools",
                     "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,Agent,NotebookEdit"],
                    input=self._flatten(messages), capture_output=True,
                    text=True, cwd=self.sandbox, timeout=1200,
                )
                text = proc.stdout.strip()
                if text and self.QUOTA_RE.search(text[:300]):
                    # plan quota exhausted — wait it out instead of burning turns
                    if on_delta:
                        on_delta(f"\n[claude quota exhausted ({text[:80]!r}) — "
                                 f"sleeping 20 min, attempt {attempt + 1}/24]\n")
                    time.sleep(1200)
                    continue
                if text:
                    if on_delta:
                        on_delta(text)
                    return {"text": text, "chunks": max(1, len(text) // 4),
                            "retries": attempt}
            except self._subprocess.TimeoutExpired:
                pass
            time.sleep(10 * (attempt + 1))
        return {"text": "", "chunks": 0, "retries": 24}

    def chat_n(self, messages, n, on_deltas=None):
        return [
            self.chat(messages, (lambda c, i=i: on_deltas(i, c)) if on_deltas else None)
            for i in range(n)
        ]


class HumanCLI:
    """Blind-play adapter: prints the exact situation the models get to the
    terminal and reads the reply from stdin — same protocol, same constraints.
    End each reply with a line containing only: GO
    """

    def __init__(self):
        self.calls = 0
        self.max_tokens = 1 << 30  # never flag token_budget anomalies
        self._sys_shown = False

    def chat(self, messages, on_delta=None):
        self.calls += 1
        if not self._sys_shown:
            print("\n" + "#" * 78 + "\n# SYSTEM PROMPT\n" + "#" * 78)
            print(messages[0]["content"])
            self._sys_shown = True
        print("\n" + "#" * 78 + "\n# HARNESS — reply below, end with a line: GO\n" + "#" * 78)
        print(messages[-1]["content"])
        lines = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if ln.strip() == "GO":
                break
            lines.append(ln)
        text = "\n".join(lines)
        if on_delta:
            on_delta(text)
        return {"text": text, "chunks": max(1, len(text) // 4), "retries": 0}

    def chat_n(self, messages, n, on_deltas=None):
        # a human is always samples=1
        return [self.chat(messages, (lambda c: on_deltas(0, c)) if on_deltas else None)]


def _is_looping(text, min_repeats=4):
    """Detect degenerate repetition: the tail is the same 1-4 line block repeated."""
    lines = [ln for ln in text.splitlines() if ln.strip()][-24:]
    for period in (1, 2, 3, 4):
        if len(lines) < period * min_repeats:
            continue
        block = lines[-period:]
        if all(
            lines[-(k + 1) * period : len(lines) - k * period or None] == block
            for k in range(min_repeats)
        ):
            return True
    # newline-free enumeration runaway ("... for y=105 ... for y=106 ..."):
    # a 32-char shingle repeated >=10x in the last 3000 chars. Threshold is high
    # so legitimately repetitive code (grid-drawing loops) doesn't trip it.
    tail = text[-3000:]
    if len(tail) >= 2000:
        counts = {}
        for j in range(0, len(tail) - 32):
            sh = tail[j:j + 32]
            counts[sh] = counts.get(sh, 0) + 1
            if counts[sh] >= 10:
                return True
    return False


class Agent:
    def __init__(self, env, timeline, llm, run_dir: Path, log,
                 max_deliberation_turns=14, context_char_budget=60000, samples=4,
                 system=SYSTEM, vision=False):
        self.system = system
        self.vision = vision
        self.env = env
        self.timeline = timeline
        self.llm = llm
        self.run_dir = Path(run_dir)
        self.model_path = self.run_dir / "world_model.py"
        self.notes_path = self.run_dir / "notes.md"
        self.trace_path = self.run_dir / "trace.log"
        self._trace_f = open(self.trace_path, "a", buffering=1)
        self.deliberation_no = 0
        self.samples = max(1, samples)
        self._gen_lock = __import__("threading").Lock()
        self._gen_tokens = 0
        self._gen_t0 = 0.0
        self._live_last = 0.0
        self._last_wm_delta = None  # (added, removed) lines from last code write
        self.log = log
        self.max_turns = max_deliberation_turns
        self.char_budget = context_char_budget
        self.last_plan = None
        self.backtest_green = False
        self.recent_events = []  # (action, summary) since last context rebuild
        self.best_path = self.run_dir / "world_model_best.py"
        self.best_score = None  # lowest total_wrong_cells seen
        self.best_scored_len = -1  # action_count when best_score was computed
        self.last_score = None  # score of the currently saved world_model.py
        if self.best_path.exists() and self.timeline.events:
            # resumed run: re-score the saved best so it isn't clobbered
            self._rescore_best()
        if self.model_path.exists() and self.timeline.events:
            # resumed run: evaluate the current model so a green backtest
            # (and PLAN availability) survives the restart
            rep = run_worker("backtest", self.model_path, self.timeline.path)
            self.backtest_green = (bool(rep.get("ok"))
                                   and rep.get("transitions_checked", 0) > 0)
            (self.run_dir / "backtest.json").write_text(json.dumps(rep))
            if rep.get("total_wrong_cells") is not None:
                self.last_score = rep["total_wrong_cells"]

    def _rescore_best(self):
        """Champion scores are history-relative — re-score when history grew.

        Without this, 'WORSE than your best (X vs Y)' compares a fresh score
        against one computed on a shorter timeline (observed: champion '182'
        was actually 668 on the full history).
        """
        if not self.best_path.exists():
            return
        if self.best_scored_len == self.timeline.action_count:
            return
        rep = run_worker("backtest", self.best_path, self.timeline.path)
        if rep.get("ok"):
            self.best_score = 0
        elif rep.get("total_wrong_cells") is not None:
            self.best_score = rep["total_wrong_cells"]
        self.best_scored_len = self.timeline.action_count

    def trace(self, text):
        self._trace_f.write(text)

    # ---------- live generation stats ----------

    def _live_write(self, generating):
        now = time.time()
        elapsed = max(now - self._gen_t0, 1e-6)
        first = getattr(self, "_gen_first_t", None)
        prompt_toks = getattr(self, "_prompt_chars", 0) // 4
        if generating and first is None:
            phase = "prefill"
            prefill_s = elapsed
        else:
            phase = "decode" if generating else "idle"
            prefill_s = max((first or self._gen_t0) - self._gen_t0, 1e-6)
        decode_s = max(now - (first or now), 1e-6)
        try:
            (self.run_dir / "live.json").write_text(json.dumps({
                "generating": generating,
                "phase": phase,
                "tokens": self._gen_tokens,
                "seconds": round(elapsed, 1),
                # decode speed measured from first token, not request start —
                # otherwise a slow prefill masquerades as slow decoding
                "tok_s": round(self._gen_tokens / decode_s, 1) if first else 0.0,
                "prompt_tokens": prompt_toks,
                "prefill_s": round(prefill_s, 1),
                "prefill_tok_s": round(prompt_toks / prefill_s, 0) if prompt_toks else 0,
                "samples": self.samples,
                "updated": now,
            }))
        except OSError:
            pass

    def _on_deltas(self, i, chunk):
        with self._gen_lock:
            if getattr(self, "_gen_first_t", None) is None:
                self._gen_first_t = time.time()
            # streamed deltas are ~1 token; the CC adapter delivers the whole
            # reply as one chunk, so estimate by length there
            self._gen_tokens += 1 if len(chunk) <= 8 else max(1, len(chunk) // 4)
            now = time.time()
            throttle = now - self._live_last > 0.5
            if throttle:
                self._live_last = now
        if i == 0:
            self.trace(chunk)
        if throttle:
            self._live_write(True)

    # ---------- context ----------

    def notes(self):
        return self.notes_path.read_text() if self.notes_path.exists() else "(empty)"

    def goal(self):
        p = self.run_dir / "goal.md"
        return p.read_text().strip() if p.exists() else ""

    def world_model(self):
        return self.model_path.read_text() if self.model_path.exists() else None

    def situation(self, extra=""):
        # ordered stable -> volatile so the backend's prefix cache survives
        # across deliberations: static header, then append-only notes, then the
        # slow-changing world model, then everything that changes every turn.
        cur = self.timeline.events[-1]
        goal = self.goal()
        parts = [
            action_semantics(self.env.available_actions),
            f"YOUR NOTES (notes.md, append-only):\n{self.notes()}",
            "YOUR CURRENT GOAL HYPOTHESIS (rewrite with a `GOAL: ...` line "
            "whenever evidence changes it):\n"
            + (goal or "(none stated yet — every game has a win condition; state "
                       "your best guess with a GOAL: line and design probes to test it)"),
        ]
        wm = self.world_model()
        if wm:
            parts.append(f"YOUR WORLD MODEL (world_model.py):\n```python\n{wm}```")
        else:
            parts.append("YOU HAVE NO WORLD MODEL YET.")
        if self.recent_events:
            lines = []
            recent = self.recent_events[-20:]
            for j, (a, s) in enumerate(recent):
                # vision flow summaries are multi-line; keep full change maps
                # only for the last few, older ones shrink to head + MOVEMENT
                if "\n" in s and len(recent) - j > 6:
                    head = []
                    for ln in s.splitlines():
                        if "change map" in ln:
                            break
                        head.append(ln)
                    s = "\n".join(head)
                if "\n" in s:
                    lines.append(f"  after {a}:\n" + "\n".join(
                        "    " + ln for ln in s.splitlines()))
                else:
                    lines.append(f"  after {a}: {s}")
            parts.append("RECENT TRANSITIONS (newest last):\n" + "\n".join(lines))
        bt = ("no world model yet" if not wm else
              "GREEN (reproduces all recorded transitions)" if self.backtest_green
              else "RED (has mismatches — fix before planning)")
        parts.append(
            f"GAME STATUS: level {self.env.level}/{self.env.win_levels} · "
            f"{self.env.state} · {self.timeline.action_count} actions taken so far · "
            f"backtest {bt}"
        )
        parts.append(f"CURRENT GRID (hex colors, x -> right, y -> down):\n{grid_to_text(cur['grid'])}")
        if self.vision:
            parts.append("OBJECTS (connected-block decomposition of the current "
                         f"grid, computed by the harness):\n{vision_describe(cur['grid'])}")
        if extra:
            parts.append(extra)
        parts.append("Decide your next command.")
        return "\n\n".join(parts)

    # ---------- tools ----------

    def do_write_model(self, code):
        old = self.world_model()
        if old is not None and code.strip() == old.strip():
            return ("You resubmitted the SAME world model — nothing changed, and nothing will. "
                    "Only include a python code block when you are CHANGING the model. "
                    "If you want to run an experiment, reply with ONLY a COMMIT line, "
                    "e.g. `COMMIT: 2`.")
        if old is not None:
            import difflib
            ops = difflib.SequenceMatcher(
                None, old.splitlines(), code.splitlines()).get_opcodes()
            added = sum(j2 - j1 for op, i1, i2, j1, j2 in ops if op in ("insert", "replace"))
            removed = sum(i2 - i1 for op, i1, i2, j1, j2 in ops if op in ("delete", "replace"))
            self._last_wm_delta = (added, removed)
        else:
            self._last_wm_delta = (len(code.splitlines()), 0)
        self.model_path.write_text(code)
        rep = run_worker("backtest", self.model_path, self.timeline.path)
        # a green on zero transitions is vacuous — it must not unlock PLAN
        self.backtest_green = bool(rep.get("ok")) and rep.get("transitions_checked", 0) > 0
        # the full structured report goes to disk for ANALYZE (`backtest`
        # variable); the event log and the prompt only ever carry aggregates —
        # per-cell values reach the model exclusively through code it writes
        (self.run_dir / "backtest.json").write_text(json.dumps(rep))
        self.log("backtest", {k: rep.get(k) for k in (
            "ok", "transitions_checked", "n_mismatches", "total_wrong_cells",
            "n_goal_misses", "n_bad_cells", "error")})
        if rep.get("ok") and not self.backtest_green:
            return ("world_model.py saved, but there are NO recorded transitions yet — "
                    "the backtest is vacuous and proves nothing. Probe reality first "
                    "(`COMMIT: <action>`).")
        if self.backtest_green:
            # green converts into search immediately: run BFS unprompted so
            # is_goal is confronted with reachability every single time
            plan_report = self.do_plan()
            return (f"world_model.py saved. backtest GREEN: "
                    f"{rep.get('transitions_checked', 0)} recorded transitions reproduced exactly.\n"
                    f"AUTO-PLAN (the harness runs BFS whenever your model goes green):\n"
                    f"{plan_report}")
        if "error" in rep:
            return f"world_model.py saved, but backtest FAILED to run:\n{rep['error']}"
        score = rep.get("total_wrong_cells")
        self.last_score = score
        self._rescore_best()
        score_note = ""
        if score is not None:
            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self.best_scored_len = self.timeline.action_count
                self.best_path.write_text(code)
                score_note = f"This is your BEST model so far ({score} wrong cells total)."
            else:
                # neutral report — prescribing "Say REVERT" here trapped an
                # instruction-compliant model in a code->revert loop whenever
                # its champion was a degenerate identity model
                score_note = (f"This scores {score} wrong cells; your best-scoring model "
                              f"so far scored {self.best_score}.")
        mm = rep.get("mismatches", [])
        lines = [f"world_model.py saved. backtest RED: {rep.get('n_mismatches')} mismatching "
                 f"transitions out of {rep.get('transitions_checked')}. {score_note} First mismatches:"]
        n_bad = rep.get("n_bad_cells")

        def bbox(pts):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return f"x[{min(xs)}..{max(xs)}] y[{min(ys)}..{max(ys)}]"

        if n_bad and n_bad <= 16 and not rep.get("n_goal_misses"):
            lines.append(
                f"HINT: all mismatches are confined to just {n_bad} distinct cell(s) "
                f"inside {bbox(rep.get('bad_cells', []))}. Whatever lives there (a "
                f"counter? a HUD glyph?) is deterministic and CAN be modeled — decode "
                f"it from the recorded history to reach GREEN. It may even encode the goal."
            )
        # regions and counts only — never per-cell values (they tempt the
        # model into hand-counting grid text; ground truth lives in ANALYZE)
        for m in mm[:8]:
            if m.get("kind") == "grid":
                where = f" in {bbox(m['cells'])}" if m.get("cells") else ""
                lines.append(
                    f"- step {m.get('step_i')} (action {m.get('action')}): "
                    f"{m.get('n_cells')} wrong cells{where} — "
                    f"{m.get('over', 0)} you changed but reality didn't, "
                    f"{m.get('missed', 0)} reality changed but you didn't, "
                    f"{m.get('wrong', 0)} both changed differently")
            else:
                lines.append(f"- step {m.get('step_i')} (action {m.get('action')}): {m.get('detail')}")
        lines.append(
            "No cell values are listed on purpose — do NOT re-read the grid text to "
            "recover them. Run ANALYZE and print what you need: the full mismatch "
            "array is available there as `backtest` "
            "(mismatches[i]['cells'] = [[x, y, was, predicted, real], ...]).")
        lines.append("Revise init_state/step/render (or is_goal) to explain these, then resubmit.")
        if self.timeline.action_count < 12:
            lines.append(
                f"NOTE FROM HARNESS: only {self.timeline.action_count} real transitions are "
                f"recorded. With this little data, a short probe (`COMMIT: <action>`) usually "
                f"constrains the rule far more than another rewrite. Try each untested action once."
            )
        return "\n".join(lines)

    def do_analyze(self, code):
        """Run agent-written analysis code read-only over the timeline."""
        p = self.run_dir / "_analysis.py"
        p.write_text(code)
        rep = run_worker("analyze", p, self.timeline.path)
        self.log("analyze", {"ok": rep.get("ok"),
                             "stdout_chars": len(rep.get("stdout") or "")})
        out = rep.get("stdout") or ""
        if "stdout" not in rep:  # worker crash / timeout, code never ran
            return f"ANALYZE failed: {rep.get('error', 'unknown worker failure')}"
        tail = f"\n[your code raised]\n{rep['error']}" if rep.get("error") else ""
        if not out and not tail:
            out = "(no output — use print() to see results)"
        return "ANALYZE stdout:\n" + out + tail

    def do_plan(self):
        if not self.world_model():
            return "No world model yet — write one first."
        if not self.backtest_green:
            return "Backtest is RED — a plan inside a wrong model is worthless. Fix the model first."
        args = {"actions": [a for a in self.env.available_actions if a in {"1", "2", "3", "4", "5", "7"}]}
        rep = run_worker("bfs", self.model_path, self.timeline.path, args)
        self.log("bfs", rep)
        if rep.get("ok") and rep.get("plan") is not None:
            self.last_plan = rep["plan"]
            return (f"BFS found a goal in {len(rep['plan'])} action(s) "
                    f"(expanded {rep.get('expanded', '?')} nodes, {rep.get('distinct_states', '?')} states): "
                    f"{' '.join(rep['plan']) or '(empty — already at goal)'}\n"
                    f"Reply `COMMIT PLAN` to execute, or refine the model.")
        return (f"BFS found NO goal (expanded {rep.get('expanded', '?')} nodes"
                f", {rep.get('distinct_states', '?')} distinct states). "
                f"{rep.get('error', rep.get('note', ''))}\n"
                "Either is_goal is wrong, a mechanism is missing from step(), or the goal "
                "needs clicks (define candidate_clicks). Probe reality to find out.")

    def do_commit(self, actions):
        """Execute actions; with a green model, per-step prediction check."""
        predicted = None
        if self.backtest_green and "RESET" not in actions:
            rep = run_worker("predict", self.model_path, self.timeline.path, {"plan": actions})
            if rep.get("ok"):
                predicted = rep
        out = []
        start_level = self.env.level
        for i, a in enumerate(actions):
            before = self.timeline.events[-1]["grid"]
            ev = self.env.act(a)
            summ = (vision_flow(before, ev["grid"]) if self.vision
                    else diff_summary(before, ev["grid"]))
            self.recent_events.append((a, summ))
            leveled = ev["level"] > start_level
            if ev["state"] == "WIN":
                out.append(f"[{i+1}/{len(actions)}] {a}: *** WIN — game complete! ***")
                break
            if leveled:
                out.append(f"[{i+1}/{len(actions)}] {a}: *** LEVEL UP -> level {ev['level']} *** (new level grid shown below)")
                self.backtest_green = False  # new level: model unverified against it
                self.last_plan = None
                break
            if ev["state"] == "GAME_OVER":
                out.append(f"[{i+1}/{len(actions)}] {a}: GAME OVER — level restarts via RESET. Remaining plan discarded.")
                self.last_plan = None
                break
            out.append(f"[{i+1}/{len(actions)}] {a}: {summ}")
            if predicted is not None:
                pred_grid = predicted["grids"][i]
                if pred_grid != ev["grid"]:
                    bad = sum(
                        1 for y in range(64) for x in range(64)
                        if pred_grid[y][x] != ev["grid"][y][x]
                    )
                    out.append(f"SURPRISE: reality differs from your model's prediction on this "
                               f"step ({bad} cells). Plan aborted. This transition is now in the "
                               f"timeline — fix the model so the backtest is green again.")
                    self.backtest_green = False
                    self.last_plan = None
                    break
        result = "\n".join(out)
        self.log("commit", {"actions": actions, "result": result})
        return result

    # ---------- best-of-N ----------

    def _select_candidate(self, results):
        """Pick the reply to adopt. Code candidates are backtested; best wins.

        Returns (reply_text, kind, payload, bestofn_log_or_None).
        """
        alive = [(i, r) for i, r in enumerate(results) if r and r["text"].strip()]
        if not alive:
            return "", "none", None, None
        parsed = [(i, r, *self.parse(r["text"], apply_notes=False)) for i, r in alive]
        coded = [(i, r, k, p) for i, r, k, p in parsed if k == "code"]
        if len(coded) >= 2:
            scores = {}
            for i, r, k, code in coded:
                cand = self.run_dir / f"_candidate_{i}.py"
                cand.write_text(code)
                rep = run_worker("backtest", cand, self.timeline.path)
                cand.unlink(missing_ok=True)
                if rep.get("ok"):
                    scores[i] = 0
                elif rep.get("total_wrong_cells") is not None:
                    scores[i] = rep["total_wrong_cells"]
                else:
                    scores[i] = float("inf")  # crashed / no JSON
            win = min(scores, key=scores.get)
            i, r, k, p = next(c for c in coded if c[0] == win)
            if win != alive[0][0]:
                self.trace(f"\n[best-of-{len(results)}: adopted sample {win} "
                           f"(score {scores[win]}) over streamed sample; all: {scores}]\n"
                           + r["text"] + "\n")
            self.parse(r["text"])  # apply NOTEs of the adopted reply
            return r["text"], k, p, {
                "n": len(results), "code_candidates": len(coded),
                "scores": {str(a): (None if b == float("inf") else b) for a, b in scores.items()},
                "adopted": win,
            }
        # no (or one) code candidate: prefer the first reply with a real command
        for i, r, k, p in parsed:
            if k not in ("none", "error"):
                if i != alive[0][0]:
                    self.trace(f"\n[adopted sample {i} — first with a valid command]\n"
                               + r["text"] + "\n")
                self.parse(r["text"])
                return r["text"], k, p, None
        i, r, k, p = parsed[0]
        self.parse(r["text"])
        return r["text"], k, p, None

    def _turn_anomalies(self, results, reply, kind):
        out = []
        alive = [r for r in results if r]
        if any("TRUNCATED BY HARNESS" in r["text"] for r in alive):
            out.append("loop_abort")
        if any(r["chunks"] >= self.llm.max_tokens - 2 for r in alive):
            out.append("token_budget")
        if any(r["retries"] > 0 for r in alive):
            out.append("llm_retry")
        if len(alive) < len(results):
            out.append("sample_failed")
        if kind == "none":
            out.append("no_command")
        return out

    def _log_turn(self, turn, seconds, kind, anomalies, results, result_text=""):
        if "timed out" in result_text:
            anomalies = anomalies + ["worker_timeout"]
        with self._gen_lock:
            toks = self._gen_tokens
        self.log("turn", {
            "deliberation": self.deliberation_no, "turn": turn,
            "seconds": round(seconds, 1), "gen_tokens": toks,
            "tok_s": round(toks / max(seconds, 1e-6), 1),
            "samples": self.samples, "kind": kind,
            "sample_tokens": [r["chunks"] if r else 0 for r in results],
            "max_tokens": self.llm.max_tokens,
            "prompt_chars": getattr(self, "_prompt_chars", 0),
            "wm_added": self._last_wm_delta[0] if self._last_wm_delta else 0,
            "wm_removed": self._last_wm_delta[1] if self._last_wm_delta else 0,
            "anomalies": anomalies,
        })

    # ---------- parsing ----------

    def parse(self, text, apply_notes=True):
        """Return (kind, payload) for the single command in the reply."""
        if apply_notes:
            for note in NOTE_RE.findall(text):
                with open(self.notes_path, "a") as f:
                    f.write(note + "\n")
            goals = GOAL_RE.findall(text)
            if goals:  # GOAL replaces (a hypothesis is revised, not appended to)
                (self.run_dir / "goal.md").write_text(goals[-1] + "\n")
        code_blocks = CODE_RE.findall(text)
        # an ANALYZE line claims the code block for read-only analysis, not the model
        if code_blocks and ANALYZE_RE.search(text):
            return "analyze", code_blocks[-1]
        # an explicit COMMIT outranks a code block that doesn't change the model
        # (weak models love pasting the current model back while narrating a probe)
        if code_blocks:
            wm = self.world_model()
            changed = wm is None or code_blocks[-1].strip() != wm.strip()
            if changed or not (COMMIT_RE.search(text) or COMMIT_PLAN_RE.search(text)):
                return "code", code_blocks[-1]
        if COMMIT_PLAN_RE.search(text):
            return "commit_plan", None
        m = COMMIT_RE.search(text)
        if m:
            toks = self._rejoin_clicks(m.group(1))
            bad = [t for t in toks if not ACTION_TOKEN_RE.match(t)]
            if bad:
                return "error", f"COMMIT contained invalid action tokens: {bad}. Valid: 1-5, 7, RESET, 6@x,y"
            return "commit", toks
        if PLAN_RE.search(text):
            return "plan", None
        if re.search(r"^\s*REVERT\s*$", text, re.M):
            return "revert", None
        return "none", None

    @staticmethod
    def _rejoin_clicks(s):
        return [t for t in re.split(r"[\s;]+", s.strip()) if t]

    # ---------- main loop ----------

    def deliberate(self, opening_extra=""):
        """One deliberation: converse until a COMMIT executes or turns run out."""
        self.deliberation_no += 1
        # champion score is history-relative — refresh it before any comparison
        self._rescore_best()
        # a deliberation starts from the best-verified theory, not the last experiment
        if (self.best_path.exists() and self.best_score is not None
                and (self.last_score is None or self.last_score > self.best_score)):
            self.model_path.write_text(self.best_path.read_text())
            self.last_score = self.best_score
            opening_extra = (opening_extra + "\n" if opening_extra else "") + (
                f"(world_model.py was restored to your best-scoring version, "
                f"{self.best_score} wrong cells — later revisions scored worse.)")
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.situation(opening_extra)},
        ]
        consecutive_no_command = 0
        for turn in range(self.max_turns):
            self.trace(
                f"\n\n{'═' * 78}\n═ deliberation {self.deliberation_no} · turn {turn + 1} · "
                f"level {self.env.level}/{self.env.win_levels} · "
                f"{self.timeline.action_count} actions taken\n{'═' * 78}\n[model]\n"
            )
            with self._gen_lock:
                self._gen_tokens = 0
                self._gen_first_t = None
                self._gen_t0 = time.time()
            self._prompt_chars = sum(len(m["content"]) for m in messages)
            self._live_write(True)
            # ticker keeps live.json fresh during prefill, when no deltas flow
            import threading
            stop_tick = threading.Event()
            def _tick():
                while not stop_tick.wait(2.0):
                    self._live_write(True)
            ticker = threading.Thread(target=_tick, daemon=True)
            ticker.start()
            try:
                results = self.llm.chat_n(messages, self.samples, on_deltas=self._on_deltas)
            finally:
                stop_tick.set()
            self._live_write(False)
            seconds = time.time() - self._gen_t0
            reply, kind, payload, bestofn = self._select_candidate(results)
            self.log("llm", {"turn": turn, "reply": reply[-3000:]})
            if bestofn:
                self.log("bestofn", {"turn": turn, **bestofn})
            anomalies = self._turn_anomalies(results, reply, kind)
            self._last_wm_delta = None
            if kind == "code":
                result = self.do_write_model(payload)
            elif kind == "analyze":
                result = self.do_analyze(payload)
            elif kind == "plan":
                result = self.do_plan()
            elif kind == "commit_plan":
                if not self.last_plan:
                    result = "There is no stored plan. Run PLAN first."
                else:
                    result = self.do_commit(self.last_plan)
                    self.trace(f"\n\n[harness — executed in game]\n{result}\n")
                    self._log_turn(turn, seconds, kind, anomalies, results, result)
                    return result
            elif kind == "commit":
                result = self.do_commit(payload)
                self.trace(f"\n\n[harness — executed in game]\n{result}\n")
                self._log_turn(turn, seconds, kind, anomalies, results, result)
                return result
            elif kind == "revert":
                if self.best_path.exists():
                    result = self.do_write_model(self.best_path.read_text())
                else:
                    result = "There is no saved best model to revert to."
            elif kind == "error":
                result = payload
            else:
                result = ("I could not find a command in your reply. End with a python code "
                          "block, PLAN, COMMIT PLAN, or COMMIT: <actions>.")
            # commandless spiral: repeated no-command turns never recover in-context
            # (observed: 5 straight turns of enumeration prose). End the deliberation
            # so the outer loop restarts with a fresh situation + probe nudge.
            consecutive_no_command = consecutive_no_command + 1 if kind == "none" else 0
            if consecutive_no_command >= 3:
                self.trace("\n\n[harness]\n3 commandless turns — ending deliberation, "
                           "fresh context next.\n")
                self._log_turn(turn, seconds, kind, anomalies, results, result)
                self.log("deliberation", {"result": "no-command spiral break"})
                return None
            self.trace(f"\n\n[harness]\n{result}\n")
            self._log_turn(turn, seconds, kind, anomalies, results, result)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": result + "\n\nDecide your next command."})
            if sum(len(m["content"]) for m in messages) > self.char_budget:
                # compress: keep system, drop middle, rebuild situation
                messages = [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": self.situation(
                        "Context was compacted. Your notes and world model above are the "
                        "durable state; recent tool result:\n" + result)},
                ]
        # ran out of turns without committing — force a note and end deliberation
        self.log("deliberation", {"result": "no commit in max turns"})
        return None
