"""Deliberation loop: talk to the LLM, parse protocol commands, drive tools."""

import json
import re
import time
from pathlib import Path

import httpx

from .grids import grid_to_text, diff_summary
from .prompts import SYSTEM, action_semantics
from .sandbox import run_worker

CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.S)
COMMIT_RE = re.compile(r"^\s*COMMIT:\s*(.+?)\s*$", re.M)
COMMIT_PLAN_RE = re.compile(r"^\s*COMMIT PLAN\s*$", re.M)
PLAN_RE = re.compile(r"^\s*PLAN\s*$", re.M)
NOTE_RE = re.compile(r"^\s*NOTE:\s*(.+?)\s*$", re.M)
ACTION_TOKEN_RE = re.compile(r"^(RESET|[123457]|6@\d{1,2},\d{1,2})$")


class LLM:
    def __init__(self, base_url, model, max_tokens=4096, temperature=0.7):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = httpx.Client(timeout=600)
        self.calls = 0

    def chat(self, messages, on_delta=None):
        """Returns {"text", "chunks", "retries"}. chunks ~ generated tokens."""
        self.calls += 1
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = None
        for attempt in range(5):
            try:
                text, chunks = self._stream_once(payload, on_delta)
                return {"text": text, "chunks": chunks, "retries": attempt}
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                # local backend may crash and get relaunched by launchd — wait it out
                last_err = e
                if on_delta:
                    on_delta(f"\n[llm error, retry {attempt + 1}/5: {type(e).__name__}]\n")
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"LLM unreachable after 5 attempts: {last_err}")

    def chat_n(self, messages, n, on_deltas=None):
        """n concurrent samples (the backend batches them). on_deltas(i, chunk).

        Returns a list of n result dicts (None where a sample failed hard).
        """
        import threading

        results = [None] * n
        def work(i):
            try:
                results[i] = self.chat(
                    messages,
                    (lambda c, i=i: on_deltas(i, c)) if on_deltas else None,
                )
            except Exception:
                results[i] = None
        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def _stream_once(self, payload, on_delta):
        chunks = []
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
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if not delta:
                    continue
                chunks.append(delta)
                if on_delta:
                    on_delta(delta)
                if len(chunks) % 32 == 0 and _is_looping("".join(chunks)):
                    msg = "\n[TRUNCATED BY HARNESS: your output was repeating in a loop]"
                    if on_delta:
                        on_delta(msg + "\n")
                    return "".join(chunks) + msg, len(chunks)
        return "".join(chunks), len(chunks)


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
    return False


class Agent:
    def __init__(self, env, timeline, llm, run_dir: Path, log,
                 max_deliberation_turns=14, context_char_budget=60000, samples=4):
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
        self.last_score = None  # score of the currently saved world_model.py
        if self.best_path.exists() and self.timeline.events:
            # resumed run: re-score the saved best so it isn't clobbered
            rep = run_worker("backtest", self.best_path, self.timeline.path)
            if rep.get("ok"):
                self.best_score = 0
            elif rep.get("total_wrong_cells") is not None:
                self.best_score = rep["total_wrong_cells"]

    def trace(self, text):
        self._trace_f.write(text)

    # ---------- live generation stats ----------

    def _live_write(self, generating):
        elapsed = max(time.time() - self._gen_t0, 1e-6)
        try:
            (self.run_dir / "live.json").write_text(json.dumps({
                "generating": generating,
                "tokens": self._gen_tokens,
                "seconds": round(elapsed, 1),
                "tok_s": round(self._gen_tokens / elapsed, 1),
                "samples": self.samples,
                "updated": time.time(),
            }))
        except OSError:
            pass

    def _on_deltas(self, i, chunk):
        with self._gen_lock:
            self._gen_tokens += 1
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

    def world_model(self):
        return self.model_path.read_text() if self.model_path.exists() else None

    def situation(self, extra=""):
        cur = self.timeline.events[-1]
        parts = [
            f"GAME STATUS: level {self.env.level}/{self.env.win_levels} · "
            f"{self.env.state} · {self.timeline.action_count} actions taken so far",
            action_semantics(self.env.available_actions),
            f"YOUR NOTES (notes.md):\n{self.notes()}",
        ]
        wm = self.world_model()
        if wm:
            bt = "GREEN (reproduces all recorded transitions)" if self.backtest_green \
                else "RED (has mismatches — fix before planning)"
            parts.append(f"YOUR WORLD MODEL (world_model.py) — backtest {bt}:\n```python\n{wm}```")
        else:
            parts.append("YOU HAVE NO WORLD MODEL YET.")
        if self.recent_events:
            lines = [f"  after {a}: {s}" for a, s in self.recent_events[-20:]]
            parts.append("RECENT TRANSITIONS (newest last):\n" + "\n".join(lines))
        parts.append(f"CURRENT GRID (hex colors, x -> right, y -> down):\n{grid_to_text(cur['grid'])}")
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
        self.backtest_green = bool(rep.get("ok"))
        self.log("backtest", rep)
        if self.backtest_green:
            return (f"world_model.py saved. backtest GREEN: "
                    f"{rep.get('transitions_checked', 0)} recorded transitions reproduced exactly. "
                    f"You may PLAN now.")
        if "error" in rep:
            return f"world_model.py saved, but backtest FAILED to run:\n{rep['error']}"
        score = rep.get("total_wrong_cells")
        self.last_score = score
        score_note = ""
        if score is not None:
            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self.best_path.write_text(code)
                score_note = f"This is your BEST model so far ({score} wrong cells total)."
            else:
                score_note = (f"WORSE than your best model ({score} vs {self.best_score} wrong "
                              f"cells). Say REVERT to restore the best one.")
        mm = rep.get("mismatches", [])
        lines = [f"world_model.py saved. backtest RED: {rep.get('n_mismatches')} mismatching "
                 f"transitions out of {rep.get('transitions_checked')}. {score_note} First mismatches:"]
        for m in mm:
            if m.get("kind") == "grid":
                lines.append(f"- step {m.get('step_i')} (action {m.get('action')}): "
                             f"{m.get('n_cells')} wrong cells ({m.get('breakdown', '')}); "
                             + "; ".join(m.get("cells", [])[:6]))
            else:
                lines.append(f"- step {m.get('step_i')} (action {m.get('action')}): {m.get('detail')}")
        lines.append("Revise init_state/step/render (or is_goal) to explain these, then resubmit.")
        if self.timeline.action_count < 12:
            lines.append(
                f"NOTE FROM HARNESS: only {self.timeline.action_count} real transitions are "
                f"recorded. With this little data, a short probe (`COMMIT: <action>`) usually "
                f"constrains the rule far more than another rewrite. Try each untested action once."
            )
        return "\n".join(lines)

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
            summ = diff_summary(before, ev["grid"])
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

    def _log_turn(self, turn, seconds, kind, anomalies, result_text=""):
        if "timed out" in result_text:
            anomalies = anomalies + ["worker_timeout"]
        with self._gen_lock:
            toks = self._gen_tokens
        self.log("turn", {
            "deliberation": self.deliberation_no, "turn": turn,
            "seconds": round(seconds, 1), "gen_tokens": toks,
            "tok_s": round(toks / max(seconds, 1e-6), 1),
            "samples": self.samples, "kind": kind,
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
        code_blocks = CODE_RE.findall(text)
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
        # a deliberation starts from the best-verified theory, not the last experiment
        if (self.best_path.exists() and self.best_score is not None
                and (self.last_score is None or self.last_score > self.best_score)):
            self.model_path.write_text(self.best_path.read_text())
            self.last_score = self.best_score
            opening_extra = (opening_extra + "\n" if opening_extra else "") + (
                f"(world_model.py was restored to your best-scoring version, "
                f"{self.best_score} wrong cells — later revisions scored worse.)")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.situation(opening_extra)},
        ]
        for turn in range(self.max_turns):
            self.trace(
                f"\n\n{'═' * 78}\n═ deliberation {self.deliberation_no} · turn {turn + 1} · "
                f"level {self.env.level}/{self.env.win_levels} · "
                f"{self.timeline.action_count} actions taken\n{'═' * 78}\n[model]\n"
            )
            with self._gen_lock:
                self._gen_tokens = 0
                self._gen_t0 = time.time()
            self._live_write(True)
            results = self.llm.chat_n(messages, self.samples, on_deltas=self._on_deltas)
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
            elif kind == "plan":
                result = self.do_plan()
            elif kind == "commit_plan":
                if not self.last_plan:
                    result = "There is no stored plan. Run PLAN first."
                else:
                    result = self.do_commit(self.last_plan)
                    self.trace(f"\n\n[harness — executed in game]\n{result}\n")
                    self._log_turn(turn, seconds, kind, anomalies, result)
                    return result
            elif kind == "commit":
                result = self.do_commit(payload)
                self.trace(f"\n\n[harness — executed in game]\n{result}\n")
                self._log_turn(turn, seconds, kind, anomalies, result)
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
            self.trace(f"\n\n[harness]\n{result}\n")
            self._log_turn(turn, seconds, kind, anomalies, result)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": result + "\n\nDecide your next command."})
            if sum(len(m["content"]) for m in messages) > self.char_budget:
                # compress: keep system, drop middle, rebuild situation
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": self.situation(
                        "Context was compacted. Your notes and world model above are the "
                        "durable state; recent tool result:\n" + result)},
                ]
        # ran out of turns without committing — force a note and end deliberation
        self.log("deliberation", {"result": "no commit in max turns"})
        return None
