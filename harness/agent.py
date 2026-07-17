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

    def chat(self, messages):
        self.calls += 1
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = None
        for attempt in range(5):
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                # local backend may crash and get relaunched by launchd — wait it out
                last_err = e
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"LLM unreachable after 5 attempts: {last_err}")


class Agent:
    def __init__(self, env, timeline, llm, run_dir: Path, log,
                 max_deliberation_turns=14, context_char_budget=60000):
        self.env = env
        self.timeline = timeline
        self.llm = llm
        self.run_dir = Path(run_dir)
        self.model_path = self.run_dir / "world_model.py"
        self.notes_path = self.run_dir / "notes.md"
        self.log = log
        self.max_turns = max_deliberation_turns
        self.char_budget = context_char_budget
        self.last_plan = None
        self.backtest_green = False
        self.recent_events = []  # (action, summary) since last context rebuild

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
        mm = rep.get("mismatches", [])
        lines = [f"world_model.py saved. backtest RED: {rep.get('n_mismatches')} mismatching "
                 f"transitions out of {rep.get('transitions_checked')}. First mismatches:"]
        for m in mm:
            if m.get("kind") == "grid":
                lines.append(f"- step {m.get('step_i')} (action {m.get('action')}): "
                             f"{m.get('n_cells')} wrong cells; " + "; ".join(m.get("cells", [])[:6]))
            else:
                lines.append(f"- step {m.get('step_i')} (action {m.get('action')}): {m.get('detail')}")
        lines.append("Revise init_state/step/render (or is_goal) to explain these, then resubmit.")
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
        return "\n".join(out)

    # ---------- parsing ----------

    def parse(self, text):
        """Return (kind, payload) for the single command in the reply."""
        for note in NOTE_RE.findall(text):
            with open(self.notes_path, "a") as f:
                f.write(note + "\n")
        code_blocks = CODE_RE.findall(text)
        if code_blocks:
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
        return "none", None

    @staticmethod
    def _rejoin_clicks(s):
        return [t for t in re.split(r"[\s;]+", s.strip()) if t]

    # ---------- main loop ----------

    def deliberate(self, opening_extra=""):
        """One deliberation: converse until a COMMIT executes or turns run out."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.situation(opening_extra)},
        ]
        for turn in range(self.max_turns):
            reply = self.llm.chat(messages)
            self.log("llm", {"turn": turn, "reply": reply[-3000:]})
            kind, payload = self.parse(reply)
            if kind == "code":
                result = self.do_write_model(payload)
            elif kind == "plan":
                result = self.do_plan()
            elif kind == "commit_plan":
                if not self.last_plan:
                    result = "There is no stored plan. Run PLAN first."
                else:
                    result = self.do_commit(self.last_plan)
                    return result
            elif kind == "commit":
                result = self.do_commit(payload)
                return result
            elif kind == "error":
                result = payload
            else:
                result = ("I could not find a command in your reply. End with a python code "
                          "block, PLAN, COMMIT PLAN, or COMMIT: <actions>.")
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
