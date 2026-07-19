"""Subprocess worker: runs model-written world_model.py against the timeline.

Invoked as:  python -m harness.worker <cmd> <world_model.py> <timeline.jsonl> [json-args]
cmd: backtest | bfs | predict
Prints a single JSON object to stdout. Run by sandbox.py with a timeout;
never imported by the agent process (model code may hang or crash).
"""

import json
import resource
import sys
import traceback
from collections import deque


def load_model(path):
    ns = {}
    with open(path) as f:
        code = f.read()
    exec(compile(code, "world_model.py", "exec"), ns)
    for fn in ("init_state", "step", "render", "is_goal"):
        if fn not in ns:
            raise RuntimeError(f"world_model.py must define {fn}()")
    return ns


def parse_action(a):
    """'1'..'5' -> int; '6@x,y' -> (6, x, y). RESET never reaches step()."""
    if a.startswith("6@"):
        x, y = a[2:].split(",")
        return (6, int(x), int(y))
    return int(a)


def state_key(ns, state):
    try:
        return json.dumps(state, sort_keys=True, default=repr)
    except Exception:
        return repr(state)


def fold_segment(ns, seg, check=True, max_report=64):
    """Fold a timeline segment through the model.

    Returns (final_state, report) where report lists mismatches:
    {step_i, action, kind, detail}. Grid comparison is skipped on level-up
    steps (the frame already shows the next level) — is_goal is checked
    there instead.
    """
    state = ns["init_state"]([row[:] for row in seg["start"]["grid"]])
    mismatches = []
    bad_xy = set()
    checked = 0
    prev_level = seg["start"]["level"]
    prev_grid = seg["start"]["grid"]
    for ev in seg["steps"]:
        state = ns["step"](state, parse_action(ev["action"]))
        if not check:
            prev_level = ev["level"]
            continue
        checked += 1
        leveled = ev["level"] > prev_level or ev["state"] == "WIN"
        if leveled:
            try:
                ok = bool(ns["is_goal"](state))
            except Exception as e:
                ok = False
            if not ok:
                mismatches.append({
                    "step_i": ev["i"], "action": ev["action"], "kind": "goal",
                    "detail": "level completed here in reality, but is_goal(state) is not True",
                })
        else:
            pred = ns["render"](state)
            real = ev["grid"]
            if pred != real:
                bad = []
                over = missed = wrong = 0
                for y in range(64):
                    if pred[y] != real[y]:
                        for x in range(64):
                            if pred[y][x] != real[y][x]:
                                b = prev_grid[y][x]
                                bad.append((x, y, b, pred[y][x], real[y][x]))
                                bad_xy.add((x, y))
                                if real[y][x] == b:
                                    over += 1     # model changed a cell reality left alone
                                elif pred[y][x] == b:
                                    missed += 1   # reality changed a cell model left alone
                                else:
                                    wrong += 1    # both changed it, differently
                if len(mismatches) < max_report:
                    mismatches.append({
                        "step_i": ev["i"], "action": ev["action"], "kind": "grid",
                        "n_cells": len(bad),
                        "over": over, "missed": missed, "wrong": wrong,
                        # structured [x, y, was, predicted, real] for the
                        # ANALYZE `backtest` variable — never rendered as
                        # per-cell text into the prompt (pixel values in
                        # context tempt the model into counting characters)
                        "cells": [list(c) for c in bad[:400]],
                    })
                else:
                    mismatches.append({"step_i": ev["i"], "kind": "grid", "n_cells": len(bad)})
        prev_level = ev["level"]
        prev_grid = ev["grid"]
    return state, mismatches, checked, bad_xy


def cmd_backtest(ns, segments):
    total_checked, all_mismatches = 0, []
    all_bad_xy = set()
    for seg in segments:
        _, mm, checked, bad_xy = fold_segment(ns, seg)
        total_checked += checked
        all_mismatches.extend(mm)
        all_bad_xy |= bad_xy
    return {
        "ok": len(all_mismatches) == 0,
        "transitions_checked": total_checked,
        "mismatches": all_mismatches[:64],
        "n_mismatches": len(all_mismatches),
        "total_wrong_cells": sum(m.get("n_cells", 0) for m in all_mismatches)
        + sum(50 for m in all_mismatches if m.get("kind") == "goal"),
        "n_goal_misses": sum(1 for m in all_mismatches if m.get("kind") == "goal"),
        "n_bad_cells": len(all_bad_xy),
        "bad_cells": sorted(all_bad_xy)[:64],
    }


def cmd_bfs(ns, segments, args):
    """BFS inside the model from the current state (end of last segment)."""
    actions = [parse_action(a) for a in args.get("actions", ["1", "2", "3", "4"])]
    max_nodes = int(args.get("max_nodes", 150000))
    max_depth = int(args.get("max_depth", 120))
    seg = segments[-1]
    start, _, _, _ = fold_segment(ns, seg, check=False)

    def expand(state):
        acts = list(actions)
        if "candidate_clicks" in ns:
            try:
                acts += [(6, int(x), int(y)) for x, y in ns["candidate_clicks"](state)]
            except Exception:
                pass
        return acts

    if ns["is_goal"](start):
        return {"ok": True, "plan": [], "note": "already at goal"}
    seen = {state_key(ns, start)}
    q = deque([(start, [])])
    nodes = 0
    while q and nodes < max_nodes:
        state, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for a in expand(state):
            nodes += 1
            try:
                s2 = ns["step"](state, a)
            except Exception:
                continue
            k = state_key(ns, s2)
            if k in seen:
                continue
            seen.add(k)
            p2 = path + [a]
            try:
                if ns["is_goal"](s2):
                    plan = [f"6@{a[1]},{a[2]}" if isinstance(a, tuple) else str(a) for a in p2]
                    return {"ok": True, "plan": plan, "expanded": nodes,
                            "distinct_states": len(seen)}
            except Exception:
                pass
            q.append((s2, p2))
    return {"ok": False, "plan": None, "expanded": nodes, "distinct_states": len(seen),
            "note": "no goal state found within search limits"}


def cmd_analyze(code_path, timeline_path):
    """Run agent-written analysis code READ-ONLY over the timeline; return stdout.

    A REPL over the agent's own memory: no game internals, no world-model
    coupling — the code sees exactly what the agent has already observed,
    plus the vision helpers so it can compute shapes instead of eyeballing hex.
    """
    import contextlib
    import io
    import os

    from harness import vision

    with open(timeline_path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    with open(code_path) as f:
        code = f.read()
    # latest backtest report, structured — the ONLY route to per-cell
    # mismatch ground truth (reports in the prompt carry aggregates only)
    backtest = None
    bt_file = os.path.join(os.path.dirname(timeline_path), "backtest.json")
    if os.path.exists(bt_file):
        try:
            with open(bt_file) as f:
                backtest = json.load(f)
        except Exception:
            pass
    g = {
        "events": events,
        "backtest": backtest,
        "components": vision.components,
        "describe": vision.describe,
        "flow": vision.flow,
        "HEX": vision.HEX,
    }
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "analysis.py", "exec"), g)
    except Exception:
        err = traceback.format_exc(limit=4)
    out = buf.getvalue()
    if len(out) > 6000:
        out = out[:6000] + "\n...[stdout truncated at 6000 chars]"
    return {"ok": err is None, "stdout": out, **({"error": err} if err else {})}


def cmd_predict(ns, segments, args):
    """Predict grids for a plan from the current state (for execution checks)."""
    plan = args["plan"]
    seg = segments[-1]
    state, _, _, _ = fold_segment(ns, seg, check=False)
    grids, goals = [], []
    for a in plan:
        if a == "RESET":
            return {"ok": False, "error": "cannot predict through RESET"}
        state = ns["step"](state, parse_action(a))
        grids.append(ns["render"](state))
        goals.append(bool(ns["is_goal"](state)))
    return {"ok": True, "grids": grids, "goals": goals}


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (110, 115))
    cmd, model_path, timeline_path = sys.argv[1], sys.argv[2], sys.argv[3]
    args = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    sys.path.insert(0, ".")
    from harness.timeline import Timeline

    try:
        if cmd == "analyze":
            # analysis code needs no world model — model_path is the code file
            print(json.dumps(cmd_analyze(model_path, timeline_path)))
            return
        ns = load_model(model_path)
        segments = Timeline(timeline_path).segments()
        if not segments:
            print(json.dumps({"ok": False, "error": "timeline is empty"}))
            return
        if cmd == "backtest":
            out = cmd_backtest(ns, segments)
        elif cmd == "bfs":
            out = cmd_bfs(ns, segments, args)
        elif cmd == "predict":
            out = cmd_predict(ns, segments, args)
        else:
            out = {"ok": False, "error": f"unknown cmd {cmd}"}
    except Exception:
        out = {"ok": False, "error": traceback.format_exc(limit=6)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
