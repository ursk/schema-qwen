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


def mismatch_digest(mismatches, max_lines=14):
    """Deterministic auto-analysis of RED backtests: wrong cells grouped into
    connected blocks with a 'color story' each, plus cross-step recurrence and
    movement. The eye-level tabulation a competent ANALYZE call would print —
    pushed by the harness because the model reads what it's handed but never
    initiates tool use. Never emits raw per-cell dumps.
    """
    from collections import Counter, defaultdict

    def components(cells):
        pts = {(x, y): (b, p, r) for x, y, b, p, r in cells}
        seen, comps = set(), []
        for xy in pts:
            if xy in seen:
                continue
            stack, comp = [xy], []
            seen.add(xy)
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if nb in pts and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            comps.append(comp)
        return pts, comps

    def summarize(pts, comp):
        xs = [x for x, y in comp]
        ys = [y for x, y in comp]
        story = Counter()
        for xy in comp:
            b, p, r = pts[xy]
            if r == b:
                story[f"you painted {p}, reality kept {b}"] += 1
            elif p == b:
                story[f"you kept {b}, reality painted {r}"] += 1
            else:
                story[f"was {b}: you painted {p}, reality painted {r}"] += 1
        return {"n": len(comp), "w": max(xs) - min(xs) + 1, "h": max(ys) - min(ys) + 1,
                "x": min(xs), "y": min(ys), "story": story.most_common(1)[0][0]}

    lines, occs = [], []
    for m in mismatches:
        if m.get("kind") != "grid" or not m.get("cells"):
            continue
        pts, comps = components([tuple(c) for c in m["cells"]])
        digs = sorted((summarize(pts, c) for c in comps), key=lambda d: -d["n"])
        blocks = "; ".join(
            f"{d['w']}x{d['h']} block at ({d['x']},{d['y']}): {d['story']}"
            for d in digs[:4])
        extra = f" (+{len(digs) - 4} smaller blocks)" if len(digs) > 4 else ""
        lines.append(f"step {m['step_i']} (action {m['action']}): "
                     f"{m['n_cells']} wrong cells — {blocks}{extra}")
        for d in digs[:6]:
            occs.append((m.get("action"), m["step_i"], d))

    groups = defaultdict(list)
    for a, si, d in occs:
        groups[(a, d["w"], d["h"], d["story"])].append((si, d["x"], d["y"]))
    for (a, w, h, story), oc in sorted(groups.items(),
                                       key=lambda kv: -len(kv[1])):
        if len(oc) < 2:
            continue
        oc.sort()
        dxs = {oc[i + 1][1] - oc[i][1] for i in range(len(oc) - 1)}
        dys = {oc[i + 1][2] - oc[i][2] for i in range(len(oc) - 1)}
        move = (f", top-left moving ({dxs.pop():+d},{dys.pop():+d}) per occurrence"
                if len(dxs) == 1 and len(dys) == 1 and (dxs != {0} or dys != {0})
                else f", at x={[x for _, x, _ in oc]} y={[y for _, _, y in oc]}")
        lines.append(f"PATTERN: the same {w}x{h} block ({story}) recurs in "
                     f"{len(oc)} mismatched action-{a} steps{move}")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"...[{len(lines) - max_lines} more digest lines omitted]"]
    return lines


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
        "digest": mismatch_digest(all_mismatches[:64]),
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


def crop_text(grid, x0, y0, x1, y1):
    """Raw cells for a SMALL region (<=16x16) as coordinate-labeled text —
    the only sanctioned raw-cell view. Ask for exactly the region you need;
    full-grid text is banned because positions in long runs can't be counted.
    """
    from harness import vision

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(63, x1), min(63, y1)
    if x1 < x0 or y1 < y0:
        return "crop refused: empty region"
    if x1 - x0 + 1 > 16 or y1 - y0 + 1 > 16:
        return "crop refused: keep the region <= 16x16 (that is the point)"
    head = "y\\x " + " ".join(f"{x:2d}" for x in range(x0, x1 + 1))
    rows = [head] + [
        f"{y:3d} " + " ".join(f"{vision.HEX[grid[y][x] & 15]:>2}"
                              for x in range(x0, x1 + 1))
        for y in range(y0, y1 + 1)
    ]
    return "\n".join(rows)


def _helper_globals():
    """The shared code-space perception surface for ANALYZE and sense.py."""
    from harness import vision

    return {
        "components": vision.components,
        "describe": vision.describe,
        "flow": vision.flow,
        "crop": crop_text,
        "HEX": vision.HEX,
    }


def cmd_sense(code_path, timeline_path):
    """Run the model-OWNED perception module: sense(events) -> str.

    The harness calls this every turn; its output is the model's entire
    view of the board. Crash -> {"ok": False, "error": ...} and the agent
    falls back to the last working version.
    """
    import contextlib
    import io

    with open(timeline_path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    with open(code_path) as f:
        code = f.read()
    g = _helper_globals()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "sense.py", "exec"), g)
            if "sense" not in g or not callable(g["sense"]):
                return {"ok": False, "error": "sense.py must define sense(events) -> str"}
            out = str(g["sense"](events))
    except Exception:
        return {"ok": False, "error": traceback.format_exc(limit=4)}
    if len(out) > 5000:
        out = out[:5000] + "\n...[sense output truncated at 5000 chars — keep your view compact]"
    return {"ok": True, "text": out}


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
        **_helper_globals(),
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
        if cmd == "sense":
            # perception module needs no world model — model_path is sense.py
            print(json.dumps(cmd_sense(model_path, timeline_path)))
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
