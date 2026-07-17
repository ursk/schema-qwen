"""No-LLM plumbing test: synthetic 'move the dot to the flag' game.

Mechanic: a dot (color 3) on a 64x64 field of 0s; action 1/2/3/4 moves it
up/down/left/right. Goal cell at (10, 5) marked with color 5. Level completes
when the dot reaches the flag.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.sandbox import run_worker
from harness.timeline import Timeline

DOT, FLAG = 3, 5
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def make_grid(dot, flag=(10, 5)):
    g = [[0] * 64 for _ in range(64)]
    g[flag[1]][flag[0]] = FLAG
    g[dot[1]][dot[0]] = DOT
    return g


def build_timeline(path):
    tl = Timeline(path)
    pos = (5, 5)
    tl.append(None, make_grid(pos), 0, "NOT_FINISHED")
    for a in [4, 4, 2, 1, 4]:  # wander a bit (ends at (8,5))
        dx, dy = MOVES[a]
        pos = (pos[0] + dx, pos[1] + dy)
        tl.append(str(a), make_grid(pos), 0, "NOT_FINISHED")
    return tl, pos


GOOD_MODEL = '''
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

def _find(grid, color):
    for y in range(64):
        for x in range(64):
            if grid[y][x] == color:
                return (x, y)
    return None

def init_state(grid):
    return {"dot": _find(grid, 3), "flag": _find(grid, 5)}

def step(state, action):
    if isinstance(action, tuple):
        return state
    dx, dy = MOVES.get(action, (0, 0))
    x, y = state["dot"]
    return {"dot": (max(0, min(63, x + dx)), max(0, min(63, y + dy))), "flag": state["flag"]}

def render(state):
    g = [[0] * 64 for _ in range(64)]
    fx, fy = state["flag"]
    g[fy][fx] = 5
    x, y = state["dot"]
    g[y][x] = 3
    return g

def is_goal(state):
    return state["dot"] == state["flag"]
'''

BAD_MODEL = GOOD_MODEL.replace('{1: (0, -1), 2: (0, 1)', '{1: (0, 1), 2: (0, -1)')  # up/down swapped


def main(tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    tl_path = tmp / "timeline.jsonl"
    if tl_path.exists():
        tl_path.unlink()
    tl, pos = build_timeline(tl_path)

    good, bad = tmp / "good.py", tmp / "bad.py"
    good.write_text(GOOD_MODEL)
    bad.write_text(BAD_MODEL)

    r = run_worker("backtest", good, tl_path)
    assert r["ok"] and r["transitions_checked"] == 5, r
    print("backtest green on correct model:", r)

    r = run_worker("backtest", bad, tl_path)
    # up/down swap: '2' mispredicts, then '1' cancels the error — 1 mismatch
    assert not r["ok"] and r["n_mismatches"] == 1, r
    assert r["mismatches"][0]["kind"] == "grid" and r["mismatches"][0]["n_cells"] == 2, r
    print("backtest red on wrong model:", json.dumps(r["mismatches"][0])[:160])

    r = run_worker("bfs", good, tl_path, {"actions": ["1", "2", "3", "4"]})
    assert r["ok"] and len(r["plan"]) == 2, r  # (8,5) -> (10,5): right right
    print("bfs plan:", r["plan"])

    r = run_worker("predict", good, tl_path, {"plan": r["plan"]})
    assert r["ok"] and r["goals"] == [False, True], r
    print("predict ok, goal on last step")

    hang = tmp / "hang.py"
    hang.write_text(GOOD_MODEL + "\nimport time\ntime.sleep(999)\n")
    import harness.sandbox as sb
    sb.TIMEOUTS["backtest"] = 5
    r = run_worker("backtest", hang, tl_path)
    assert not r["ok"] and "timed out" in r["error"], r
    print("timeout handled:", r["error"])

    print("ALL PLUMBING TESTS PASSED")


if __name__ == "__main__":
    main(ROOT / "runs" / "_plumbing_test")
