"""Append-only ground-truth record of every real environment transition."""

import json
from pathlib import Path


class Timeline:
    """Each event: {i, action, grid, level, state, full_reset}.

    - action: None for the initial observation of a segment, else a string
      "1".."5", "RESET", or "6@x,y".
    - grid: the 64x64 grid observed AFTER the action (last animation layer).
    - level: levels_completed reported by the env at that moment.
    - state: "NOT_FINISHED" | "WIN" | "GAME_OVER".
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.events = []
        if self.path.exists():
            with open(self.path) as f:
                self.events = [json.loads(line) for line in f if line.strip()]

    def append(self, action, grid, level, state, full_reset=False):
        ev = {
            "i": len(self.events),
            "action": action,
            "grid": grid,
            "level": level,
            "state": state,
            "full_reset": full_reset,
        }
        self.events.append(ev)
        with open(self.path, "a") as f:
            f.write(json.dumps(ev) + "\n")
        return ev

    def segments(self):
        """Split events into level segments.

        A segment starts at an initial observation (action None / RESET) or
        right after a level-up, and contains (start_grid, [(action, grid,
        level, state), ...]). Backtest folds each segment independently:
        state = init_state(start_grid), then step() through the actions.
        """
        segs = []
        cur = None
        prev_level = None
        for ev in self.events:
            boundary = (
                cur is None
                or ev["action"] is None
                or ev["action"] == "RESET"
                or ev["full_reset"]
                or (prev_level is not None and ev["level"] > prev_level)
            )
            if boundary:
                # a level-up event still belongs to the OLD segment (the
                # transition that caused it), then opens a new one
                if (
                    cur is not None
                    and ev["action"] not in (None, "RESET")
                    and not ev["full_reset"]
                ):
                    cur["steps"].append(ev)
                cur = {"start": ev, "steps": []}
                segs.append(cur)
            else:
                cur["steps"].append(ev)
            prev_level = ev["level"]
        return segs

    @property
    def action_count(self):
        """Real actions taken (RESETs count per official rules)."""
        return sum(1 for ev in self.events if ev["action"] is not None)
