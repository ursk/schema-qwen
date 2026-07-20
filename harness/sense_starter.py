"""Your perception module — these are YOUR eyes.

The harness calls sense(events) every turn and shows you ONLY its return
string (capped ~5000 chars). events = the full recorded history:
[{i, action, grid, level, state}, ...] — "grid" is the 64x64 int grid AFTER
that event's action (first entry: the initial frame, action None).

Helpers available as globals (same as ANALYZE): describe(grid),
flow(before, after), components(grid), crop(grid, x0, y0, x1, y1), HEX.

This file is code YOU own. If your current view hides something you need
(a texture, a counter region, diagonal structure), rewrite it with a SENSE
command. Keep it fast and the output compact — it is your entire view.
"""


def sense(events):
    cur = events[-1]["grid"]
    parts = ["OBJECTS (connected-block decomposition of the current frame):",
             describe(cur)]  # noqa: F821 — injected by the harness
    acted = [i for i, e in enumerate(events) if e["action"] is not None][-3:]
    if acted:
        parts.append("RECENT TRANSITIONS (newest last):")
        for i in acted:
            parts.append(f"after {events[i]['action']}:")
            parts.append(flow(events[i - 1]["grid"], events[i]["grid"]))  # noqa: F821
    return "\n".join(parts)
