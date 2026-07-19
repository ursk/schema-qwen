"""Object-level perception: block decomposition + sparse optic flow.

Deterministic preprocessing of the agent's OWN observations — no game
internals are read. Everything here is computed from the same grids the
model already sees. Still a documented deviation from the published Schema
tool list (which feeds raw grids + cell-list diffs only): it moves part of
the state-grounding work into the harness.
"""

HEX = "0123456789abcdef"


def components(grid):
    """4-connected same-color components: list of (color, [(x, y), ...])."""
    seen = [[False] * 64 for _ in range(64)]
    comps = []
    for y in range(64):
        for x in range(64):
            if seen[y][x]:
                continue
            c = grid[y][x]
            seen[y][x] = True
            stack = [(x, y)]
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < 64 and 0 <= ny < 64 and not seen[ny][nx] \
                            and grid[ny][nx] == c:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            comps.append((c, cells))
    return comps


def _stencil(color, cells, x0, y0, w, h):
    rows = []
    cellset = set(cells)
    for y in range(y0, y0 + h):
        rows.append("".join(
            HEX[color] if (x, y) in cellset else "." for x in range(x0, x0 + w)))
    return "/".join(rows)


def describe(grid, max_blocks=48):
    """The grid as blocks, biggest first. Exact rects stay one line; small
    irregular shapes get an inline stencil ('.'-padded rows joined by '/')."""
    comps = components(grid)
    comps.sort(key=lambda t: -len(t[1]))
    lines = [f"{len(comps)} connected blocks (biggest first — the biggest is "
             f"usually background/terrain):"]
    for c, cells in comps[:max_blocks]:
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if len(cells) == w * h:
            lines.append(f"- color {HEX[c]}: rect {w}x{h} at ({x0},{y0})")
        elif w <= 16 and h <= 16:
            lines.append(f"- color {HEX[c]}: {len(cells)} cells in {w}x{h} bbox at "
                         f"({x0},{y0}), shape {_stencil(c, cells, x0, y0, w, h)}")
        else:
            lines.append(f"- color {HEX[c]}: {len(cells)} cells in {w}x{h} bbox at "
                         f"({x0},{y0}) (irregular)")
    if len(comps) > max_blocks:
        lines.append(f"  ... +{len(comps) - max_blocks} smaller blocks")
    return "\n".join(lines)


def _translations(before, after):
    """Per-color translation detection: for each color whose cell set changed,
    check whether vanished cells map onto appeared cells by one (dx, dy).
    Colors sharing a (dx, dy) are grouped — that is one multi-color object
    moving. Background colors fail the check and stay silent (their vanished/
    appeared sets are where the object arrived/left — different shapes)."""
    per = {}
    for c in range(16):
        b = {(x, y) for y in range(64) for x in range(64) if before[y][x] == c}
        a = {(x, y) for y in range(64) for x in range(64) if after[y][x] == c}
        if b == a or not b or not a:
            continue
        van, app = b - a, a - b
        if not van or not app or len(van) != len(app):
            continue
        dx = min(x for x, _ in app) - min(x for x, _ in van)
        dy = min(y for _, y in app) - min(y for _, y in van)
        if {(x + dx, y + dy) for x, y in van} == app:
            per.setdefault((dx, dy), []).append((c, len(van)))
    out = []
    for (dx, dy), colors in sorted(per.items()):
        desc = ", ".join(f"{HEX[c]} ({n} cells)" for c, n in colors)
        out.append(f"MOVEMENT: color(s) {desc} translated by (dx={dx:+d}, dy={dy:+d}) "
                   f"— consistent with an object of that shape MOVING, not colors changing")
    return out


def flow(before, after, max_rows=30):
    """Sparse change map in grid space + movement detection.

    Only the bbox of changed cells is rendered; '.' = unchanged. One glance
    shows the shape of what vanished (left column) and what appeared (right)."""
    changed = [(x, y) for y in range(64) for x in range(64)
               if before[y][x] != after[y][x]]
    if not changed:
        return "no change (0 cells)"
    xs = [c[0] for c in changed]
    ys = [c[1] for c in changed]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    lines = [f"{len(changed)} cells changed in bbox x[{x0}..{x1}] y[{y0}..{y1}]"]
    lines += _translations(before, after)
    if y1 - y0 + 1 <= max_rows:
        lines.append(f"change map, BEFORE -> AFTER ('.' = unchanged), "
                     f"cols {x0}..{x1}:")
        for y in range(y0, y1 + 1):
            b = "".join(HEX[before[y][x]] if before[y][x] != after[y][x] else "."
                        for x in range(x0, x1 + 1))
            a = "".join(HEX[after[y][x]] if before[y][x] != after[y][x] else "."
                        for x in range(x0, x1 + 1))
            lines.append(f"{y:2d}| {b} -> {a}")
    else:
        lines.append(f"(change region spans {y1 - y0 + 1} rows — too tall to render; "
                     f"use ANALYZE to inspect)")
    return "\n".join(lines)
