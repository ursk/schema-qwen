"""Grid encoding and diffing utilities.

Grids are 64x64 lists of ints 0..15. Text encoding is one hex digit per
cell so a full grid is 64 lines of 64 chars — compact enough to put in a
prompt, precise enough to reason about coordinates.
"""

HEX = "0123456789abcdef"


def grid_to_text(grid, ruler=True):
    lines = []
    if ruler:
        lines.append("    " + "".join(HEX[(c // 10) % 10] if c % 10 == 0 and c > 0 else " " for c in range(64)))
        lines.append("    " + "".join(str(c % 10) for c in range(64)))
    for y, row in enumerate(grid):
        prefix = f"{y:2d}| " if ruler else ""
        lines.append(prefix + "".join(HEX[v & 15] for v in row))
    return "\n".join(lines)


def diff_cells(a, b):
    """All (x, y, old, new) cells where grids differ."""
    out = []
    for y in range(64):
        ra, rb = a[y], b[y]
        if ra == rb:
            continue
        for x in range(64):
            if ra[x] != rb[x]:
                out.append((x, y, ra[x], rb[x]))
    return out


def diff_summary(a, b, max_cells=40):
    """Human/model-readable diff between two grids."""
    cells = diff_cells(a, b)
    if not cells:
        return "no change (0 cells)"
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    head = (
        f"{len(cells)} cells changed in bbox x[{min(xs)}..{max(xs)}] y[{min(ys)}..{max(ys)}]"
    )
    if len(cells) <= max_cells:
        body = ", ".join(f"({x},{y}):{HEX[o]}->{HEX[n]}" for x, y, o, n in cells)
    else:
        body = (
            ", ".join(f"({x},{y}):{HEX[o]}->{HEX[n]}" for x, y, o, n in cells[:max_cells])
            + f", ... (+{len(cells) - max_cells} more)"
        )
    return head + ": " + body
