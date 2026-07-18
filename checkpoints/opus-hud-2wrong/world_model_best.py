def init_state(grid):
    cs = [(x, y) for y in range(64) for x in range(64) if grid[y][x] == 12]
    bx = min(x for x, y in cs); by = min(y for x, y in cs)
    base = [row[:] for row in grid]
    for yy in range(by, by+5):
        for xx in range(bx, bx+5):
            base[yy][xx] = 3
    bar_cols = sorted(x for x in range(64) if grid[61][x] == 11)
    return {"base": base, "bx": bx, "by": by, "cnt": 0, "used": 0, "bar": bar_cols}

def _flags(base, bx, by):
    if bx < 0 or by < 0 or bx+5 > 64 or by+5 > 64:
        return True, True, True
    has9 = has4 = False
    for yy in range(by, by+5):
        for xx in range(bx, bx+5):
            v = base[yy][xx]
            if v == 9: has9 = True
            elif v == 4: has4 = True
    return has9, has4, False

def step(state, action):
    bx, by = state["bx"], state["by"]
    d = {1:(0,-5), 2:(0,5), 3:(-5,0), 4:(5,0)}
    nbx, nby, used = bx, by, state["used"]
    if action in d:
        dx, dy = d[action]
        h9, h4, oob = _flags(state["base"], bx+dx, by+dy)
        if not h9 and not oob:
            used += 1
            if not h4:
                nbx, nby = bx+dx, by+dy
    return {"base": state["base"], "bx": nbx, "by": nby,
            "cnt": state["cnt"]+1, "used": used, "bar": state["bar"]}

def render(state):
    g = [row[:] for row in state["base"]]
    bx, by = state["bx"], state["by"]
    for i in range(5):
        for xx in range(bx, bx+5):
            g[by+i][xx] = 12 if i < 2 else 9
    for k in range(min(state["used"], len(state["bar"]))):
        col = state["bar"][k]
        g[61][col] = 3; g[62][col] = 3
    return g

def is_goal(state):
    return False
