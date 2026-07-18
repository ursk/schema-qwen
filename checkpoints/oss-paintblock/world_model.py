def init_state(grid):
    # Preserve the raw grid and track horizontal offsets for the two moving blocks.
    return {
        "grid": [row[:] for row in grid],
        "off1": 0,          # offset for actions 1/2 (block anchored at column 34)
        "off3": 0,          # offset for actions 3/4 (block anchored at column 29)
    }

# ---------- constants ----------
_BASE1_X = 34                     # initial column for block controlled by actions 1/2
_BASE1_Y = 40                     # rows 40‑41 become colour 0xc, rows 42‑44 become 0x9
_MAX_OFF1 = 64 - _BASE1_X - 5     # max horizontal offset before the block would leave the grid

_BASE3_X = 29                     # initial column for block controlled by actions 3/4
_BASE3_Y = 45                     # rows 45‑46 become colour 0xc, rows 47‑49 become 0x9
_MAX_OFF3 = 64 - _BASE3_X - 5

# ---------- helpers ----------
def _revert_block(grid, base_x, base_y, off):
    """
    Revert any painted cell (0xc or 0x9) back to 0x3.
    """
    for dx in range(5):
        x = base_x + off + dx
        if not (0 <= x < 64):
            continue
        for dy in range(5):
            y = base_y + dy
            if not (0 <= y < 64):
                continue
            if grid[y][x] in (0xc, 0x9):
                grid[y][x] = 0x3

def _paint_block(grid, base_x, base_y, off):
    """
    Paint the 5×5 block:
        top two rows (dy < 2) → colour 0xc
        bottom three rows (dy >= 2) → colour 0x9
    """
    for dx in range(5):
        x = base_x + off + dx
        if not (0 <= x < 64):
            continue
        for dy in range(5):
            y = base_y + dy
            if not (0 <= y < 64):
                continue
            if dy < 2:
                # top rows become 0xc
                if grid[y][x] == 0x3:
                    grid[y][x] = 0xc
            else:
                # bottom rows become 0x9
                if grid[y][x] == 0x3:
                    grid[y][x] = 0x9

def step(state, action):
    new_grid = [row[:] for row in state["grid"]]
    off1 = state["off1"]
    off3 = state["off3"]

    if action == 1:
        # Paint at current offset
        _paint_block(new_grid, _BASE1_X, _BASE1_Y, off1)
        # Increment offset for next action
        off1 = (off1 + 1) % (_MAX_OFF1 + 1)
        # Revert block that was at previous offset (off1-1)
        old_off = (off1 - 1) % (_MAX_OFF1 + 1)
        _revert_block(new_grid, _BASE1_X, _BASE1_Y, old_off)

    elif action == 2:
        # Increment offset (no painting)
        off1 = (off1 + 1) % (_MAX_OFF1 + 1)
        # Revert block that was at previous offset
        old_off = (off1 - 1) % (_MAX_OFF1 + 1)
        _revert_block(new_grid, _BASE1_X, _BASE1_Y, old_off)

    elif action == 3:
        # Paint at current offset
        _paint_block(new_grid, _BASE3_X, _BASE3_Y, off3)
        # Increment offset for next action
        off3 = (off3 + 1) % (_MAX_OFF3 + 1)
        # Revert block that was at previous offset
        old_off = (off3 - 1) % (_MAX_OFF3 + 1)
        _revert_block(new_grid, _BASE3_X, _BASE3_Y, old_off)

    elif action == 4:
        # Increment offset (no painting)
        off3 = (off3 + 1) % (_MAX_OFF3 + 1)
        # Revert block that was at previous offset
        old_off = (off3 - 1) % (_MAX_OFF3 + 1)
        _revert_block(new_grid, _BASE3_X, _BASE3_Y, old_off)

    elif action == 5:
        # no‑op
        pass

    return {
        "grid": new_grid,
        "off1": off1,
        "off3": off3,
    }

def render(state):
    return [row[:] for row in state["grid"]]

def is_goal(state):
    # Goal: no cells of colour 0xc remain on the grid
    for row in state["grid"]:
        for cell in row:
            if cell == 0xc:
                return False
    return True
