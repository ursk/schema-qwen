def init_state(grid):
    return {"grid": [row[:] for row in grid]}

def step(state, action):
    new_grid = [row[:] for row in state["grid"]]
    
    if action == 1:
        # Action 1: Sets HUD and a block
        for y in [61, 62]:
            for x in [13, 14]:
                new_grid[y][x] = 3
        for y in range(40, 50):
            for x in range(34, 39):
                new_grid[y][x] = 3
    elif action == 2:
        # Action 2: Transformation of the block
        for y in range(40, 50):
            for x in range(34, 39):
                if y <= 44: target = 3
                elif y <= 46: target = 12
                else: target = 9
                new_grid[y][x] = target
    elif action == 3:
        # Action 3: Another transformation
        for y in [61, 62]:
            for x in [13, 14]:
                if x == 13: new_grid[y][x] = 3
        for y in range(45, 50):
            for x in range(29, 39):
                if 45 <= y <= 46:
                    val = 12 if x < 34 else 3
                    new_grid[y][x] = val
                else:
                    val = 9 if x < 34 else 3
                    new_grid[y][x] = val
    elif action == 4:
        # Action 4: Expanding/changing the block and HUD
        # Mismatch showed x=15 was affected
        for y in [61, 62]:
            for x in [13, 14, 15]:
                if x == 13: new_grid[y][x] = 3
                elif x == 14: new_grid[y][x] = 3 # placeholder
                elif x == 15: new_grid[y][x] = 3
        # Correcting the x range for the block based on mismatch
        for y in range(40, 50):
            for x in range(29, 39):
                if y <= 44:
                    val = 3 if x >= 34 else 4
                elif 45 <= y <= 46:
                    val = 3 if x < 34 else 12
                else:
                    val = 3 if x < 34 else 9
                new_grid[y][x] = val
    return {"grid": new_grid}

def render(state):
    return state["grid"]

def is_goal(state):
    return False
