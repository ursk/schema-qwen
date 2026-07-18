def init_state(grid):
    return {"grid": [row[:] for row in grid]}

def step(state, action):
    new_grid = [row[:] for row in state["grid"]]
    
    if action == 1:
        # HUD
        for y in [61, 62]:
            for x in [13, 14, 15]:
                new_grid[y][x] = 3
        # Block
        for y in range(40, 50):
            for x in range(29, 39):
                curr = state["grid"][y][x]
                if curr == 3:
                    new_grid[y][x] = 9
                elif curr == 9:
                    new_grid[y][x] = 3
                elif curr == 12:
                    new_grid[y][x] = 3 # or something
                elif curr == 4:
                    new_grid[y][x] = 3 # or something
                else:
                    new_grid[y][x] = curr
    elif action == 2:
        # ... (previous logic)
        pass
    elif action == 3:
        # ...
        pass
    elif action == 4:
        # ...
        pass
    return {"grid": new_grid}
