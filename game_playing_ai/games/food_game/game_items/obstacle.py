from game_playing_ai.games.food_game.TileType import TileType

import random

def place_obstacles(grid, obstacle_count):
    rows = len(grid)
    cols = len(grid[0])
    placed_obstacles = 0

    while placed_obstacles < obstacle_count:
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)

        # Check if current spot is empty and not surrounded
        if grid[row][col] == 0 and not is_surrounded(grid, row, col):
            grid[row][col] = TileType.OBSTACLE
            placed_obstacles += 1
            
            # Ensure surrounding cells are not made inaccessible
            for r, c in get_neighbors(row, col, rows, cols):
                if is_surrounded(grid, r, c):
                    grid[row][col] = 0  # Remove the obstacle if it surrounds any neighbor
                    placed_obstacles -= 1
                    break
    return grid

def is_surrounded(grid, row, col):
    """Check if the cell is surrounded by obstacles."""
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            if grid[r][c] == TileType.EMPTY:
                return False
    return True

def get_neighbors(row, col, rows, cols):
    """Get valid neighbors for a given cell."""
    neighbors = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols:
            neighbors.append((r, c))
    return neighbors