from game_playing_ai.games.food_game.food_game import TileType

import random


class DRLAgentSprite():
    def __init__(self, rows, cols, pos = None):
        self.rows = rows
        self.cols = cols
        self.food_collected = 0
        self.hp = 100
        if pos is None:
            self.x = random.randint(0, cols - 1)
            self.y = random.randint(0, rows - 1)
        else:
            self.x = pos[0]
            self.y = pos[1]

    def update(self, action):
        if action is None:
            return
        new_x = self.x
        new_y = self.y

        if action == 0:
            new_x -= 1
        elif action == 1:
            new_x += 1
        elif action == 2:
            new_y -= 1
        elif action == 3:
            new_y += 1

        # Check if new position is within bounds
        if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
            # Update position if within bounds
            self.x = new_x
            self.y = new_y



    def set_pos_in_map(self, map):
        while map[self.y][self.x] != TileType.EMPTY:
            self.x = random.randint(0, self.cols - 1)
            self.y = random.randint(0, self.rows - 1)
        map[self.y][self.x] = TileType.DRL_AGENT
        return map

    @property
    def pos(self):
        return (self.x, self.y)

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]

    def get_obs(self, map):
        map = map.copy()
        map[self.y][self.x] = TileType.AGENT_LOCATION
        return map
    
    def increase_food_collected(self):
        self.prev_food_collected = self.food_collected
        self.food_collected += 1
        return self.food_collected
    