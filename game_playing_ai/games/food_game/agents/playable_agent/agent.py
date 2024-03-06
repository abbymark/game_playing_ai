from game_playing_ai.games.food_game.TileType import TileType

import pygame

import random
class PlayableAgent:
    def __init__(self, rows, cols, pos = None):
        self.rows = rows
        self.cols = cols
        self.food_collected = 0
        self.hp = 100
        self.is_alive = True

        if pos is None:
            self.x = random.randint(0, cols - 1)
            self.y = random.randint(0, rows - 1)
        else:
            self.x = pos[0]
            self.y = pos[1]

    def update(self, events):
        if events is None:
            return
        for event in events:
            if event.type == pygame.KEYDOWN:
                new_x = self.x
                new_y = self.y
                if event.key == pygame.K_LEFT:
                    new_x -= 1
                elif event.key == pygame.K_RIGHT:
                    new_x += 1
                elif event.key == pygame.K_UP:
                    new_y -= 1
                elif event.key == pygame.K_DOWN:
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
        map[self.y][self.x] = TileType.PLAYABLE_AGENT
        return map

    @property
    def pos(self):
        return (self.x, self.y)
    
    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]