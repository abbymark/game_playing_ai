import pygame

import random

class Food():
    def __init__(self, rows, cols, x, y):
        self.x = x
        self.y = y
        self.rows = rows
        self.cols = cols

    @classmethod
    def generate_foods(cls, rows, cols, n_food, existing_foods=None):
        if existing_foods is None:
            existing_foods = []
        foods = existing_foods
        for i in range(n_food):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            while (x, y) in [(food.x, food.y) for food in foods]:
                x = random.randint(0, cols - 1)
                y = random.randint(0, rows - 1)
            foods.append(cls(rows, cols, x, y))
        return foods
    
    def set_pos_in_map(self, map):
        while map[self.y][self.x] != 0 and map[self.y][self.x] != 2:
            self.x = random.randint(0, self.cols - 1)
            self.y = random.randint(0, self.rows - 1)
        map[self.y][self.x] = 2
        return map

    @property
    def pos(self):
        return (self.x, self.y)

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]