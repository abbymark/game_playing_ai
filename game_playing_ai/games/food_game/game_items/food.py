import pygame

import random

from typing import List
class Food():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def generate_foods(cls, map, n_food, existing_foods=None) -> 'List[Food]':
        if existing_foods is None:
            existing_foods = []
        foods = existing_foods
        for i in range(n_food):
            x = random.randint(0, len(map[0]) - 1)
            y = random.randint(0, len(map) - 1)
            while map[y][x] != 0:
                x = random.randint(0, len(map[0]) - 1)
                y = random.randint(0, len(map) - 1)
            foods.append(Food(x, y))
        return foods
    

    @property
    def pos(self):
        return (self.x, self.y)

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]