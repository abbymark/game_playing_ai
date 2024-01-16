import pygame
from scipy.spatial import KDTree

import random
class PreprogrammedAgent:
    def __init__(self, rows, cols, pos = None):
        self.rows = rows
        self.cols = cols
        if pos is None:
            self.x = random.randint(0, cols - 1)
            self.y = random.randint(0, rows - 1)
        else:
            self.x = pos[0]
            self.y = pos[1]

    def update(self, food:list):
        # Create a KDTree for the food
        food_positions = [(food.x, food.y) for food in food]
        food_tree = KDTree(food_positions)

        # Get the closest food
        closest_food = food[food_tree.query([self.x, self.y])[1]]

        # Get the position of the closest food
        food_x = closest_food.x
        food_y = closest_food.y

        # Calculate the Manhattan distances to the food
        distance_x = food_x - self.x
        distance_y = food_y - self.y

        # Move horizontally towards the food
        if distance_x != 0:
            self.x += 1 if distance_x > 0 else -1

        # If aligned horizontally, start moving vertically
        elif distance_y != 0:
            self.y += 1 if distance_y > 0 else -1

        # Ensure the agent doesn't move out of the screen
        self.x = max(0, min(self.x, self.cols - 1))
        self.y = max(0, min(self.y, self.rows - 1))
    
    @property
    def pos(self):
        return (self.x, self.y)

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]