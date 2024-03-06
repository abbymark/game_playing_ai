from game_playing_ai.games.food_game.TileType import TileType

import pygame
from scipy.spatial import KDTree

import random
from typing import List
import heapq
class PreprogrammedAgent:
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

        self.actions = {
            0: (self.x - 1, self.y),
            1: (self.x + 1, self.y),
            2: (self.x, self.y - 1),
            3: (self.x, self.y + 1)
        }

        self.action_from_direction = {
            (-1, 0): 0,
            (1, 0): 1,
            (0, -1): 2,
            (0, 1): 3
        }

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
    
    def act(self, map, foods: List):

        if self.hp < 20:
            return self.evade(map)
        else:
            return self.seek(map, foods)

    def evade(self, map):
        # check if enemy is nearby
        possible_actions = []
        for k, v in self.actions.items():
            dr, dc = v
            r, c = self.y + dr, self.x + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if map[r][c] not in [TileType.OBSTACLE, TileType.PLAYABLE_AGENT, TileType.DRL_AGENT]:
                    possible_actions.append(k)
        if len(possible_actions) == 0:
            return random.choice([0, 1, 2, 3])
        return random.choice(possible_actions)
    
    def seek(self, map, foods):
    
        def get_closest_food_position(foods, n_th_closest):
            # Create a KDTree for the food
            food_positions = [(food.x, food.y) for food in foods]
            food_tree = KDTree(food_positions)

            # Get the closest food
            distance, indices = food_tree.query([self.x, self.y], k = 10)  # 10 is to get the 10 closest food(when food count is 10)
            
            closest_food = foods[indices[n_th_closest]]

            # Get the position of the closest food
            food_x = closest_food.x
            food_y = closest_food.y
            return (food_x, food_y)

        def heuristic(start, goal):
            return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        
        def a_star_search(map, start, goal):
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            close_set = set()
            came_from = {}
            gscore = {start: 0}
            fscore = {start: heuristic(start, goal)}
            oheap = []

            heapq.heappush(oheap, (fscore[start], start))

            while oheap:
                current = heapq.heappop(oheap)[1]

                if current == goal:
                    data = []
                    while current in came_from:
                        data.append(current)
                        current = came_from[current]
                    return data

                close_set.add(current)
                for i, j in neighbors:
                    neighbor = current[0] + i, current[1] + j
                    tentative_g_score = gscore[current] + 1

                    if 0 <= neighbor[0] < len(map[0]) and 0 <= neighbor[1] < len(map) and map[neighbor[1]][neighbor[0]] in [TileType.EMPTY, TileType.FOOD]:
                        if neighbor not in close_set and (neighbor not in gscore or tentative_g_score < gscore[neighbor]):
                            came_from[neighbor] = current
                            gscore[neighbor] = tentative_g_score
                            fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                            if neighbor not in [i[1] for i in oheap]:
                                heapq.heappush(oheap, (fscore[neighbor], neighbor))

        for i in range(10):  # 11 is to get the 10th closest food(when food count is 10)
            x, y = get_closest_food_position(foods, i)
            path = a_star_search(map, (self.x, self.y), (x, y))
            if path:
                path = path[::-1]
                return self.action_from_direction[(path[0][0] - self.x, path[0][1] - self.y)]

        breakpoint()
        return random.choice([0, 1, 2, 3])





    
    def set_pos_in_map(self, map):
        while map[self.y][self.x] != TileType.EMPTY:
            self.x = random.randint(0, self.cols - 1)
            self.y = random.randint(0, self.rows - 1)
        map[self.y][self.x] = TileType.PREPROGRAMMED_AGENT
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