import random


class DRLAgentSprite():
    def __init__(self, rows, cols, pos = None):
        self.rows = rows
        self.cols = cols
        self.food_collected = 0
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
        while map[self.y][self.x] != 0 and map[self.y][self.x] != 3:
            self.x = random.randint(0, self.cols - 1)
            self.y = random.randint(0, self.rows - 1)
        map[self.y][self.x] = 5
        return map

    @property
    def pos(self):
        return (self.x, self.y)

    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]