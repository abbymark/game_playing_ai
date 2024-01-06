import pygame
from scipy.spatial import KDTree

class PreprogrammedAgent(pygame.sprite.Sprite):
    def __init__(self, screen, rows, cols, x, y):
        super().__init__()
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.cell_width = self.screen.get_width() / self.cols
        self.cell_height = self.screen.get_height() / self.rows
        self.image = pygame.Surface((self.cell_width, self.cell_height))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()

        # Set the position of the agent (for example, at the center of the cell)
        self.rect.x = x * self.cell_width
        self.rect.y = y * self.cell_height

    def draw(self):
        self.screen.blit(self.image, self.rect)

    def update(self, food:list):
        # Create a KDTree for the food
        food_positions = [(food.rect.x, food.rect.y) for food in food]
        food_tree = KDTree(food_positions)

        # Get the closest food
        closest_food = food[food_tree.query([self.rect.x, self.rect.y])[1]]

        # Get the position of the closest food
        food_x = closest_food.rect.x
        food_y = closest_food.rect.y

        # Calculate the Manhattan distances to the food
        distance_x = food_x - self.rect.x
        distance_y = food_y - self.rect.y

        # Move horizontally towards the food
        if distance_x != 0:
            self.rect.x += self.cell_width if distance_x > 0 else -self.cell_width

        # If aligned horizontally, start moving vertically
        elif distance_y != 0:
            self.rect.y += self.cell_height if distance_y > 0 else -self.cell_height

        # Ensure the agent doesn't move out of the screen
        self.rect.x = max(0, min(self.rect.x, self.screen.get_width() - self.cell_width))
        self.rect.y = max(0, min(self.rect.y, self.screen.get_height() - self.cell_height))