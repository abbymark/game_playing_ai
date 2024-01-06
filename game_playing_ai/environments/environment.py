import pygame

class Environment:
    def __init__(self, screen, rows, cols):
        self.screen = screen
        self.rows = rows
        self.cols = cols

    def draw(self):
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col*self.screen.get_width()/self.cols, row*self.screen.get_height()/self.rows, self.screen.get_width()/self.cols, self.screen.get_height()/self.rows)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
                