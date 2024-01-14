import pygame

class Food(pygame.sprite.Sprite):
    def __init__(self, screen, rows, cols, x, y):
        super().__init__()
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.cell_width = self.screen.get_width() / self.cols
        self.cell_height = self.screen.get_height() / self.rows
        self.image = pygame.Surface((self.cell_width, self.cell_height))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.x = x * self.cell_width
        self.rect.y = y * self.cell_height

    def draw(self):
        self.screen.blit(self.image, self.rect)