import pygame

class PlayableAgent(pygame.sprite.Sprite):
    def __init__(self, screen, rows, cols, x, y):
        super().__init__()
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.cell_width = self.screen.get_width() / self.cols
        self.cell_height = self.screen.get_height() / self.rows
        self.image = pygame.Surface((self.cell_width, self.cell_height))
        self.image.fill((255, 255, 0))
        self.rect = self.image.get_rect()

        # Set the position of the agent (for example, at the center of the cell)
        self.rect.x = x * self.cell_width
        self.rect.y = y * self.cell_height

    def draw(self):
        self.screen.blit(self.image, self.rect)

    def update(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                new_x = self.rect.x
                new_y = self.rect.y

                if event.key == pygame.K_LEFT:
                    new_x -= self.cell_width
                elif event.key == pygame.K_RIGHT:
                    new_x += self.cell_width
                elif event.key == pygame.K_UP:
                    new_y -= self.cell_height
                elif event.key == pygame.K_DOWN:
                    new_y += self.cell_height

                # Check if new position is within bounds
                if 0 <= new_x < self.screen.get_width() and 0 <= new_y < self.screen.get_height():
                    # Update position if within bounds
                    self.rect.x = new_x
                    self.rect.y = new_y