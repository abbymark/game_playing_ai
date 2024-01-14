from game_playing_ai.games.food_game.food_game import FoodGame

import pygame

import sys
class GameStarter:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Game Playing AI")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = "menu"

        self.title = "Game Playing AI"
        self.title_font = pygame.font.Font(size=32)
        self.title_pos = (self.width // 2, self.height // 4)
        self.title_button = Button(self.screen, "Game Playing AI", self.title_font, self.title_pos, (255, 255, 255))

        self.food_game_button_text = "Food collecting game"
        self.food_game_button_font = pygame.font.Font(size=32)
        self.food_game_button_text_pos = (self.width // 2, self.height // 2)
        self.food_game_button = Button(self.screen, self.food_game_button_text, self.food_game_button_font, self.food_game_button_text_pos, (255, 255, 255))

        self.team_game_button_text = "Team food collecting game"
        self.team_game_button_font = pygame.font.Font(size=32)
        self.team_game_button_text_pos = (self.width // 2, self.height // 2 + 100)
        self.team_game_button = Button(self.screen, self.team_game_button_text, self.team_game_button_font, self.team_game_button_text_pos, (255, 255, 255))

        

    def run(self):
        while self.running:
            self.clock.tick(5)
            if self.state == "menu":
                events = pygame.event.get()
                self.events(events)
                self.update(events)
                self.draw()
            elif self.state == "food_game":
                food_game = FoodGame()
                food_game.run()
                self.state = "menu"
            elif self.state == "team_game":
                pass
    
    def draw(self):
        self.draw_menu()
        pygame.display.update()


    def draw_menu(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.title_button.text_surface, self.title_button.text_rect)
        self.screen.blit(self.food_game_button.text_surface, self.food_game_button.text_rect)
        self.screen.blit(self.team_game_button.text_surface, self.team_game_button.text_rect)


    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if self.food_game_button.collidepoint(x, y):
                    self.state = "food_game"
                elif self.team_game_button.collidepoint(x, y):
                    self.state = "team_game"
    
    def update(self, events):
        pass


class Button(pygame.sprite.Sprite):
    def __init__(self, screen, text, text_font, text_pos, text_color):
        super().__init__()
        self.screen = screen
        self.text = text
        self.text_font = text_font
        self.text_pos = text_pos
        self.text_color = text_color

        self.text_surface = self.text_font.render(self.text, True, self.text_color)
        self.text_rect = self.text_surface.get_rect()
        self.text_rect.center = self.text_pos


    def draw(self):
        # pygame.draw.rect(self.screen, self.rect_color, self.rect)
        self.screen.blit(self.text_surface, self.text_rect)

    def collidepoint(self, x, y):
        return self.text_rect.collidepoint(x, y)



if __name__ == "__main__":



    game = GameStarter()
    game.run()