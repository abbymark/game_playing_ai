from game_playing_ai.games.food_game.food_game import FoodGame
from game_playing_ai.games.food_game.agents.drl_agent.trainer import DQNTrainer, PPOTrainer
from game_playing_ai.pages.food_game_train_page import FoodGameTrainPage
from game_playing_ai.pages.food_game_run_page import FoodGameRunPage
from game_playing_ai.pages.main_page import MainPage

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

        # pages
        self.main_page = MainPage(self.width, self.height)
        self.main_page.show()
        self.food_game_train_page = FoodGameTrainPage(self.width, self.height, self.main_page)
        self.food_game_train_page.hide()
        self.food_game_run_page = FoodGameRunPage(self.width, self.height, self.main_page)
        self.food_game_run_page.hide()
        

        self.main_page.set_changeable_pages(
            {"food_game_train_page": self.food_game_train_page, 
             "food_game_run_page": self.food_game_run_page}
        )

    def run(self):
        while self.running:
            self.time_delta = self.clock.tick(60)/1000.0
            events = pygame.event.get()
            self._events(events)
            self._update(events)
            self._draw()
    
    def _draw(self):
        self.main_page.draw(self.screen)
        self.food_game_train_page.draw(self.screen)
        self.food_game_run_page.draw(self.screen)
        pygame.display.update()

    def _events(self, events):
        self.main_page.process_events(events)
        self.food_game_train_page.process_events(events)
        self.food_game_run_page.process_events(events)
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
        
    
    def _update(self):
        self.main_page.update(self.time_delta)
        self.food_game_train_page.update(self.time_delta)
        self.food_game_run_page.update(self.time_delta)






if __name__ == "__main__":

    game = GameStarter()
    game.run()