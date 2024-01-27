from game_playing_ai.games.food_game.food_game import FoodGame, train_drl_agent

import pygame
import pygame_gui

import sys
import threading
class GameStarter:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.manager = pygame_gui.UIManager((self.width, self.height), "theme.json")
        self.container = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (self.width, self.height)), manager=self.manager)
        self.game_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 100)), container=self.container, anchors={"centerx": "centerx"},
                                                            text='Game Playing AI', manager=self.manager, object_id="#main_title")
        
        self.food_game_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 100), (150, 50)), container=self.container, anchors={"centerx": "centerx"},
                                                             text='Run Food Game', manager=self.manager)
        self.food_game_train_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 100), (150, 50)), container=self.container, anchors={"centerx": "centerx"},
                                                             text='Train Food Game', manager=self.manager)


        pygame.display.set_caption("Game Playing AI")
        self.clock = pygame.time.Clock()
        self.running = True


    def run(self):
        while self.running:
            self.time_delta = self.clock.tick(5)/1000.0
            events = pygame.event.get()
            self.events(events)
            self.update(events)
            self.draw()
    
    def draw(self):
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type ==pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.food_game_button:
                    food_game = FoodGame(render_mode="human")
                    food_game.run()
                elif event.ui_element == self.food_game_train_button:
                    train_drl_agent()
            self.manager.process_events(event)

    
    def update(self, events):
        self.manager.update(self.time_delta)






if __name__ == "__main__":

    # train_drl_agent()

    game = GameStarter()
    game.run()