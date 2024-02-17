from game_playing_ai.games.food_game.food_game import FoodGame
from game_playing_ai.games.food_game.agents.drl_agent.trainer import DQNTrainer, PPOTrainer
from game_playing_ai.pages.food_game_train_page import FoodGameTrainPage

import pygame
import pygame_gui

import sys
import threading
import json
class GameStarter:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.manager = pygame_gui.UIManager((self.width, self.height), "theme.json")

        self.food_game_play_config = {}

        pygame.display.set_caption("Game Playing AI")
        self.clock = pygame.time.Clock()
        self.running = True

        # pages
        self.food_game_train_page = FoodGameTrainPage(self.width, self.height)
        self.food_game_train_page.hide()

        self._main_menu_panel()
        
        

        self.page_state = "main_menu"  # main_menu, food_game_play_panel, food_game_train_panel


    def run(self):
        while self.running:
            self.time_delta = self.clock.tick(60)/1000.0
            events = pygame.event.get()
            self._events(events)
            self._update(events)
            self._draw()
    
    def _draw(self):
        self.food_game_train_page.draw(self.screen)
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    def _events(self, events):
        self.food_game_train_page.process_events(events)
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type ==pygame_gui.UI_BUTTON_PRESSED:
                if self.page_state == "main_menu":
                    if event.ui_element == self.food_game_play_button:
                        self._food_game_play_panel()
                        self.main_menu_panel.hide()
                        self.page_state = "food_game_play_panel"
                    elif event.ui_element == self.food_game_train_button:
                        self.main_menu_panel.hide()
                        self.food_game_train_page.show()
                        self.page_state = "food_game_train_panel"

                if self.page_state == "food_game_play_panel":
                    if event.ui_element == self.food_game_run_button:
                        with open(f"{self.food_game_play_config['model_path']}/config.json", "r") as f:
                            config = json.load(f)

                        food_game = FoodGame(rows = config["rows"], cols = config["cols"], 
                                             drl_model_path=self.food_game_play_config["model_path"], 
                                             solo=self.food_game_play_config["solo"])

                        food_game.run()
                    elif event.ui_element == self.food_game_back_button:
                        self.food_game_play_panel.hide()
                        self.main_menu_panel.show()
                        self.page_state = "main_menu"


            elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if self.page_state == "food_game_play_panel":
                    if event.ui_element == self.solo_drop_down_menu:
                        self.food_game_play_config["solo"] = True if event.text == "True" else False

            elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:

                # Food Game Play Config
                if self.page_state == "food_game_play_panel":
                    if event.ui_element == self.model_path_text_entry:
                        self.food_game_play_config["model_path"] = event.text




            self.manager.process_events(event)
    
    def _main_menu_panel(self):
        self.main_menu_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (self.width, self.height)), manager=self.manager)
        self.game_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 100)), 
                                                            container=self.main_menu_panel, anchors={"centerx": "centerx"},
                                                            text='Game Playing AI', manager=self.manager, object_id="#main_title")
        
        self.food_game_play_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 100), (150, 50)), 
                                                             container=self.main_menu_panel, anchors={"centerx": "centerx"},
                                                             text='Play Food Game', manager=self.manager)
        self.food_game_train_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 100), (150, 50)), 
                                                                   container=self.main_menu_panel, anchors={"centerx": "centerx"},
                                                             text='Train Food Game', manager=self.manager)
    
    def _food_game_play_panel(self):
        self.food_game_play_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (self.width, self.height)), 
                                                              manager=self.manager, object_id="#play_panel")
        self.food_game_play_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 50)), 
                                                            container=self.food_game_play_panel, anchors={"centerx": "centerx"},
                                                            text='Food Game', manager=self.manager, object_id="#main_title")
        
        # Model Path
        self.model_path_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,50), (150, 30)),
                                                              container=self.food_game_play_panel,
                                                              text='Model Path', manager=self.manager)
        
        self.model_path_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 50), (500, 30)), 
                                                            container=self.food_game_play_panel,
                                                            manager=self.manager)
        self.food_game_play_config["model_path"] = ""


        # Solo runnning
        self.solo_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,100), (150, 30)), 
                                                            container=self.food_game_play_panel,
                                                            text='Solo', manager=self.manager)
        
        self.solo_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((200, 100), (100, 30)),
                                                                    container=self.food_game_play_panel,
                                                                    manager=self.manager,
                                                                    options_list=["True", "False"],
                                                                    starting_option="False")
        self.food_game_play_config["solo"] = False



        # Run button
        self.food_game_run_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 550), (150, 50)), 
                                                             container=self.food_game_play_panel, anchors={"centerx": "centerx"},
                                                             text='Run', manager=self.manager)
        # Back button
        self.food_game_back_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 550), (150, 50)),
                                                        container=self.food_game_play_panel, anchors={"centerx": "centerx"},
                                                        text='Back', manager=self.manager)

        
        
    
    def _update(self, events):
        self.food_game_train_page.update(self.time_delta)
        self.manager.update(self.time_delta)






if __name__ == "__main__":

    # train_drl_agent()

    game = GameStarter()
    game.run()