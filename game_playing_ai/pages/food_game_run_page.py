from game_playing_ai.games.food_game.food_game import FoodGame

import pygame
import pygame_gui

import json

class FoodGameRunPage:
    def __init__(self, width, height, main_page):
        self.width = width
        self.height = height
        self.manager = pygame_gui.UIManager((self.width, self.height), "theme.json")
        self.main_page = main_page
        self.arguments_to_pass = {}
        self.make_page()
    
    def process_events(self, events):
        for event in events:
            if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                if event.ui_element == self.model_path_text_entry:
                    self.arguments_to_pass["model_path"] = event.text
            
            elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.solo_drop_down_menu:
                    self.arguments_to_pass["solo"] = True if event.text == "True" else False

            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.food_game_run_button:
                    with open(f"{self.arguments_to_pass['model_path']}/config.json", "r") as f:
                        config = json.load(f)

                    food_game = FoodGame(rows = config["rows"], cols = config["cols"], 
                                            drl_model_path=self.arguments_to_pass["model_path"], 
                                            solo=self.arguments_to_pass["solo"])

                    food_game.run()
                elif event.ui_element == self.food_game_back_button:
                    self.food_game_play_panel.hide()
                    self.main_page.show()
        
            self.manager.process_events(event)
  

    def draw(self, screen):
        self.manager.draw_ui(screen)
    
    def update(self, delta_time):
        self.manager.update(delta_time)

    def show(self):
        self.food_game_play_panel.show()

    def hide(self):
        self.food_game_play_panel.hide()

    def make_page(self):
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
        self.arguments_to_pass["model_path"] = ""


        # Solo runnning
        self.solo_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,100), (150, 30)), 
                                                            container=self.food_game_play_panel,
                                                            text='Solo', manager=self.manager)
        
        self.solo_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((200, 100), (100, 30)),
                                                                    container=self.food_game_play_panel,
                                                                    manager=self.manager,
                                                                    options_list=["True", "False"],
                                                                    starting_option="False")
        self.arguments_to_pass["solo"] = False



        # Run button
        self.food_game_run_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 550), (150, 50)), 
                                                             container=self.food_game_play_panel, anchors={"centerx": "centerx"},
                                                             text='Run', manager=self.manager)
        # Back button
        self.food_game_back_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 550), (150, 50)),
                                                        container=self.food_game_play_panel, anchors={"centerx": "centerx"},
                                                        text='Back', manager=self.manager)