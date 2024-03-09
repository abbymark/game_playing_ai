import pygame
import pygame_gui

from typing import Dict

class MainPage:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.manager = pygame_gui.UIManager((self.width, self.height), "theme.json")
        self.make_page()

    def process_events(self, events):
        for event in events:
            if event.type ==pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.food_game_play_button:
                    self.main_page.hide()
                    self.food_game_run_page.show()
                elif event.ui_element == self.food_game_train_button:
                    self.main_page.hide()
                    self.food_game_train_page.show()
            
            self.manager.process_events(event)
    
    def draw(self, screen):
        self.manager.draw_ui(screen)
    
    def update(self, delta_time):
        self.manager.update(delta_time)

    def show(self):
        self.main_page.show()

    def hide(self):
        self.main_page.hide()
    
    def set_changeable_pages(self, pages:Dict):
        self.food_game_train_page = pages["food_game_train_page"]
        self.food_game_run_page = pages["food_game_run_page"]

    def make_page(self):
        self.main_page = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (self.width, self.height)), manager=self.manager)
        self.game_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 100)), 
                                                            container=self.main_page, anchors={"centerx": "centerx"},
                                                            text='Game Playing AI', manager=self.manager, object_id="#main_title")
        
        self.food_game_play_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 100), (150, 50)), 
                                                             container=self.main_page, anchors={"centerx": "centerx"},
                                                             text='Play Food Game', manager=self.manager)
        self.food_game_train_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 100), (150, 50)), 
                                                                   container=self.main_page, anchors={"centerx": "centerx"},
                                                             text='Train Food Game', manager=self.manager)