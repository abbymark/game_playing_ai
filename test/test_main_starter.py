import pygame_gui

from main_starter import GameStarter

class TestGameStarter:
    @classmethod
    def setup_class(cls):
        cls.game_starter = GameStarter()

    def test__main_menu_panel__contain_all_elements(self):
        self.game_starter.make_main_page()
        assert isinstance(self.game_starter.main_page, pygame_gui.elements.UIPanel)
        assert isinstance(self.game_starter.game_title_label, pygame_gui.elements.UILabel)
        assert isinstance(self.game_starter.food_game_button, pygame_gui.elements.UIButton)
        assert isinstance(self.game_starter.food_game_train_button, pygame_gui.elements.UIButton)
    
    def test__dqn_train_config_panel__contain_all_elements(self):
        self.game_starter._dqn_train_config_panel()
        assert isinstance(self.game_starter.train_config_panel, pygame_gui.elements.UIPanel)
        assert isinstance(self.game_starter.train_config_title_label, pygame_gui.elements.UILabel)