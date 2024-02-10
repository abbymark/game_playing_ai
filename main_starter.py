from game_playing_ai.games.food_game.food_game import FoodGame
from game_playing_ai.games.food_game.trainer import Trainer

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
        self._main_menu_panel()
        
        self.page_state = "main_menu"  # main_menu, food_game_play_panel, food_game_train_panel

        self.food_game_train_config = {}
        self.food_game_play_config = {}

        pygame.display.set_caption("Game Playing AI")
        self.clock = pygame.time.Clock()
        self.running = True


    def run(self):
        while self.running:
            self.time_delta = self.clock.tick(60)/1000.0
            events = pygame.event.get()
            self._events(events)
            self._update(events)
            self._draw()
    
    def _draw(self):
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    def _events(self, events):
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
                        self._dqn_train_config_panel()
                        self.page_state = "food_game_train_panel"

                if self.page_state == "food_game_play_panel":
                    if event.ui_element == self.food_game_run_button:
                        with open(f"{self.food_game_play_config['model_path']}/config.json", "r") as f:
                            config = json.load(f)

                        food_game = FoodGame(rows = config["rows"], cols = config["cols"], 
                                             drl_model_path=self.food_game_play_config["model_path"], 
                                             solo=self.food_game_play_config["solo"], 
                                             use_featured_states=config["use_featured_states"])

                        food_game.run()
                    elif event.ui_element == self.food_game_back_button:
                        self.food_game_play_panel.hide()
                        self.main_menu_panel.show()
                        self.page_state = "main_menu"

                if self.page_state == "food_game_train_panel":
                    if event.ui_element == self.train_button:
                        trainer = Trainer()
                        trainer.train_drl_agent(self.food_game_train_config)
                    elif event.ui_element == self.back_button:
                        self.train_config_panel.hide()
                        self.main_menu_panel.show()
                        self.page_state = "main_menu"



            elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if self.page_state == "food_game_play_panel":
                    if event.ui_element == self.solo_drop_down_menu:
                        self.food_game_play_config["solo"] = True if event.text == "True" else False


                elif self.page_state == "food_game_train_panel":
                    if event.ui_element == self.algorithm_drop_down_menu:
                        self.food_game_train_config["algorithm"] = event.text
                    elif event.ui_element == self.render_drop_down_menu:
                        self.food_game_train_config["render"] = event.text
                    elif event.ui_element == self.nn_type_drop_down_menu:
                        self.food_game_train_config["nn_type"] = event.text
                    elif event.ui_element == self.solo_drop_down_menu:
                        self.food_game_train_config["solo_training"] = True if event.text == "True" else False
                    elif event.ui_element == self.use_featured_states_drop_down_menu:
                        self.food_game_train_config["use_featured_states"] = True if event.text == "True" else False
            elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:

                # Food Game Play Config
                if self.page_state == "food_game_play_panel":
                    if event.ui_element == self.model_path_text_entry:
                        self.food_game_play_config["model_path"] = event.text

                # Food Game Train Config
                elif self.page_state == "food_game_train_panel":
                    if event.ui_element == self.memory_size_text_entry:
                        self.food_game_train_config["memory_size"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.gamma_text_entry:
                        self.food_game_train_config["gamma"] = float(event.text) if event.text != "" else 0
                    elif event.ui_element == self.epsilon_min_text_entry:
                        self.food_game_train_config["epsilon_min"] = float(event.text) if event.text != "" else 0
                    elif event.ui_element == self.epsilon_decay_text_entry:
                        self.food_game_train_config["epsilon_decay"] = float(event.text) if event.text != "" else 0
                    elif event.ui_element == self.learning_rate_text_entry:
                        self.food_game_train_config["learning_rate"] = float(event.text) if event.text != "" else 0
                    elif event.ui_element == self.batch_size_text_entry:
                        self.food_game_train_config["batch_size"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.episodes_text_entry:
                        self.food_game_train_config["episodes"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.nn_update_frequency_text_entry:
                        self.food_game_train_config["target_update_freq"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.map_size_rows_text_entry:
                        self.food_game_train_config["map_size_rows"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.map_size_cols_text_entry:
                        self.food_game_train_config["map_size_cols"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.food_count_text_entry:
                        self.food_game_train_config["food_count"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.agent_input_col_size_text_entry:
                        self.food_game_train_config["agent_input_col_size"] = int(event.text) if event.text != "" else 0
                    elif event.ui_element == self.agent_input_row_size_text_entry:
                        self.food_game_train_config["agent_input_row_size"] = int(event.text) if event.text != "" else 0


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



    def _dqn_train_config_panel(self):
        self.train_config_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (self.width, self.height)), 
                                                              manager=self.manager, object_id="#config_panel")
        self.train_config_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 50)), 
                                                            container=self.train_config_panel, anchors={"centerx": "centerx"},
                                                            text='Train Config', manager=self.manager, object_id="#main_title")
        
        # Algorithm
        self.algorithm_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,50), (150, 30)),
                                                              container=self.train_config_panel,
                                                              text='Algorithm', manager=self.manager)
        self.algorithm_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((200, 50), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["DQN"],
                                                                    starting_option="DQN")
        self.food_game_train_config["algorithm"] = "DQN"

        # Memory Size
        self.memory_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,100), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Memory Size', manager=self.manager)
        self.memory_size_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 100), (100, 30)), 
                                                            container=self.train_config_panel,
                                                            initial_text="100000",
                                                            manager=self.manager)
        self.food_game_train_config["memory_size"] = 100000

        # Gamma
        self.gamma_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,150), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Gamma', manager=self.manager)
        self.gamma_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 150), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.95",
                                                                    manager=self.manager)
        self.food_game_train_config["gamma"] = 0.95

        # Epsilon min
        self.epsilon_min_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,200), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Epsilon Min', manager=self.manager)
        self.epsilon_min_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 200), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.01",
                                                                    manager=self.manager)
        self.food_game_train_config["epsilon_min"] = 0.01

        # Epsilon decay
        self.epsilon_decay_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,250), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Epsilon Decay', manager=self.manager)
        self.epsilon_decay_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 250), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.995",
                                                                    manager=self.manager)
        self.food_game_train_config["epsilon_decay"] = 0.995

        # Learning rate
        self.learning_rate_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,300), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Learning Rate', manager=self.manager)
        self.learning_rate_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 300), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.0001",
                                                                    manager=self.manager)
        self.food_game_train_config["learning_rate"] = 0.0001

        # Batch size
        self.batch_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,350), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Batch Size', manager=self.manager)
        self.batch_size_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 350), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="128",
                                                                    manager=self.manager)
        self.food_game_train_config["batch_size"] = 128

        # Episodes
        self.episodes_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,400), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Episodes', manager=self.manager)
        self.episodes_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 400), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="1000",
                                                                    manager=self.manager)
        self.food_game_train_config["episodes"] = 1000

        # Render dropdown memu(human, rgb_array)
        self.render_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,450), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Render', manager=self.manager)
        self.render_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((200, 450), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["human", "rgb_array"],
                                                                    starting_option="human")
        self.food_game_train_config["render"] = "human"

        # nn update frequency
        self.nn_update_frequency_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,500), (150, 30)),
                                                                        container=self.train_config_panel,
                                                                        text='Target Update Freq', manager=self.manager)
        self.nn_update_frequency_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 500), (100, 30)),
                                                                        container=self.train_config_panel,
                                                                        initial_text="1000",
                                                                        manager=self.manager)
        self.food_game_train_config["target_update_freq"] = 1000



        # Type of Neural Network
        self.nn_type_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,50), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Neural Network', manager=self.manager)
        self.nn_type_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((600, 50), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["DNN", "CNN"],
                                                                    starting_option="DNN")
        self.food_game_train_config["nn_type"] = "DNN"


        # map size: rows
        self.map_size_rows_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,100), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Map Size: Rows', manager=self.manager)
        self.map_size_rows_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 100), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="15",
                                                                    manager=self.manager)
        self.food_game_train_config["map_size_rows"] = 15

        # map size: cols
        self.map_size_cols_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,150), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Map Size: Cols', manager=self.manager)
        self.map_size_cols_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 150), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="20",
                                                                    manager=self.manager)
        self.food_game_train_config["map_size_cols"] = 20

        # food count
        self.food_count_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,200), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Food Count', manager=self.manager)
        self.food_count_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 200), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="10",
                                                                    manager=self.manager)
        self.food_game_train_config["food_count"] = 10

        # Agent input colum size
        self.agent_input_col_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,250), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Agent Input Col Size', manager=self.manager)
        self.agent_input_col_size_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 250), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="5",
                                                                    manager=self.manager)
        self.food_game_train_config["agent_view_col_size"] = 5
        self.agent_input_col_size_label.disable()
        self.agent_input_col_size_text_entry.disable()

        # Agent input row size
        self.agent_input_row_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,300), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Agent Input Row Size', manager=self.manager)
        self.agent_input_row_size_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 300), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="5",
                                                                    manager=self.manager)
        self.food_game_train_config["agent_view_row_size"] = 5
        self.agent_input_row_size_label.disable()
        self.agent_input_row_size_text_entry.disable()

        # Solo training
        self.solo_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,350), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Solo', manager=self.manager)
        self.solo_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((600, 350), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["True", "False"],
                                                                    starting_option="True")
        self.food_game_train_config["solo"] = True

        # Uses featured states
        self.use_featured_states_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,400), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Use Featured States', manager=self.manager)
        self.use_featured_states_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((600, 400), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["True", "False"],
                                                                    starting_option="False")
        self.food_game_train_config["use_featured_states"] = False


        # Train button
        self.train_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 550), (150, 50)), 
                                                             container=self.train_config_panel, anchors={"centerx": "centerx"},
                                                             text='Train', manager=self.manager)
        # Back button
        self.back_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 550), (150, 50)),
                                                        container=self.train_config_panel, anchors={"centerx": "centerx"},
                                                        text='Back', manager=self.manager)
        
        
        
    
    def _update(self, events):
        self.manager.update(self.time_delta)






if __name__ == "__main__":

    # train_drl_agent()

    game = GameStarter()
    game.run()