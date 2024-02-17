from game_playing_ai.games.food_game.agents.drl_agent.trainer import DQNTrainer, PPOTrainer

import pygame
import pygame_gui

class FoodGameTrainPage:
    def __init__(self, width, height):
        self.manager = pygame_gui.UIManager((width, height), "theme.json")
        self.width = width
        self.height = height
        self.arguments_to_pass = {}
        self.make_page()

    def process_events(self, events):
        for event in events:
            if event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                if event.ui_element == self.memory_size_text_entry:
                    self.arguments_to_pass["memory_size"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.gamma_text_entry:
                    self.arguments_to_pass["gamma"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.epsilon_min_text_entry:
                    self.arguments_to_pass["epsilon_min"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.epsilon_decay_text_entry:
                    self.arguments_to_pass["epsilon_decay"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.learning_rate_text_entry:
                    self.arguments_to_pass["learning_rate"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.batch_size_text_entry:
                    self.arguments_to_pass["batch_size"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.episodes_text_entry:
                    self.arguments_to_pass["episodes"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.nn_update_frequency_text_entry:
                    self.arguments_to_pass["target_update_freq"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.map_size_rows_text_entry:
                    self.arguments_to_pass["map_size_rows"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.map_size_cols_text_entry:
                    self.arguments_to_pass["map_size_cols"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.food_count_text_entry:
                    self.arguments_to_pass["food_count"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.epsilon_text_entry:
                    self.arguments_to_pass["epsilon"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.num_timesteps_text_entry:
                    self.arguments_to_pass["num_timesteps"] = int(event.text) if event.text != "" else 0
                elif event.ui_element == self.lambda_gae_text_entry:
                    self.arguments_to_pass["lambda_gae"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.entropy_coef_text_entry:
                    self.arguments_to_pass["entropy_coef"] = float(event.text) if event.text != "" else 0
                elif event.ui_element == self.epochs_text_entry:
                    self.arguments_to_pass["epochs"] = int(event.text) if event.text != "" else 0

            elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.algorithm_drop_down_menu:
                    self.arguments_to_pass["algorithm"] = event.text
                elif event.ui_element == self.render_drop_down_menu:
                    self.arguments_to_pass["render"] = event.text
                elif event.ui_element == self.nn_type_drop_down_menu:
                    self.arguments_to_pass["nn_type"] = event.text
                elif event.ui_element == self.solo_drop_down_menu:
                    self.arguments_to_pass["solo_training"] = True if event.text == "True" else False

            elif event.type ==pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.train_button:
                    if self.arguments_to_pass["algorithm"] == "DQN":
                        trainer = DQNTrainer()
                        trainer.train_drl_agent(self.arguments_to_pass)
                    elif self.arguments_to_pass["algorithm"] == "PPO":
                        trainer = PPOTrainer()
                        trainer.train_drl_agent(self.arguments_to_pass)
                elif event.ui_element == self.back_button:
                    self.train_config_panel.hide()
                    self.main_menu_panel.show()
                    self.page_state = "main_menu"

            self.manager.process_events(event)
    
    def draw(self, screen):
        self.manager.draw_ui(screen)

    def update(self, delta_time):
        self.manager.update(delta_time)

    def show(self):
        self.train_config_panel.show()
    
    def hide(self):
        self.train_config_panel.hide()


    def make_page(self):
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
                                                                    options_list=["DQN", "PPO"],
                                                                    starting_option="DQN")
        self.arguments_to_pass["algorithm"] = "DQN"

        # Memory Size
        self.memory_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,100), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Memory Size', manager=self.manager)
        self.memory_size_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 100), (100, 30)), 
                                                            container=self.train_config_panel,
                                                            initial_text="100000",
                                                            manager=self.manager)
        self.arguments_to_pass["memory_size"] = 100000

        # Gamma
        self.gamma_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,150), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Gamma', manager=self.manager)
        self.gamma_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 150), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.95",
                                                                    manager=self.manager)
        self.arguments_to_pass["gamma"] = 0.95

        # Epsilon min
        self.epsilon_min_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,200), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Epsilon Min', manager=self.manager)
        self.epsilon_min_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 200), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.01",
                                                                    manager=self.manager)
        self.arguments_to_pass["epsilon_min"] = 0.01

        # Epsilon decay
        self.epsilon_decay_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,250), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Epsilon Decay', manager=self.manager)
        self.epsilon_decay_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 250), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.995",
                                                                    manager=self.manager)
        self.arguments_to_pass["epsilon_decay"] = 0.995

        # Learning rate
        self.learning_rate_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,300), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Learning Rate', manager=self.manager)
        self.learning_rate_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 300), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.0001",
                                                                    manager=self.manager)
        self.arguments_to_pass["learning_rate"] = 0.0001

        # Batch size
        self.batch_size_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,350), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Batch Size', manager=self.manager)
        self.batch_size_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 350), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="128",
                                                                    manager=self.manager)
        self.arguments_to_pass["batch_size"] = 128

        # Episodes
        self.episodes_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,400), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Episodes', manager=self.manager)
        self.episodes_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 400), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="1000",
                                                                    manager=self.manager)
        self.arguments_to_pass["episodes"] = 1000

        # Render dropdown memu(human, rgb_array)
        self.render_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,450), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Render', manager=self.manager)
        self.render_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((200, 450), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["human", "rgb_array"],
                                                                    starting_option="human")
        self.arguments_to_pass["render"] = "human"

        # nn update frequency
        self.nn_update_frequency_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50,500), (150, 30)),
                                                                        container=self.train_config_panel,
                                                                        text='Target Update Freq', manager=self.manager)
        self.nn_update_frequency_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 500), (100, 30)),
                                                                        container=self.train_config_panel,
                                                                        initial_text="1000",
                                                                        manager=self.manager)
        self.arguments_to_pass["target_update_freq"] = 1000



        # Type of Neural Network
        self.nn_type_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,50), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Neural Network', manager=self.manager)
        self.nn_type_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((600, 50), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["DNN", "CNN"],
                                                                    starting_option="CNN")
        self.arguments_to_pass["nn_type"] = "CNN"


        # map size: rows
        self.map_size_rows_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,100), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Map Size: Rows', manager=self.manager)
        self.map_size_rows_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 100), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="15",
                                                                    manager=self.manager)
        self.arguments_to_pass["map_size_rows"] = 15

        # map size: cols
        self.map_size_cols_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,150), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Map Size: Cols', manager=self.manager)
        self.map_size_cols_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 150), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="20",
                                                                    manager=self.manager)
        self.arguments_to_pass["map_size_cols"] = 20

        # food count
        self.food_count_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,200), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Food Count', manager=self.manager)
        self.food_count_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 200), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="10",
                                                                    manager=self.manager)
        self.arguments_to_pass["food_count"] = 10

        # Epsilon
        self.epsilon_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,250), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Epsilon', manager=self.manager)
        self.epsilon_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 250), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.2",
                                                                    manager=self.manager)
        self.arguments_to_pass["epsilon"] = 0.2

        # Num_timesteps
        self.num_timesteps_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,300), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Num Timesteps', manager=self.manager)
        self.num_timesteps_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 300), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="1280",
                                                                    manager=self.manager)
        self.arguments_to_pass["num_timesteps"] = 1280

        # Solo training
        self.solo_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,350), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Solo', manager=self.manager)
        self.solo_drop_down_menu = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect((600, 350), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    manager=self.manager,
                                                                    options_list=["True", "False"],
                                                                    starting_option="True")
        self.arguments_to_pass["solo"] = True

        # lambda_gae
        self.lambda_gae_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,400), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Lambda GAE', manager=self.manager)
        self.lambda_gae_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 400), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.95",
                                                                    manager=self.manager)
        self.arguments_to_pass["lambda_gae"] = 0.95

        # Entropy Coef
        self.entropy_coef_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,450), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Entropy Coef', manager=self.manager)
        self.entropy_coef_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 450), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="0.01",
                                                                    manager=self.manager)
        self.arguments_to_pass["entropy_coef"] = 0.01

        # Epochs
        self.epochs_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((450,500), (150, 30)), 
                                                            container=self.train_config_panel,
                                                            text='Epochs', manager=self.manager)    
        self.epochs_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((600, 500), (100, 30)),
                                                                    container=self.train_config_panel,
                                                                    initial_text="5",
                                                                    manager=self.manager)
        self.arguments_to_pass["epochs"] = 5




        # Train button
        self.train_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((-100, 550), (150, 50)), 
                                                             container=self.train_config_panel, anchors={"centerx": "centerx"},
                                                             text='Train', manager=self.manager)
        # Back button
        self.back_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((100, 550), (150, 50)),
                                                        container=self.train_config_panel, anchors={"centerx": "centerx"},
                                                        text='Back', manager=self.manager)