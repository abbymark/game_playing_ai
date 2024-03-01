from game_playing_ai.games.food_game.agents.drl_agent.DRLAgentSprite import DRLAgentSprite
from game_playing_ai.games.food_game.game_items.environment import Environment
from game_playing_ai.games.food_game.agents.preprogrammed_agent.agent import PreprogrammedAgent
from game_playing_ai.games.food_game.agents.playable_agent.agent import PlayableAgent
from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgent
from game_playing_ai.games.food_game.game_items.food import Food
from game_playing_ai.games.food_game.game_items.obstacle import place_obstacles

import pygame
import pygame_gui
import numpy as np

import sys
import random
import datetime
from typing import Dict, List, Literal
import json



class FoodGame:
    WIDTH = 1200
    HEIGHT = 600

    GAME_WIDTH = 800
    GAME_HEIGHT = 600


    NUM_SPECIFICATIONS = 6
    # Map specification
    # 0: Empty
    # 1: Wall
    # 2: Food
    # 3: Playable agent
    # 4: Preprogrammed agent
    # 5: DRL agent

    # 6: Your agent location(required only when training)


    def __init__(self, rows:int=30, cols:int=40, n_food:int=10, render_mode:Literal["human", "rgb_array"]="human", 
                 is_training:bool=False, solo:bool=False, num_drl_agents:int=1, num_preprogrammed_agents:int=1, drl_model_path:str=None,
                 obstacles:bool=False, combat:bool=False):
        self.render_mode = render_mode
        self.solo = solo
        self.rows = rows
        self.cols = cols

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            
            if is_training:
                self._setup_train_side_panel()
            else:
                self._setup_run_side_panel()
            

            self.run_speed = 5
            self.clock = pygame.time.Clock()
            self.running = True
            self.canvas = pygame.Surface((self.GAME_WIDTH, self.GAME_HEIGHT))
            self.environment = Environment(self.canvas, rows, cols)

        self.map = np.zeros((rows, cols))

        if obstacles:
            self.map = place_obstacles(self.map, rows*cols//10)

        self.playable_agent_food_collected = 0
        self.preprogrammed_agent_food_collected = 0
        self.drl_agent_sprite_food_collected = 0

        # Agents
        self.playable_agent = PlayableAgent(rows, cols)
        self.map = self.playable_agent.set_pos_in_map(self.map)
        self.preprogrammed_agents = [PreprogrammedAgent(rows, cols) for _ in range(num_preprogrammed_agents)]
        for agent in self.preprogrammed_agents:
            self.map = agent.set_pos_in_map(self.map)
        
        if not is_training:
            self.drl_agent = self._load_drl_agent(drl_model_path)

        self.drl_agent_sprites = [DRLAgentSprite(rows, cols) for _ in range(num_drl_agents)]
        for agent in self.drl_agent_sprites:
            self.map = agent.set_pos_in_map(self.map)


        # Food
        self.foods = Food.generate_foods(self.map, n_food)
        for food in self.foods:
            self.map[food.y][food.x] = 2
    
    def _load_drl_agent(self, drl_model_path:str):
        if drl_model_path is None:
            raise ValueError("drl_model_path is required")
        return DQNAgent.load(drl_model_path, is_training=False)


    def run(self):
        while self.running:
            self.time_delta = self.clock.tick(self.run_speed)/1000.0
            events = pygame.event.get()
            # state = np.reshape(self.map, (1, self.map.shape[0] * self.map.shape[1]))

            actions = []
            for agent in self.drl_agent_sprites:
                actions.append(self.drl_agent.act(agent.get_obs(self.map)))
            
            self._update(events, actions)
            self._events(events)
            self._draw()


    def _events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.run_speed_slider:
                    self.run_speed = int(event.value)
            elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                pass

        
            self.manager.process_events(event)

    def _update(self, events, actions):
        prev_pos = self.playable_agent.pos
        self.playable_agent.update(events)
        if prev_pos != self.playable_agent.pos and self.map[self.playable_agent.pos[1]][self.playable_agent.pos[0]] in [0, 2]:
            self.map[prev_pos[1]][prev_pos[0]] = 0
            self.map[self.playable_agent.pos[1]][self.playable_agent.pos[0]] = 3
        else:
            self.playable_agent.pos = prev_pos
        
        for agent in self.preprogrammed_agents:
            prev_pos = agent.pos
            if not self.solo:
                agent.update(self.foods)
            if prev_pos != agent.pos and self.map[agent.pos[1]][agent.pos[0]] in [0, 2]:
                self.map[prev_pos[1]][prev_pos[0]] = 0
                self.map[agent.pos[1]][agent.pos[0]] = 4
            else:
                agent.pos = prev_pos
        
        for agent, action in zip(self.drl_agent_sprites, actions):
            prev_pos = agent.pos
            agent.update(action)
            if prev_pos != agent.pos and self.map[agent.pos[1]][agent.pos[0]] in [0, 2]:
                self.map[prev_pos[1]][prev_pos[0]] = 0
                self.map[agent.pos[1]][agent.pos[0]] = 5
            else:
                agent.pos = prev_pos


        self._check_collisions()
        if self.render_mode == "human":
            self.manager.update(self.time_delta)

    def _draw(self):
        self.screen.fill((0, 0, 0))
        self.environment.draw(self.map)
        self.playable_agent_food_collected_value_label.set_text(str(self.playable_agent_food_collected))
        self.preprogrammed_agent_food_collected_value_label.set_text(str(self.preprogrammed_agent_food_collected))
        self.drl_agent_sprite_food_collected_value_label.set_text(str(self.drl_agent_sprite_food_collected))
        self.screen.blit(self.canvas, self.canvas.get_rect())
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    def _check_collisions(self):
        for food in self.foods:
            if self.playable_agent.pos == food.pos:
                self.playable_agent_food_collected += 1
                self.foods.remove(food)
                self.foods = Food.generate_foods(self.map, 1, self.foods)
                self.map[self.foods[-1].y][self.foods[-1].x] = 2

            for agent in self.preprogrammed_agents:
                if agent.pos == food.pos:
                    self.preprogrammed_agent_food_collected += 1
                    self.foods.remove(food)
                    self.foods = Food.generate_foods(self.map, 1, self.foods)
                    self.map[self.foods[-1].y][self.foods[-1].x] = 2

            for agent in self.drl_agent_sprites:
                if agent.pos == food.pos:
                    agent.increase_food_collected()
                    self.drl_agent_sprite_food_collected += 1
                    self.foods.remove(food)
                    self.foods = Food.generate_foods(self.map, 1, self.foods)
                    self.map[self.foods[-1].y][self.foods[-1].x] = 2

 
    
    def train(self, actions):
        if self.render_mode == "human":
            self.time_delta = self.clock.tick(self.run_speed)/1000.0
            events = pygame.event.get()
            self._update(events, actions)
            self._events(events)
            self._draw()
        elif self.render_mode == "rgb_array":
            self._update(None, actions)
    
    def get_obs(self) -> List[List[int]]:
        return self.map.copy()
    
    def get_drl_agent_sprites(self):
        return self.drl_agent_sprites

    def _setup_train_side_panel(self):
        self.manager = pygame_gui.UIManager((self.WIDTH, self.HEIGHT), "theme.json")
        self.container = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((self.GAME_WIDTH, 0), (self.WIDTH - self.GAME_WIDTH, self.HEIGHT)), 
                                                        manager=self.manager, 
                                                        object_id="#side_panel")
        self.side_panel_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 100)), 
                                                                    container=self.container, 
                                                                    anchors={"centerx": "centerx"},
                                                                    text='Side Panel', manager=self.manager, object_id="#main_title")

        self.run_speed_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 70), (300, 50)), 
                                                            container=self.container, 
                                                            anchors={"centerx": "centerx"},
                                                            manager=self.manager, object_id="#side_panel_label", text="Run Speed")

        self.run_speed_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((0, 120), (300, 50)), 
                                                                        container=self.container, 
                                                                        anchors={"centerx": "centerx"},
                                                                        manager=self.manager, object_id="#run_speed_slider", 
                                                                        start_value=5, value_range=(5, 100))
    
        self.playable_agent_food_collected_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 200), (300, 50)), 
                                                                                container=self.container, 
                                                                                anchors={"centerx": "centerx"},
                                                                                manager=self.manager, object_id="#side_panel_label", text="Playable Agent Food Collected: ")

        self.playable_agent_food_collected_value_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 250), (300, 50)), 
                                                                                    container=self.container, 
                                                                                    anchors={"centerx": "centerx"},
                                                                                    manager=self.manager, object_id="#side_panel_label", text="0")
        
        self.preprogrammed_agent_food_collected_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 300), (300, 50)), 
                                                                                    container=self.container, 
                                                                                    anchors={"centerx": "centerx"},
                                                                                    manager=self.manager, object_id="#side_panel_label", text="Preprogrammed Agent Food Collected: ")
        
        self.preprogrammed_agent_food_collected_value_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 350), (300, 50)), 
                                                                                        container=self.container, 
                                                                                        anchors={"centerx": "centerx"},
                                                                                        manager=self.manager, object_id="#side_panel_label", text="0")
        
        self.drl_agent_sprite_food_collected_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 400), (300, 50)), 
                                                                                    container=self.container, 
                                                                                    anchors={"centerx": "centerx"},
                                                                                    manager=self.manager, object_id="#side_panel_label", text="DRL Agent Food Collected: ")
        
        self.drl_agent_sprite_food_collected_value_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 450), (300, 50)), 
                                                                                        container=self.container, 
                                                                                        anchors={"centerx": "centerx"},
                                                                                        manager=self.manager, object_id="#side_panel_label", text="0")

    def _setup_run_side_panel(self):
        self.manager = pygame_gui.UIManager((self.WIDTH, self.HEIGHT), "theme.json")
        self.container = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((self.GAME_WIDTH, 0), (self.WIDTH - self.GAME_WIDTH, self.HEIGHT)), 
                                                        manager=self.manager, 
                                                        object_id="#side_panel")
        self.side_panel_title_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 0), (300, 100)), 
                                                                    container=self.container, 
                                                                    anchors={"centerx": "centerx"},
                                                                    text='Side Panel', manager=self.manager, object_id="#main_title")

        self.run_speed_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 70), (300, 50)), 
                                                            container=self.container, 
                                                            anchors={"centerx": "centerx"},
                                                            manager=self.manager, object_id="#side_panel_label", text="Run Speed")

        self.run_speed_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((0, 120), (300, 50)), 
                                                                        container=self.container, 
                                                                        anchors={"centerx": "centerx"},
                                                                        manager=self.manager, object_id="#run_speed_slider", 
                                                                        start_value=5, value_range=(5, 100))

        self.playable_agent_food_collected_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 200), (300, 50)), 
                                                                                container=self.container, 
                                                                                anchors={"centerx": "centerx"},
                                                                                manager=self.manager, object_id="#side_panel_label", text="Playable Agent Food: ")

        self.playable_agent_food_collected_value_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 250), (300, 50)), 
                                                                                    container=self.container, 
                                                                                    anchors={"centerx": "centerx"},
                                                                                    manager=self.manager, object_id="#side_panel_label", text="0")
        
        self.preprogrammed_agent_food_collected_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 300), (300, 50)), 
                                                                                    container=self.container, 
                                                                                    anchors={"centerx": "centerx"},
                                                                                    manager=self.manager, object_id="#side_panel_label", text="Preprogrammed Agent Food: ")
        
        self.preprogrammed_agent_food_collected_value_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 350), (300, 50)), 
                                                                                        container=self.container, 
                                                                                        anchors={"centerx": "centerx"},
                                                                                        manager=self.manager, object_id="#side_panel_label", text="0")
        
        self.drl_agent_sprite_food_collected_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 400), (300, 50)), 
                                                                                    container=self.container, 
                                                                                    anchors={"centerx": "centerx"},
                                                                                    manager=self.manager, object_id="#side_panel_label", text="DRL Agent Food: ")
        
        self.drl_agent_sprite_food_collected_value_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((0, 450), (300, 50)), 
                                                                                        container=self.container, 
                                                                                        anchors={"centerx": "centerx"},
                                                                                        manager=self.manager, object_id="#side_panel_label", text="0")



