from game_playing_ai.games.food_game.tile_type import TileType
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
from collections import deque


class FoodGame:
    WIDTH = 1200
    HEIGHT = 600

    GAME_WIDTH = 800
    GAME_HEIGHT = 600



    def __init__(self, rows:int=30, cols:int=40, n_food:int=10, render_mode:Literal["human", "rgb_array"]="human", 
                 is_training:bool=False, solo:bool=False, num_drl_agents:int=1, num_preprogrammed_agents:int=1, drl_model_path:str=None,
                 obstacles:bool=False, combat:bool=False):
        self.render_mode = render_mode
        self.solo = solo
        self.rows = rows
        self.cols = cols
        self.combat = combat

        self.respawn_queue = deque([False] * 100, maxlen=100)

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
            self.map[food.y][food.x] = TileType.FOOD
    
    def _load_drl_agent(self, drl_model_path:str):
        if drl_model_path is None:
            raise ValueError("drl_model_path is required")
        return DQNAgent.load(drl_model_path, is_training=False)


    def run(self):
        while self.running:
            self.time_delta = self.clock.tick(self.run_speed)/1000.0
            events = pygame.event.get()
            # state = np.reshape(self.map, (1, self.map.shape[0] * self.map.shape[1]))

            drl_actions = []
            for agent in self.drl_agent_sprites:
                drl_actions.append(self.drl_agent.act(agent.get_obs(self.map)))
            
            preprogrammed_actions = []
            for agent in self.preprogrammed_agents:
                preprogrammed_actions.append(agent.act(agent.get_obs(self.map), self.foods))
            
            self._update(events, drl_actions, preprogrammed_actions)
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

    def _update(self, events, drl_actions, preprogrammed_actions):
        if self.playable_agent.is_alive:
            prev_pos = self.playable_agent.pos
            self.playable_agent.update(events)
            if prev_pos != self.playable_agent.pos and self.map[self.playable_agent.pos[1]][self.playable_agent.pos[0]] in [TileType.EMPTY, TileType.FOOD]:
                self.map[prev_pos[1]][prev_pos[0]] = TileType.EMPTY
                self.map[self.playable_agent.pos[1]][self.playable_agent.pos[0]] = TileType.PLAYABLE_AGENT
            else:
                self.playable_agent.pos = prev_pos
        
        for agent, action in zip(self.preprogrammed_agents, preprogrammed_actions):
            prev_pos = agent.pos
            if not self.solo:
                agent.update(action)
            if prev_pos != agent.pos and self.map[agent.pos[1]][agent.pos[0]] in [TileType.EMPTY, TileType.FOOD]:
                self.map[prev_pos[1]][prev_pos[0]] = TileType.EMPTY
                self.map[agent.pos[1]][agent.pos[0]] = TileType.PREPROGRAMMED_AGENT
            else:
                agent.pos = prev_pos
        
        for agent, action in zip(self.drl_agent_sprites, drl_actions):
            prev_pos = agent.pos
            agent.update(action)
            if prev_pos != agent.pos and self.map[agent.pos[1]][agent.pos[0]] in [TileType.EMPTY, TileType.FOOD]:
                self.map[prev_pos[1]][prev_pos[0]] = TileType.EMPTY
                self.map[agent.pos[1]][agent.pos[0]] = TileType.DRL_AGENT
            else:
                agent.pos = prev_pos


        self._check_collisions()
        if self.combat:
            self._check_enemies_nearby()
            self._remove_dead_agents()
            self._respawn_agents()

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
            if self.playable_agent.is_alive and self.playable_agent.pos == food.pos:
                self.playable_agent_food_collected += 1
                self.foods.remove(food)
                self.foods = Food.generate_foods(self.map, 1, self.foods)
                self.map[self.foods[-1].y][self.foods[-1].x] = TileType.FOOD

            for agent in self.preprogrammed_agents:
                if agent.pos == food.pos:
                    self.preprogrammed_agent_food_collected += 1
                    self.foods.remove(food)
                    self.foods = Food.generate_foods(self.map, 1, self.foods)
                    self.map[self.foods[-1].y][self.foods[-1].x] = TileType.FOOD

            for agent in self.drl_agent_sprites:
                if agent.pos == food.pos:
                    agent.increase_food_collected()
                    self.drl_agent_sprite_food_collected += 1
                    self.foods.remove(food)
                    self.foods = Food.generate_foods(self.map, 1, self.foods)
                    self.map[self.foods[-1].y][self.foods[-1].x] = TileType.FOOD
    
    def _check_enemies_nearby(self):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for drl_agent in self.drl_agent_sprites:
            for x, y in directions:
                if drl_agent.y + y >= 0 and drl_agent.y + y < self.rows and \
                drl_agent.x + x >= 0 and drl_agent.x + x < self.cols and \
                self.map[drl_agent.y + y][drl_agent.x + x] in [TileType.PREPROGRAMMED_AGENT, TileType.PLAYABLE_AGENT]:
                    drl_agent.hp -= 10
            if drl_agent.hp < 100:
                drl_agent.hp += 1
        
        for preprog_agent in self.preprogrammed_agents:
            for x, y in directions:
                if preprog_agent.y + y >= 0 and preprog_agent.y + y < self.rows and \
                preprog_agent.x + x >= 0 and preprog_agent.x + x < self.cols and \
                self.map[preprog_agent.y + y][preprog_agent.x + x] in [TileType.DRL_AGENT, TileType.PLAYABLE_AGENT]:
                    preprog_agent.hp -= 10
            if preprog_agent.hp < 100:
                preprog_agent.hp += 1
        
        if self.playable_agent.is_alive:
            for x, y in directions:
                if self.playable_agent.y + y >= 0 and self.playable_agent.y + y < self.rows and \
                self.playable_agent.x + x >= 0 and self.playable_agent.x + x < self.cols and \
                self.map[self.playable_agent.y + y][self.playable_agent.x + x] in [TileType.DRL_AGENT, TileType.PREPROGRAMMED_AGENT]:
                    self.playable_agent.hp -= 10
            if self.playable_agent.hp < 100:
                self.playable_agent.hp += 1

    def _remove_dead_agents(self):
        survived_drl_agents = []
        for agent in self.drl_agent_sprites:
            if agent.hp > 0:
                survived_drl_agents.append(agent)
            else:
                self.respawn_queue.append('drl_agent')
                self.map[agent.y][agent.x] = 0
        self.drl_agent_sprites = survived_drl_agents

        survived_preprog_agents = []
        for agent in self.preprogrammed_agents:
            if agent.hp > 0:
                survived_preprog_agents.append(agent)
            else:
                self.respawn_queue.append('preprogrammed_agent')
                self.map[agent.y][agent.x] = TileType.EMPTY
        self.preprogrammed_agents = survived_preprog_agents

        if self.playable_agent.hp <= 0 and self.playable_agent.is_alive:
            self.playable_agent.is_alive = False
            self.respawn_queue.append('playable_agent')
            self.map[self.playable_agent.y][self.playable_agent.x] = TileType.EMPTY
        
        
    
    def _respawn_agents(self):
        respawn_agent = self.respawn_queue.popleft()
        if respawn_agent == 'drl_agent':
            self.drl_agent_sprites.append(DRLAgentSprite(self.rows, self.cols))
            self.map = self.drl_agent_sprites[-1].set_pos_in_map(self.map)
        elif respawn_agent == 'preprogrammed_agent':
            self.preprogrammed_agents.append(PreprogrammedAgent(self.rows, self.cols))
            self.map = self.preprogrammed_agents[-1].set_pos_in_map(self.map)
        elif respawn_agent == 'playable_agent':
            self.playable_agent = PlayableAgent(self.rows, self.cols)
            self.map = self.playable_agent.set_pos_in_map(self.map)
            self.playable_agent.is_alive = True
            self.playable_agent.hp = 100
        self.respawn_queue.append(False)
    
    def train(self, drl_actions):
        preprogrammed_actions = []
        for agent in self.preprogrammed_agents:
            preprogrammed_actions.append(agent.act(agent.get_obs(self.map), self.foods))
        if self.render_mode == "human":
            self.time_delta = self.clock.tick(self.run_speed)/1000.0
            events = pygame.event.get()
            self._update(events, drl_actions, preprogrammed_actions)
            self._events(events)
            self._draw()
        elif self.render_mode == "rgb_array":
            self._update(None, drl_actions, preprogrammed_actions)
    
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



