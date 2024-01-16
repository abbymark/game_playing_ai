from game_playing_ai.games.food_game.game_items.environment import Environment
from game_playing_ai.games.food_game.agents.preprogrammed_agent.agent import PreprogrammedAgent
from game_playing_ai.games.food_game.agents.playable_agent.agent import PlayableAgent
from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgentSprite
from game_playing_ai.games.food_game.game_items.food import Food

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
import random

# Map specification
# 0: Empty
# 1: Wall
# 2: Food
# 3: Playable agent
# 4: Preprogrammed agent
# 5: DRL agent


# TODO:
# 시작할때 agent 위치가 나와도록 해야함 즉 environment 에 map의 initial state를 넣어줘야함
# Gym에서 agent pos, food pos를 받아서 게임을 시작하도록 수정
# self.map 을 통해서 업데이트 와 observation 을 구현
# observation space에 agent 여러개의 위치 구현



class FoodGame:
    WIDTH = 800
    HEIGHT = 600

    def __init__(self, rows=30, cols=40, n_food=10, render_mode=None, drl_agent_pos=None, food_pos=None):
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Game Playing AI")
            self.clock = pygame.time.Clock()
            self.running = True
            self.canvas = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.map = np.zeros((rows, cols))


        # Agents
        self.playable_agent = PlayableAgent(30, 40)
        
        self.preprogrammed_agent = PreprogrammedAgent(30, 40)

        # if not drl_agent_pos:
        #     drl_agent_pos = (random.randint(0, cols - 1), random.randint(0, rows - 1))
        # self.map[drl_agent_pos[1]][drl_agent_pos[0]] = 5
        # self.drl_agent = DQNAgentSprite(self.canvas, 30, 40, drl_agent_pos[0], drl_agent_pos[1])


        # Food
        self.foods = Food.generate_foods(rows, cols, n_food)
        for food in self.foods:
            self.map[food.y][food.x] = 2

        # Environment
        self.environment = Environment(self.canvas, rows, cols)
    


    def run(self):
        while self.running:
            if self.render_mode == "human":
                self.clock.tick(5)
                events = pygame.event.get()
                self.update(events)
                self.events(events)
                self.draw()

    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()

    def update(self, events):
        prev_pos = self.playable_agent.pos
        self.playable_agent.update(events)
        if prev_pos != self.playable_agent.pos:
            self.map[prev_pos[1]][prev_pos[0]] = 0
            self.map[self.playable_agent.pos[1]][self.playable_agent.pos[0]] = 3
        


        prev_pos = self.preprogrammed_agent.pos
        self.preprogrammed_agent.update(self.foods)
        if prev_pos != self.preprogrammed_agent.pos:
            self.map[prev_pos[1]][prev_pos[0]] = 0
            self.map[self.preprogrammed_agent.pos[1]][self.preprogrammed_agent.pos[0]] = 4

        self.check_collisions()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.environment.draw(self.map)
        self.screen.blit(self.canvas, self.canvas.get_rect())
        pygame.display.update()

    def check_collisions(self):
        for food in self.foods:
            if self.playable_agent.pos == (food.x, food.y):
                self.foods.remove(food)
                # self.map[food.y][food.x] = 0
                self.foods = Food.generate_foods(30, 40, 1, self.foods)
                for food in self.foods:
                    self.map[food.y][food.x] = 2
                break
            if self.preprogrammed_agent.pos == (food.x, food.y):
                self.foods.remove(food)
                # self.map[food.y][food.x] = 0
                self.foods = Food.generate_foods(30, 40, 1, self.foods)
                for food in self.foods:
                    self.map[food.y][food.x] = 2
                break


class GridFoodGame(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 5}

    def __init__(self, render_mode, rows, cols, n_food):
        self.rows = rows
        self.cols = cols
        self.n_food = n_food


        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=int),  # for POMDP, this should be removed
                "foods": spaces.Box(low=0, high=max(self.width, self.height), shape=(self.n_food, 2), dtype=int),  # Should be relative to agent im POMDP
                # "opponent": spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=int)  # for POMDP, this should be removed
            }
        )

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.window = None
        self.clock = None

    
    def _get_obs(self):
        return {"agent": self._agent_location, "foods": self._food_locations}
    
    def _get_info(self):
        return {
            "nearest_food_distance": np.linalg.norm(self._agent_location - self._food_locations, axis=1).min()
        }

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._agent_location = np.array([random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)])

        self._food_locations = np.array([[random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)] for _ in range(self.n_food)])

        self._food_collected = 0

        while np.any(self._food_locations == self._agent_location):
            self._agent_location = np.array([random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)])
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(self._agent_location + direction, 0, np.array([self.cols - 1, self.rows - 1]))

        terminated = True if self._food_collected == self.n_food else False

        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            FoodGame(self.rows, self.cols, self.n_food, self.render_mode).run()