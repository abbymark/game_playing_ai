from game_playing_ai.games.food_game.game_items.environment import Environment
from game_playing_ai.games.food_game.agents.preprogrammed_agent.agent import PreprogrammedAgent
from game_playing_ai.games.food_game.agents.playable_agent.agent import PlayableAgent
from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgentSprite, DQNAgent
from game_playing_ai.games.food_game.game_items.food import Food

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
import random
import datetime
import os

# Map specification
# 0: Empty
# 1: Wall
# 2: Food
# 3: Playable agent
# 4: Preprogrammed agent
# 5: DRL agent


# TODO:
# Gym에서 agent pos, food pos를 받아서 게임을 시작하도록 수정
# self.map 을 통해서 업데이트 와 observation 을 구현
# observation space에 agent 여러개의 위치 구현



class FoodGame:
    WIDTH = 800
    HEIGHT = 600

    def __init__(self, rows=30, cols=40, n_food=10, render_mode="human"):
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Game Playing AI")
            self.clock = pygame.time.Clock()
            self.running = True
            self.canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
            self.environment = Environment(self.canvas, rows, cols)

        self.map = np.zeros((rows, cols))


        # Agents

        self.playable_agent = PlayableAgent(30, 40)
        self.map = self.playable_agent.set_pos_in_map(self.map)
        self.preprogrammed_agent = PreprogrammedAgent(30, 40)
        self.map = self.preprogrammed_agent.set_pos_in_map(self.map)
        self.drl_agent = DQNAgent(rows * cols, 4)
        self.drl_agent_sprite = DQNAgentSprite(30, 40)
        self.map = self.drl_agent_sprite.set_pos_in_map(self.map)


        # Food
        self.foods = Food.generate_foods(rows, cols, n_food)
        for food in self.foods:
            self.map = food.set_pos_in_map(self.map)


    def run(self):
        while self.running:
            self.clock.tick(5)
            events = pygame.event.get()
            action = self.drl_agent.act(self.map)
            self.update(events, action)
            self.events(events)
            self.draw()


    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()

    def update(self, events, action):
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
        
        prev_pos = self.drl_agent_sprite.pos
        self.drl_agent_sprite.update(action)
        if prev_pos != self.drl_agent_sprite.pos:
            self.map[prev_pos[1]][prev_pos[0]] = 0
            self.map[self.drl_agent_sprite.pos[1]][self.drl_agent_sprite.pos[0]] = 5


        self.check_collisions()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.environment.draw(self.map)
        self.screen.blit(self.canvas, self.canvas.get_rect())
        pygame.display.update()

    def check_collisions(self):
        for food in self.foods:
            if self.playable_agent.pos == (food.x, food.y):
                self.playable_agent.food_collected += 1
                self.foods.remove(food)
                self.foods = Food.generate_foods(30, 40, 1, self.foods)
                for food in self.foods:
                    self.map[food.y][food.x] = 2
                break
            elif self.preprogrammed_agent.pos == (food.x, food.y):
                self.preprogrammed_agent.food_collected += 1
                self.foods.remove(food)
                self.foods = Food.generate_foods(30, 40, 1, self.foods)
                for food in self.foods:
                    self.map[food.y][food.x] = 2
                break
            elif self.drl_agent_sprite.pos == (food.x, food.y):
                self.drl_agent_sprite.food_collected += 1
                self.foods.remove(food)
                self.foods = Food.generate_foods(30, 40, 1, self.foods)
                for food in self.foods:
                    self.map[food.y][food.x] = 2
                break
    
    def train(self, action):
        if self.render_mode == "human":
            self.clock.tick(5)
            events = pygame.event.get()
            self.update(events, action)
            self.events(events)
            self.draw()
            return self.map
        elif self.render_mode == "rgb_array":
            self.update(None, action)
            return self.map


class GridFoodGame(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 5}

    def __init__(self, render_mode:str, rows:int, cols:int, n_food:int):
        self.rows = rows
        self.cols = cols
        self.n_food = n_food


        self.observation_space = spaces.Box(low=0, high=5, shape=(self.rows, self.cols), dtype=np.int8)

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self._food_collected = 0
        self.prev__food_collected = 0

        self.playable_agent_food_collected = 0
        self.prev_playable_agent_food_collected = 0

        self.preprogrammed_agent_food_collected = 0
        self.prev_preprogrammed_agent_food_collected = 0

    def _get_obs(self):
        return self.game.map
    
    def _get_info(self):
        return {
            "agent_location": self.game.drl_agent_sprite.pos,
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game = FoodGame(self.rows, self.cols, self.n_food, self.render_mode)

        self._food_collected = 0
        
        self.render(None)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        

        self.render(action)
        self._food_collected = self.game.drl_agent_sprite.food_collected
        terminated = True if self._food_collected == self.n_food else False

        reward = 0
        if self._food_collected > self.prev__food_collected:
            reward = 1
        
        if self.playable_agent_food_collected > self.prev_playable_agent_food_collected:
            reward = -1
        
        if self.preprogrammed_agent_food_collected > self.prev_preprogrammed_agent_food_collected:
            reward = -1


        observation = self._get_obs()
        info = self._get_info()
        
        self.prev__food_collected = self._food_collected
        self.prev_playable_agent_food_collected = self.playable_agent_food_collected
        self.prev_preprogrammed_agent_food_collected = self.preprogrammed_agent_food_collected
        return observation, reward, terminated, False, info

    def render(self, action):
        return self.game.train(action)

    



    

def train_drl_agent():
    env = GridFoodGame("human", 30, 40, 10)
    state_size = env.rows * env.cols
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    sorted_models = sorted(os.listdir("data/models"), reverse=True)
    if len(sorted_models) > 0:
        agent.load(f"data/models/{sorted_models[0]}")


    episodes = 10
    batch_size = 8

    for e in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, *_ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay(batch_size)
        
        agent.save(f"data/models/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_episode_{e}.pt")

if __name__ == "__main__":
    train_drl_agent()