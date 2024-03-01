from game_playing_ai.games.food_game.food_game import FoodGame

from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
from pettingzoo import ParallelEnv

import functools

class MultiAgentFoodGame(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 5}

    def __init__(self, render_mode:str, rows:int, cols:int, n_food:int, 
                 solo:bool, num_drl_agents:int, num_preprogrammed_agents:int,
                 obstacles:bool, combat:bool):
        self.rows = rows
        self.cols = cols
        self.n_food = n_food
        self.solo = solo
        self.num_drl_agents = num_drl_agents
        self.num_preprogrammed_agents = num_preprogrammed_agents
        self.obstacles = obstacles
        self.combat = combat

        self.observation_space = Box(low=0, high=5, shape=(self.rows, self.cols), dtype=np.int8)

        self.action_spaces = Discrete(4)

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([self.rows, self.cols, 6])
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

    def _get_info(self):
        return {}

    def reset(self, seed=None):
        self.game = FoodGame(self.rows, self.cols, self.n_food, self.render_mode, is_training=True, solo=self.solo,
                             num_drl_agents=self.num_drl_agents, num_preprogrammed_agents=self.num_preprogrammed_agents,
                             obstacles=self.obstacles, combat=self.combat)

        self._food_collected = 0
        self.prev__food_collected = 0

        self._agent_food_collected = [0] * self.num_drl_agents
        self.prev_agent_food_collected = [0] * self.num_drl_agents

        self.playable_agent_food_collected = 0
        self.prev_playable_agent_food_collected = 0

        self.preprogrammed_agent_food_collected = 0
        self.prev_preprogrammed_agent_food_collected = 0

        self.render([None] * self.num_drl_agents)

        self.drl_agent_sprites = self.game.get_drl_agent_sprites()
        observations = []
        for agent in self.drl_agent_sprites:
            observations.append(agent.get_obs(self.game.get_obs()))

        info = self._get_info()

        return observations

    def step(self, actions):
        self.render(actions)
        rewards = []
        terminations = []
        self._food_collected = self.game.drl_agent_sprite_food_collected
        self.playable_agent_food_collected = self.game.playable_agent_food_collected
        self.preprogrammed_agent_food_collected = self.game.preprogrammed_agent_food_collected
        terminated = True if self._food_collected > self.n_food else False
        
        self._agent_food_collected = [agent.food_collected for agent in self.drl_agent_sprites]

        for i, agent in enumerate(self.drl_agent_sprites):

            reward = 0
            reward += self._agent_food_collected[i] - self.prev_agent_food_collected[i]

            if self._food_collected > self.prev__food_collected:
                reward += self._food_collected - self.prev__food_collected
            else:
                reward += -0.01
            
            if self.playable_agent_food_collected > self.prev_playable_agent_food_collected:
                reward -= self.playable_agent_food_collected - self.prev_playable_agent_food_collected
            
            if self.preprogrammed_agent_food_collected > self.prev_preprogrammed_agent_food_collected:
                reward -= self.preprogrammed_agent_food_collected - self.prev_preprogrammed_agent_food_collected

            rewards.append(reward)
            terminations.append(terminated)
        
        observations = []
        for agent in self.drl_agent_sprites:
            observations.append(agent.get_obs(self.game.get_obs()))
        info = self._get_info()
        self.prev__food_collected = self._food_collected
        self.prev_agent_food_collected = self._agent_food_collected[:]
        self.prev_playable_agent_food_collected = self.playable_agent_food_collected
        self.prev_preprogrammed_agent_food_collected = self.preprogrammed_agent_food_collected
        return observations, rewards, terminations, info
    
    def render(self, actions):
        self.game.train(actions)