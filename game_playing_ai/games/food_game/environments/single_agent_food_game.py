from game_playing_ai.games.food_game.food_game import FoodGame


import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SingleAgentFoodGame(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 5}

    def __init__(self, render_mode:str, rows:int, cols:int, n_food:int, solo:bool):
        self.rows = rows
        self.cols = cols
        self.n_food = n_food
        self.solo = solo

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


    def _get_info(self):
        return {
            "agent_location": self.game.drl_agent_sprite.pos,
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game = FoodGame(self.rows, self.cols, self.n_food, self.render_mode, is_training=True, solo=self.solo)

        self._food_collected = 0
        self.prev__food_collected = 0

        self.playable_agent_food_collected = 0
        self.prev_playable_agent_food_collected = 0

        self.preprogrammed_agent_food_collected = 0
        self.prev_preprogrammed_agent_food_collected = 0

        self.render(None)

        observation = self.game.get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        self.render(action)
        self._food_collected = self.game.drl_agent_sprite.food_collected
        self.playable_agent_food_collected = self.game.playable_agent.food_collected
        self.preprogrammed_agent_food_collected = self.game.preprogrammed_agent.food_collected
        terminated = True if self._food_collected == self.n_food else False

        reward = 0
        if self._food_collected > self.prev__food_collected:
            reward += 1
        else:
            reward += -0.01

        if self.playable_agent_food_collected > self.prev_playable_agent_food_collected:
            reward += -0.1

        if self.preprogrammed_agent_food_collected > self.prev_preprogrammed_agent_food_collected:
            reward += -0.1

        observation = self.game.get_obs()
        info = self._get_info()

        self.prev__food_collected = self._food_collected
        self.prev_playable_agent_food_collected = self.playable_agent_food_collected
        self.prev_preprogrammed_agent_food_collected = self.preprogrammed_agent_food_collected
        return observation, reward, terminated, False, info

    def render(self, action):
        self.game.train(action)