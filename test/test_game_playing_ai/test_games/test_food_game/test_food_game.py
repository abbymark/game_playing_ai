from game_playing_ai.games.food_game.food_game import FoodGame

import pytest
from unittest.mock import patch, Mock

class TestFoodGame:
    def test__init__(self):
        game = FoodGame(rows=30, cols=40, n_food=10, solo=True, num_drl_agents=2, is_training=True,
                        num_preprogrammed_agents=2, obstacles=True, combat=True)
        assert game.rows == 30
        assert game.cols == 40
        assert game.solo is True
        assert len(game.drl_agent_sprites) == 2
        assert len(game.preprogrammed_agents) == 2
        assert game.combat is True