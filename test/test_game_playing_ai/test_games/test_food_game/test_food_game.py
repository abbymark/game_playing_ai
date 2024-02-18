from game_playing_ai.games.food_game.environments.single_agent_food_game import SingleAgentFoodGame
from game_playing_ai.games.food_game.food_game import FoodGame, train_drl_agent

class TestFoodGame:
    def test__init__game__human_render_mode(self):
        game = FoodGame(rows=30, cols=40, n_food=10, render_mode='human', is_training=False, solo_training=False)
        assert game.render_mode == 'human'
        assert hasattr(game, 'run_speed')
        assert hasattr(game, 'clock')
        assert hasattr(game, 'canvas')
        assert hasattr(game, 'environment')

        assert hasattr(game, 'map')
        assert hasattr(game, 'playable_agent')
        assert hasattr(game, 'preprogrammed_agent')
        assert hasattr(game, 'drl_agent')
        assert hasattr(game, 'drl_agent_sprite')
        assert hasattr(game, 'foods')

        has_playable_agent = False
        has_preprogrammed_agent = False
        has_drl_agent_sprite = False
        food_count = 0
        for row in game.map:
            for cell in row:
                if cell == 2:
                    food_count += 1
                elif cell == 3:
                    has_playable_agent = True
                elif cell == 4:
                    has_preprogrammed_agent = True
                elif cell == 5:
                    has_drl_agent_sprite = True
        
        assert food_count == 10
        assert has_playable_agent
        assert has_preprogrammed_agent
        assert has_drl_agent_sprite
    
    def test__init__game__rgb_array_render_mode(self):
        game = FoodGame(rows=30, cols=40, n_food=10, render_mode='rgb_array', is_training=False, solo_training=False)
        assert game.render_mode == 'rgb_array'
        assert not hasattr(game, 'run_speed')
        assert not hasattr(game, 'clock')
        assert not hasattr(game, 'canvas')
        assert not hasattr(game, 'environment')
        
        assert hasattr(game, 'map')
        assert hasattr(game, 'playable_agent')
        assert hasattr(game, 'preprogrammed_agent')
        assert hasattr(game, 'drl_agent')
        assert hasattr(game, 'drl_agent_sprite')
        assert hasattr(game, 'foods')

        has_playable_agent = False
        has_preprogrammed_agent = False
        has_drl_agent_sprite = False
        food_count = 0
        for row in game.map:
            for cell in row:
                if cell == 2:
                    food_count += 1
                elif cell == 3:
                    has_playable_agent = True
                elif cell == 4:
                    has_preprogrammed_agent = True
                elif cell == 5:
                    has_drl_agent_sprite = True
        
        assert food_count == 10
        assert has_playable_agent
        assert has_preprogrammed_agent
        assert has_drl_agent_sprite



class TestGridFoodGame:
    @classmethod
    def setup_class(cls):
        cls.game = SingleAgentFoodGame(rows=30, cols=40, n_food=10, render_mode='human', is_training=False, solo_training=False)

    def test__init__game__human_render_mode(self):
        ...
