from main_starter import GameStarter

from unittest.mock import Mock, patch
import pytest

class TestGameStarter:
    @patch('main_starter.pygame')
    @patch('main_starter.MainPage')
    @patch('main_starter.FoodGameTrainPage')
    @patch('main_starter.FoodGameRunPage')
    def test__init__(self,MockFoodGameRunPage, MockFoodGameTrainPage, MockMainPage, MockPygame):
        game_starter = GameStarter()
        MockPygame.init.assert_called_once()
        assert game_starter.width == 800
        assert game_starter.height == 600
        MockPygame.display.set_mode.assert_called_once_with((800, 600))
        MockPygame.display.set_caption.assert_called_once_with("Game Playing AI")
        assert game_starter.running == True
        MockPygame.time.Clock.assert_called_once()
        MockMainPage.assert_called_once_with(800, 600)
        MockMainPage.return_value.show.assert_called_once()
        MockFoodGameTrainPage.assert_called_once_with(800, 600, MockMainPage.return_value)
        MockFoodGameTrainPage.return_value.hide.assert_called_once()
        MockFoodGameRunPage.assert_called_once_with(800, 600, MockMainPage.return_value)
        MockFoodGameRunPage.return_value.hide.assert_called_once()
        MockMainPage.return_value.set_changeable_pages.assert_called_once_with(
            {"food_game_train_page": MockFoodGameTrainPage.return_value, 
             "food_game_run_page": MockFoodGameRunPage.return_value}
        )

    @patch('main_starter.pygame')
    @patch('main_starter.MainPage')
    @patch('main_starter.FoodGameTrainPage')
    @patch('main_starter.FoodGameRunPage')
    def test_run(self, MockFoodGameRunPage, MockFoodGameTrainPage, MockMainPage, MockPygame):
        MockPygame.event.get.return_value = [Mock(type=MockPygame.QUIT)]
        game_starter = GameStarter()
        with pytest.raises(SystemExit):
            game_starter.run()
        MockPygame.quit.assert_called_once()
    
    @patch('main_starter.pygame')
    @patch('main_starter.MainPage')
    @patch('main_starter.FoodGameTrainPage')
    @patch('main_starter.FoodGameRunPage')
    def test_draw(self, MockFoodGameRunPage, MockFoodGameTrainPage, MockMainPage, MockPygame):
        game_starter = GameStarter()
        
        game_starter.screen = Mock()
        game_starter.main_page.draw = Mock()
        game_starter.food_game_train_page.draw = Mock()
        game_starter.food_game_run_page.draw = Mock()
        game_starter._draw()

        game_starter.main_page.draw.assert_called_once_with(game_starter.screen)
        game_starter.food_game_train_page.draw.assert_called_once_with(game_starter.screen)
        game_starter.food_game_run_page.draw.assert_called_once_with(game_starter.screen)
        MockPygame.display.update.assert_called_once()

    @patch('main_starter.pygame')
    @patch('main_starter.MainPage')
    @patch('main_starter.FoodGameTrainPage')
    @patch('main_starter.FoodGameRunPage')
    @patch('main_starter.sys')
    def test_events(self, MockSys, MockFoodGameRunPage, MockFoodGameTrainPage, MockMainPage, MockPygame):
        game_starter = GameStarter()
        game_starter.main_page.process_events = Mock()
        game_starter.food_game_train_page.process_events = Mock()
        game_starter.food_game_run_page.process_events = Mock()

        quit_event = Mock(type=MockPygame.QUIT)
        MockPygame.event.get.return_value = [quit_event]

        game_starter._events([quit_event])

        game_starter.main_page.process_events.assert_called_once()
        game_starter.food_game_train_page.process_events.assert_called_once()
        game_starter.food_game_run_page.process_events.assert_called_once()
        
        assert not game_starter.running
        MockPygame.quit.assert_called_once()
        MockSys.exit.assert_called_once()
    
    @patch('main_starter.pygame')
    @patch('main_starter.MainPage')
    @patch('main_starter.FoodGameTrainPage')
    @patch('main_starter.FoodGameRunPage')
    def test_update(self, MockFoodGameRunPage, MockFoodGameTrainPage, MockMainPage, MockPygame):
        game_starter = GameStarter()
        game_starter.main_page.update = Mock()
        game_starter.food_game_train_page.update = Mock()
        game_starter.food_game_run_page.update = Mock()

        game_starter.time_delta = 0.1
        game_starter._update()

        game_starter.main_page.update.assert_called_once()
        game_starter.food_game_train_page.update.assert_called_once()
        game_starter.food_game_run_page.update.assert_called_once()