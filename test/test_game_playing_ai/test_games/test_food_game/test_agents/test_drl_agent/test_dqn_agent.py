from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgent

import pytest

class TestDQNAgent:
    def test_init(self):
        agent = DQNAgent(10, 10, 4)
        assert agent.rows == 10
        assert agent.cols == 10
        assert agent.state_size == 10 * 10 * 6
        assert agent.action_size == 4
        assert agent.memory_size == 10000
        assert agent.memory.maxlen == 10000
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0
        assert agent.epsilon_min == 0.01
        assert agent.epsilon_decay == 0.999999
        assert agent.batch_size == 32
        assert agent.learning_rate == 0.0001
        assert agent.target_update_freq == 100
        assert agent.nn_type == "CNN"
        assert agent.solo == True
        assert agent.num_drl_agents == 1
        assert agent.num_preprogrammed_agents == 0
        assert agent.obstacles == False
        assert agent.combat == False
        assert agent.is_training == True
        assert agent.num_input_channels == 6
        assert agent.model is not None
        assert agent.target_model is not None
        assert agent.optimizer is not None
        assert agent.criterion is not None
        assert agent.update_step == 0
        assert agent.loss_sum == 0
    
    