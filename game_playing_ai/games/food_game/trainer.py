from game_playing_ai.games.food_game.food_game import GridFoodGame
from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgent

from typing import Dict
import datetime

class Trainer:
    def __init__(self):
        pass

    def train_drl_agent(self, config:Dict[str, str]):
        env = GridFoodGame(config["render"], config['map_size_rows'], config['map_size_cols'], config['food_count'], config['solo_training'])
        action_size = env.action_space.n

        agent = DQNAgent(env.rows, env.cols, action_size, config['memory_size'], config['gamma'], config['epsilon_min'], config['epsilon_decay'], 
                        config['learning_rate'],  config['target_update_freq'], config['nn_type'], is_training=True)

        # sorted_models = sorted(os.listdir("data/models"), reverse=True)
        # if len(sorted_models) > 0:
        #     agent.load(f"data/models/{sorted_models[0]}")


        step_count = 0

        for e in range(config["episodes"]):
            state, info = env.reset()

            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, *_ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                agent.replay(config["batch_size"])
                step_count += 1

                if step_count % 10000 == 0 and step_count > config["memory_size"]:
                    agent.save(f"data/models/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_episode_{e}")
            if len(agent.memory) == config['memory_size']:
                agent.update_epsilon()