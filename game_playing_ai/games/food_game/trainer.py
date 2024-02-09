from game_playing_ai.games.food_game.food_game import GridFoodGame
from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgent

from typing import Dict
import datetime

import wandb

class Trainer:
    def __init__(self):
        pass

    def train_drl_agent(self, config:Dict[str, str]):
        env = GridFoodGame(config["render"], config['map_size_rows'], config['map_size_cols'], 
                           config['food_count'], config['solo'], config['use_featured_states'])
        action_size = env.action_space.n
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1]

        agent = DQNAgent(env.rows, env.cols, state_size, action_size, 
                         config['memory_size'], config['gamma'], 
                         config['epsilon_min'], config['epsilon_decay'], 
                        config['batch_size'], config['learning_rate'],  
                        config['target_update_freq'], config['nn_type'], 
                        is_training=True, use_featured_states=config['use_featured_states'])

        # sorted_models = sorted(os.listdir("data/models"), reverse=True)
        # if len(sorted_models) > 0:
        #     agent.load(f"data/models/{sorted_models[0]}")



        for e in range(config["episodes"]):
            state, info = env.reset()

            step_count = 0
            done = False
            while not done:
                action = agent.act(state)
                
                next_state, reward, done, *_ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                agent.replay()
                step_count += 1

            if len(agent.memory) == config["memory_size"]:
                agent.save(f"data/models/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_episode_{e}")
                log = agent.get_log()
                log.update({
                    "step_count": step_count,
                })
                wandb.log(log)
            if len(agent.memory) == config['memory_size']:
                agent.update_epsilon()