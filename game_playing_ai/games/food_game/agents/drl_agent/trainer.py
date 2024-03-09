from game_playing_ai.games.food_game.environments.multi_agent_food_game import MultiAgentFoodGame
from game_playing_ai.games.food_game.agents.drl_agent.dqn_agent import DQNAgent
from game_playing_ai.games.food_game.agents.drl_agent.ppo_agent import PPOAgent, Memory

from typing import Dict
import datetime

import wandb

class DQNTrainer:
    def __init__(self):
        pass

    def train_drl_agent(self, config:Dict[str, str]):
        env = MultiAgentFoodGame(config["render"], config['map_size_rows'], config['map_size_cols'], 
                        config['food_count'], config['solo'], config['num_drl_agents'], config['num_preprogrammed_agents'],
                        config['obstacles'], config['combat'])

        action_size = 4


        agent = DQNAgent(env.rows, env.cols, action_size, 
                         config['memory_size'], config['gamma'], 
                         config['epsilon_min'], config['epsilon_decay'], 
                        config['batch_size'], config['learning_rate'],  
                        config['target_update_freq'], config['nn_type'], 
                        is_training=True, num_input_channels=7)
        

        for e in range(config["episodes"]):
            states = env.reset()

            step_count = 0
            done = False
            while not done:
                actions = []
                for state in states:
                    action = agent.act(state)
                    actions.append(action)
                
                next_states, rewards, dones, *_ = env.step(actions)
                done = any(dones)
                for i in range(len(states)):
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                states = next_states

                agent.replay()
                step_count += 1

            if len(agent.memory) == config["memory_size"]:
                agent.save({
                    'num_drl_agents': config['num_drl_agents'],
                    'num_preprogrammed_agents': config['num_preprogrammed_agents']
                }, f"data/models/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_episode_{e}")
                log = agent.get_log()
                log.update({
                    "step_count": step_count,
                })
                wandb.log(log)
            if len(agent.memory) == config['memory_size']:
                agent.update_epsilon()

class PPOTrainer:
    def __init__(self):
        pass

    def train_drl_agent(self, config:Dict[str, str]):
        memory = Memory(config['gamma'], config['lambda_gae'])

        env = MultiAgentFoodGame(config["render"], config['map_size_rows'], config['map_size_cols'], 
                        config['food_count'], config['solo'], config['num_drl_agents'], config['num_preprogrammed_agents'],
                        config['obstacles'], config['combat'])
        action_size = 4
                             
        agent = PPOAgent(env.rows, env.cols, action_size, 
                         config['gamma'], config['lambda_gae'], 
                         config['epsilon'], config['batch_size'], 
                         config['learning_rate'], config['nn_type'], 
                         is_training=True, num_input_channels=7, 
                         entropy_coef=config['entropy_coef'], epochs=config['epochs'])
        
        for e in range(config["episodes"]):
            states= env.reset()

            step_count = 0
            done = False
            while not done:
                actions = []
                log_probs = []
                values = []
                for state in states:
                    action, log_prob, value = agent.act(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                
                next_states, rewards, dones, *_ = env.step(actions)
                done = any(dones)
                memory.remember(states, actions, log_probs, values, rewards, dones)
                states = next_states

                step_count += 1
                if step_count % config['num_timesteps'] == 0:
                    agent.update(memory)
                    memory.clear_memory()


            agent.save(f"data/models/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_episode_{e}")
            log = agent.get_log()
            log.update({
                "step_count": step_count,
            })
            wandb.log(log)
    







