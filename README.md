# Game Playing AI

This project explores the application of multi-agent reinforcement learning (MARL) systems in complex grid world environments against preprogrammed agents. Utilizing Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, it aims to develop, evaluate, and compare the performance of MARL systems. 


## Installation
```
git clone [repo_url]

cd game-playing-ai

poetry install
```


## Usage
```
poetry shell

python main_starter.py
```


## Testing example model
- Run the program
- Click 'Play Food Game'
- Write down the directory of the model you want to test ex: 'data/models/20240311114022_episode_999_obs_F__comb_F__solo_T'
- Click 'Run'

In case you want to try the model in different environment, open up model's config file and change the game componenet you want to test. 
```
data/models/20240311114022_episode_999_obs_F__comb_F__solo_T/config.json


{
    "DRL_algorithm": "DQN",
    "rows": 15,
    "cols": 20,
    "state_size": 2100,
    "action_size": 4,
    "memory_size": 100000,
    "gamma": 0.95,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "target_update_freq": 1000,
    "nn_type": "CNN",
    "solo": true,
    "num_drl_agents": 2,  <-- Chage this for different number of DRL agents
    "num_preprogrammed_agents": 2,  <-- Chage this for different number of preprogrammed agents
    "obstacles": false,  <-- Chage this to true for obstacles
    "combat": false,  <-- Chage this to true for combat mode
    "num_input_channels": 7
}
```