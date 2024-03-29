# DQN codes were referenced from https://github.com/patrickloeber/snake-ai-pytorch and modified to fit the game


from game_playing_ai.games.food_game.agents.drl_agent.networks.dqn_networks import DNN
from game_playing_ai.games.food_game.agents.drl_agent.networks.dqn_networks import CNN

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from collections import deque
import os
import json
import random
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, rows:int, cols:int, action_size:int, memory_size:int=10000, 
                 gamma:float=0.95, epsilon_min:float=0.01, epsilon_decay:float=0.999999, batch_size:int=32,
                 learning_rate:float=0.0001, target_update_freq:str=100, nn_type:str="CNN", solo:bool=True,
                 num_drl_agents:int=1, num_preprogrammed_agents:int=0, obstacles:bool=False, combat:bool=False,
                 is_training:bool=True, num_input_channels=6) -> None:
        self.rows = rows
        self.cols = cols
        self.state_size = rows * cols * num_input_channels
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.nn_type = nn_type
        self.solo = solo
        self.num_drl_agents = num_drl_agents
        self.num_preprogrammed_agents = num_preprogrammed_agents
        self.obstacles = obstacles
        self.combat = combat
        self.is_training = is_training
        self.num_input_channels = num_input_channels

        self.model = self.get_model(self.nn_type)
        self.target_model = self.get_model(self.nn_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.update_step = 0

        self.loss_sum = 0

        if is_training:
            wandb.login()
            wandb.init(
                project="food_game",
                name=f"DQN_r:{self.rows}_c:{self.cols}_n_drl:{num_drl_agents}_n_pre{num_preprogrammed_agents}_obs:{obstacles}_combat:{combat}_solo:{solo}",
                config={
                    "DRL_algorithm": "DQN",
                    "rows": self.rows,
                    "cols": self.cols,
                    "state_size": int(self.state_size),
                    "action_size": int(self.action_size),
                    "memory_size": self.memory_size,
                    "gamma": self.gamma,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "target_update_freq": self.target_update_freq,
                    "nn_type": self.nn_type,
                    "solo": self.solo,
                    "num_drl_agents": self.num_drl_agents,
                    "num_preprogrammed_agents": self.num_preprogrammed_agents,
                    "obstacles": self.obstacles,
                    "combat": self.combat,
                    "num_input_channels": self.num_input_channels,
                }
            )

    def get_model(self, nn_type:str):
        if nn_type == 'DNN':
            return DNN(self.state_size, self.action_size).to(device)
        elif nn_type == 'CNN':
            return CNN(self.num_input_channels, self.action_size).to(device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and self.is_training:
            return random.randrange(self.action_size)
        state = torch.LongTensor(state).unsqueeze(0).to(device)
        flat_next_states = state.view(state.shape[0], -1)
        one_hot_flat_next_states = nn.functional.one_hot(flat_next_states, num_classes=7).float()
        state = one_hot_flat_next_states.view(*state.shape, -1)
        state = state.permute(0, 3, 1, 2).contiguous()
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        self.update_step += 1

        if len(self.memory) < self.memory_size or len(self.memory) < self.batch_size:
            return
    
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Convert list of numpy arrays to single numpy array for each type of data
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([float(x[4]) for x in minibatch])
        
        # Convert numpy arrays to PyTorch tensors
        states = torch.LongTensor(states).to(device)
        flat_states = states.view(states.shape[0], -1)
        one_hot_flat_states = nn.functional.one_hot(flat_states, num_classes=self.num_input_channels).float()
        states = one_hot_flat_states.view(*states.shape, -1)
        states = states.permute(0, 3, 1, 2).contiguous()

        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        next_states = torch.LongTensor(next_states).to(device)
        flat_next_states = next_states.view(next_states.shape[0], -1)
        one_hot_flat_next_states = nn.functional.one_hot(flat_next_states, num_classes=self.num_input_channels).float()
        next_states = one_hot_flat_next_states.view(*next_states.shape, -1)
        next_states = next_states.permute(0, 3, 1, 2).contiguous()

        dones = torch.FloatTensor(dones).to(device)
        # Predict Q-values for starting states
        Q_values = self.model(states)
        action_Q_values = Q_values.gather(1, actions).squeeze()

        # Predict next Q-values for next states
        next_Q_values = self.target_model(next_states)
        max_next_Q_values = next_Q_values.max(1)[0]

        # Compute the target Q values
        targets = rewards + (self.gamma * max_next_Q_values * (1 - dones))

        # Loss calculation
        loss = self.criterion(action_Q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_sum += loss.item()        

        # update target network
        if self.update_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    

    def get_log(self) -> Dict[str, float]:
        log = {
            "loss": self.loss_sum,
            "epsilon": self.epsilon
        }
        self.loss_sum = 0
        return log

    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def load(name:str, is_training:bool):
        with open(f"{name}/config.json", "r") as f:
            config = json.load(f)
        agent = DQNAgent(config['rows'], config['cols'], config['action_size'], 
                         nn_type=config['nn_type'], is_training=is_training,
                           num_input_channels=config['num_input_channels'],
                           obstacles=config['obstacles'], combat=config['combat'])
        agent.model.load_state_dict(torch.load(f"{name}/model.pt"))
        return agent
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pt")

        save_parameters = {
            "DRL_algorithm": "DQN",
            "rows": self.rows,
            "cols": self.cols,
            "state_size": int(self.state_size),
            "action_size": int(self.action_size),
            "memory_size": self.memory_size,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "target_update_freq": self.target_update_freq,
            "nn_type": self.nn_type,
            "solo": self.solo,
            "num_drl_agents": self.num_drl_agents,
            "num_preprogrammed_agents": self.num_preprogrammed_agents,
            "obstacles": self.obstacles,
            "combat": self.combat,
            "num_input_channels": self.num_input_channels,
        }

        with open(f"{path}/config.json", "w") as f:
            json.dump(save_parameters, f, indent=4)



