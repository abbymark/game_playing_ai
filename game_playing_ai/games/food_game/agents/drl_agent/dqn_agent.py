from game_playing_ai.games.food_game.agents.drl_agent.networks.dnn import DNN

import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import wandb

class DQNAgent:
    def __init__(self, state_size:int, action_size:int, memory_size:int=10000, 
                 gamma:float=0.95, epsilon_min:float=0.01, epsilon_decay:float=0.9999, 
                 learning_rate:float=0.0001, target_update_freq:str=100, nn_type:str="DNN", is_training:bool=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.model = self.get_model(nn_type)
        self.target_model = self.get_model(nn_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.is_training = is_training
        self.update_step = 0

        if is_training:
            wandb.login()
            wandb.init(project="food_game")

    def get_model(self, nn_type:str):
        if nn_type == 'DNN':
            return DNN(self.state_size, self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and self.is_training:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        self.update_step += 1

        if len(self.memory) < batch_size:
            return
    
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([x[0] for x in minibatch]).reshape(-1, self.state_size)
        actions = torch.LongTensor([x[1] for x in minibatch]).view(-1, 1)
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.FloatTensor([x[3] for x in minibatch]).reshape(-1, self.state_size)
        dones = torch.FloatTensor([float(x[4]) for x in minibatch])

        # Predict Q-values for starting states
        Q_values = self.model(states)

        # Predict next Q-values for next states
        next_Q_values = self.target_model(next_states)
        max_next_Q_values = next_Q_values.max(1)[0]

        # Compute the target Q values
        targets = rewards + (self.gamma * max_next_Q_values * (1 - dones))

        action_Q_values = Q_values.gather(1, actions).squeeze()

        # Loss calculation
        loss = self.criterion(action_Q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        wandb.log({"loss": loss.item()})
        wandb.log({"epsilon": self.epsilon})
        

        # update target network
        if self.update_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

            print(f"Target network updated and the loss is {loss.item()}")
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)


class DQNAgentSprite():
    def __init__(self, rows, cols, pos = None):
        self.rows = rows
        self.cols = cols
        self.food_collected = 0
        if pos is None:
            self.x = random.randint(0, cols - 1)
            self.y = random.randint(0, rows - 1)
        else:
            self.x = pos[0]
            self.y = pos[1]

    def update(self, action):
        if action is None:
            return
        new_x = self.x
        new_y = self.y

        if action == 0:
            new_x -= 1
        elif action == 1:
            new_x += 1
        elif action == 2:
            new_y -= 1
        elif action == 3:
            new_y += 1

        # Check if new position is within bounds
        if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
            # Update position if within bounds
            self.x = new_x
            self.y = new_y
        

    
    def set_pos_in_map(self, map):
        while map[self.y][self.x] != 0 and map[self.y][self.x] != 3:
            self.x = random.randint(0, self.cols - 1)
            self.y = random.randint(0, self.rows - 1)
        map[self.y][self.x] = 5
        return map
    
    @property
    def pos(self):
        return (self.x, self.y)
    
    @pos.setter
    def pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]