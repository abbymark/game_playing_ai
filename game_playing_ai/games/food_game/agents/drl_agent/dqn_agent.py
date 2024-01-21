import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
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
        next_Q_values = self.model(next_states).detach()
        max_next_Q_values = next_Q_values.max(1)[0]

        # Compute the target Q values
        targets = rewards + (self.gamma * max_next_Q_values * (1 - dones))

        # Only update the Q value for the action taken
        Q_targets = Q_values.clone()
        Q_targets[range(batch_size), actions.squeeze()] = targets

        # Loss calculation
        loss = self.criterion(Q_values, Q_targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        print(loss)
        self.optimizer.step()

        # Update epsilon
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