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
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.BoolTensor([done])

            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state).detach()))
            target_f = self.model(state)
            target_f[0][action] = target

            # Define optimizer and loss
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            loss_fn = nn.MSELoss()
            loss = loss_fn(target_f, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNAgentSprite(pygame.sprite.Sprite):
    def __init__(self, screen, rows, cols, agent):
        super().__init__()
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.agent = agent  # The DQN Agent
        self.cell_width = self.screen.get_width() / self.cols
        self.cell_height = self.screen.get_height() / self.rows
        self.image = pygame.Surface((self.cell_width, self.cell_height))
        self.image.fill((255, 0, 0))  # Agent color
        self.rect = self.image.get_rect()
        self.rect.x = self.cols // 2 * self.cell_width  # Start at middle of the screen
        self.rect.y = self.rows // 2 * self.cell_height

    def update(self, state):
        # Convert state to the appropriate format
        state = np.reshape(state, [1, self.agent.state_size])
        state = torch.FloatTensor(state)
        
        # Decide action based on current state
        action = self.agent.act(state)
        
        # Update position based on action (0: left, 1: right, 2: up, 3: down as example)
        if action == 0 and self.rect.x > 0:  # Left
            self.rect.x -= self.cell_width
        if action == 1 and self.rect.x < self.screen.get_width() - self.cell_width:  # Right
            self.rect.x += self.cell_width
        if action == 2 and self.rect.y > 0:  # Up
            self.rect.y -= self.cell_height
        if action == 3 and self.rect.y < self.screen.get_height() - self.cell_height:  # Down
            self.rect.y += self.cell_height

    def draw(self):
        self.screen.blit(self.image, self.rect)