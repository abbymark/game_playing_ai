from game_playing_ai.games.food_game.agents.drl_agent.networks.ppo_networks import CNNActor, CNNCritic

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

import os
import json
import random
from typing import Dict
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, gamma=0.99, lambda_gae=0.95):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.gamma = gamma
        self.lambda_gae = lambda_gae
    
    def remember(self, state, action, log_prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def _get_advantages(self, rewards, dones, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns
    
    def generate_batches(self, batch_size):
        advantages = self._get_advantages(self.rewards, self.dones, self.values)
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        return (np.array(self.states), np.array(self.actions), np.array(self.log_probs),
                np.array(self.values), np.array(self.rewards), np.array(self.dones),
                np.array(advantages), batches)


class PPOAgent:
    def __init__(self, rows:int, cols:int, action_size:int, timestep_per_batch:int=1000, 
                 gamma:float=0.95, lambda_gae:float=0.95, epsilon:float=0.2, batch_size:int=32,
                 learning_rate:float=0.0001, nn_type:str="CNN", is_training:bool=True,
                 num_input_channels=6, entropy_coef:float=0.01, epochs=5) -> None:
        self.rows = rows
        self.cols = cols
        self.state_size = rows * cols * num_input_channels
        self.action_size = action_size
        self.timestep_per_batch = timestep_per_batch
        self.gamma = gamma  # discount rate
        self.lambda_gae = lambda_gae
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nn_type = nn_type
        self.is_training = is_training
        self.num_input_channels = num_input_channels
        self.entropy_coef = entropy_coef
        self.epochs = epochs

        self.actor = self.get_actor(self.nn_type)
        self.critic = self.get_critic(self.nn_type)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()


        self.loss_sum = 0

        if is_training:
            wandb.login()
            wandb.init(
                project="food_game",
                config={
                    "DRL_algorithm": "PPO",
                    "rows": self.rows,
                    "cols": self.cols,
                    "state_size": int(self.state_size),
                    "action_size": int(self.action_size),
                    "memory_size": self.memory_size,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "nn_type": self.nn_type,
                    "is_training": self.is_training,
                    "num_input_channels": self.num_input_channels,
                }
            )
            wandb.watch(self.actor)
            wandb.watch(self.critic)

    def get_actor(self, nn_type:str):
        if nn_type == "CNN":
            return CNNActor(self.num_input_channels, self.action_size).to(device)
        else:
            raise NotImplementedError

    def get_critic(self, nn_type:str):
        if nn_type == "CNN":
            return CNNCritic(self.num_input_channels).to(device)
        else:
            raise NotImplementedError

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        flat_next_states = state.view(state.shape[0], -1)
        one_hot_flat_next_states = nn.functional.one_hot(flat_next_states, num_classes=6).float()
        state = one_hot_flat_next_states.view(*state.shape, -1)
        state = state.permute(0, 3, 1, 2).contiguous()
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        value = self.critic(state)
        return action.item(), dist.log_prob(action), value
    
    def get_log(self):
        return {
            "loss_sum": self.loss_sum,
        }
    
    def save(self, path:str):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load(self, path:str):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))

    def update(self, memory: Memory):
        for i in range(self.epochs):
            states, actions, old_log_probs, values, rewards, dones, advantages, batches = memory.generate_batches(self.batch_size)
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_values = values[batch]
                batch_rewards = rewards[batch]
                batch_dones = dones[batch]
                batch_advantages = advantages[batch]

                # Convert numpy arrays to pytorch tensors
                batch_states = torch.LongTensor(batch_states).to(device)
                batch_flat_states = batch_states.view(states.shape[0], -1)
                batch_one_hot_flat_states = nn.functional.one_hot(batch_flat_states, num_classes=6).float()
                batch_states = batch_one_hot_flat_states.view(*states.shape, -1)
                batch_states = batch_states.permute(0, 3, 1, 2).contiguous()

                batch_actions = torch.LongTensor(batch_actions).view(-1, 1).to(device)
                batch_old_probs = torch.FloatTensor(batch_old_probs).to(device)
                batch_values = torch.FloatTensor(batch_values).to(device)
                batch_rewards = torch.FloatTensor(batch_rewards).to(device)
                batch_dones = torch.FloatTensor(batch_dones).to(device)

                # Update the actor
                probs = self.actor(states)
                dist = Categorical(probs)
                entropy_bonus = dist.entropy().mean()
                current_log_probs = dist.log_prob(actions)

                ratios = torch.exp(current_log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Update the critic
                value_preds = self.critic(states)
                critic_loss = self.criterion(value_preds, advantages + values)
                

                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy_bonus

                loss.backward()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.step()
                self.critic_optimizer.step()


                self.loss_sum += loss.item()

    