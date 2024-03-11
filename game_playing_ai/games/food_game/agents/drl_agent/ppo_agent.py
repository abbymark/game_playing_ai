# PPO codes were referenced from https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/main.py and modified to fit the game

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
        self.states.extend(state)
        self.actions.extend(action)
        self.log_probs.extend(log_prob)
        self.values.extend(val)
        self.rewards.extend(reward)
        self.dones.extend(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def _get_advantages(self, rewards, dones, values):
        # gae = 0
        # advantages = []
        # for step in reversed(range(len(rewards)-1)):
        #     delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
        #     gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
        #     advantages.insert(0, gae + values[step])
        # advantages.append(0)
        # return advantages

        advantages = np.zeros_like(rewards)
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + self.gamma * values[k+1] * (1-dones[k]) - values[k])
                discount *= self.gamma * self.lambda_gae

            advantages[t] = a_t
        return advantages

    
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
    def __init__(self, rows:int, cols:int, action_size:int, gamma:float=0.95, 
                 lambda_gae:float=0.95, epsilon:float=0.2, batch_size:int=32,
                 learning_rate:float=0.0001, nn_type:str="CNN", solo:bool=True,
                 num_drl_agents:int=1, num_preprogrammed_agents:int=0,
                 obstacles:bool=False, combat:bool=False, is_training:bool=True,
                 num_input_channels=6, entropy_coef:float=0.01, epochs=5) -> None:
        self.rows = rows
        self.cols = cols
        self.state_size = rows * cols * num_input_channels
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.lambda_gae = lambda_gae
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nn_type = nn_type
        self.solo = solo
        self.num_drl_agents = num_drl_agents
        self.num_preprogrammed_agents = num_preprogrammed_agents
        self.obstacles = obstacles
        self.combat = combat
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


        self.total_loss_sum = 0
        self.actor_loss_sum = 0
        self.critic_loss_sum = 0
        self.entropy_bonus_sum = 0
        self.steps = 0

        if is_training:
            wandb.login()
            wandb.init(
                project="food_game",
                name=f"PPO_r:{self.rows}_c:{self.cols}_n_drl:{num_drl_agents}_n_pre{num_preprogrammed_agents}_obs:{obstacles}_combat:{combat}_solo:{solo}",
                config={
                    "DRL_algorithm": "PPO",
                    "rows": self.rows,
                    "cols": self.cols,
                    "state_size": int(self.state_size),
                    "action_size": int(self.action_size),
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "nn_type": self.nn_type,
                    "solo": self.solo,
                    "num_drl_agents": self.num_drl_agents,
                    "num_preprogrammed_agents": self.num_preprogrammed_agents,
                    "obstacles": self.obstacles,
                    "combat": self.combat,
                    "num_input_channels": self.num_input_channels,
                    "entropy_coef": self.entropy_coef,
                    "epochs": self.epochs,
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
        state = torch.LongTensor(state).unsqueeze(0).to(device)
        flat_next_states = state.view(state.shape[0], -1)
        one_hot_flat_next_states = nn.functional.one_hot(flat_next_states, num_classes=self.num_input_channels).float()
        state = one_hot_flat_next_states.view(*state.shape, -1)
        state = state.permute(0, 3, 1, 2).contiguous()
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        value = self.critic(state)
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def get_log(self):
        return {
            "total_loss": self.total_loss_sum,
            "actor_loss": self.actor_loss_sum,
            "critic_loss": self.critic_loss_sum,
        }

    @staticmethod
    def load(name:str, is_training:bool):
        with open(f"{name}/config.json", "r") as f:
            config = json.load(f)
        agent = PPOAgent(config['rows'], config['cols'], config['action_size'], 
                         nn_type=config['nn_type'], is_training=is_training, 
                         num_input_channels=config['num_input_channels'],
                         obstacles=config['obstacles'], combat=config['combat'],)
        agent.actor.load_state_dict(torch.load(f"{name}/actor.pt"))
        agent.critic.load_state_dict(torch.load(f"{name}/critic.pt"))
        return agent
    
    def save(self, path:str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")

        save_params = {
            "DRL_algorithm": "PPO",
            "rows": self.rows,
            "cols": self.cols,
            "state_size": int(self.state_size),
            "action_size": int(self.action_size),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "nn_type": self.nn_type,
            "solo": self.solo,
            "num_drl_agents": self.num_drl_agents,
            "num_preprogrammed_agents": self.num_preprogrammed_agents,
            "obstacles": self.obstacles,
            "combat": self.combat,
            "num_input_channels": self.num_input_channels,
            "entropy_coef": self.entropy_coef,
            "epochs": self.epochs,
        }

        with open(f"{path}/config.json", "w") as f:
            json.dump(save_params, f, indent=4)



    def update(self, memory: Memory):
        self.steps += 1
        self.total_loss_sum = 0
        self.actor_loss_sum = 0
        self.critic_loss_sum = 0
        self.entropy_bonus_sum = 0
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
                batch_flat_states = batch_states.view(batch_states.shape[0], -1)
                batch_one_hot_flat_states = nn.functional.one_hot(batch_flat_states, num_classes=self.num_input_channels).float()
                batch_states = batch_one_hot_flat_states.view(*batch_states.shape, -1)
                batch_states = batch_states.permute(0, 3, 1, 2).contiguous()

                batch_actions = torch.LongTensor(batch_actions).view(-1, 1).to(device)
                batch_old_log_probs = torch.FloatTensor(batch_old_log_probs).to(device)
                batch_values = torch.FloatTensor(batch_values).to(device)
                batch_rewards = torch.FloatTensor(batch_rewards).to(device)
                batch_dones = torch.FloatTensor(batch_dones).to(device)
                batch_advantages = torch.FloatTensor(batch_advantages).to(device)

                # Update the actor
                probs = self.actor(batch_states)
                dist = Categorical(probs)
                entropy_bonus = dist.entropy().mean()

                current_log_probs = dist.log_prob(batch_actions)

                ratios = torch.exp(current_log_probs - batch_old_log_probs)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Update the critic
                value_preds = self.critic(batch_states).squeeze()
                critic_loss = self.criterion(value_preds, batch_advantages + batch_values).mean()
                

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_bonus
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()


                self.total_loss_sum += loss.item()
                self.actor_loss_sum += actor_loss.item()
                self.critic_loss_sum += critic_loss.item()
                self.entropy_bonus_sum += entropy_bonus.item()

        wandb.log({
            "total_loss": self.total_loss_sum,
            "actor_loss": self.actor_loss_sum,
            "critic_loss": self.critic_loss_sum,
            "entropy_bonus": self.entropy_bonus_sum,
        })