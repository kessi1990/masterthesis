import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from collections import deque
from abc import ABC, abstractmethod

from models import models
from utils import transformation


class Agent(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def append_sample(self, *args):
        pass

    @abstractmethod
    def policy(self, *args):
        pass

    @abstractmethod
    def minimize_epsilon(self, *args):
        pass

    @abstractmethod
    def train(self, *args):
        pass


class DQN(Agent):
    def __init__(self, in_channels, input_size, nr_actions, kernel_size, stride, padding, epsilon, epsilon_decay,
                 learning_rate, discount_factor, batch_size, k_target, config):
        super(Agent).__init__()
        self.memory = deque()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.k_target = k_target
        self.k_count = 0
        self.action_space = [_ for _ in range(nr_actions)]
        self.policy_net = models.DQNModel(in_channels, input_size, nr_actions, kernel_size, stride, padding)
        self.target_net = models.DQNModel(in_channels, input_size, nr_actions, kernel_size, stride, padding)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.transformation = transformation.Transformation(config)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def policy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            net_input = self.transformation.transform(state)
            net_input.unsqueeze_(dim=0)
            q_values = self.policy_net.forward(net_input)
            q_values = q_values.detach().numpy()
            return np.argmax(q_values[0])

    def minimize_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        self.optimizer.zero_grad()
        self.policy_net.train()
        self.target_net.train()
        mini_batch = random.sample(self.memory, self.batch_size)
        mini_batch = np.array(mini_batch)
        states = torch.stack([self.transformation.transform(i) for i in mini_batch[:, 0]], dim=0)
        actions = mini_batch[:, 1]
        rewards = mini_batch[:, 2]
        next_states = torch.stack([self.transformation.transform(i) for i in mini_batch[:, 3]], dim=0)
        dones = mini_batch[:, 4]

        loss = 0
        for i in range(len(mini_batch)):
            q_old = self.policy_net.forward(states[i].unsqueeze(dim=0))[0]
            prediction = q_old[actions[i]]
            prediction = prediction.clone().detach().requires_grad_(True)
            if dones[i]:
                target = rewards[i]
            else:
                q_new = self.target_net.forward(next_states[i].unsqueeze(dim=0))[0]
                target = rewards[i] + self.discount_factor * np.argmax(q_new.detach().numpy())
            target = torch.tensor(target, requires_grad=True)
            loss += self.criterion(prediction, target)
            self.k_count += 1
        loss.backward()
        self.optimizer.step()
        self.policy_net.eval()
        self.target_net.eval()

        if self.k_count == self.k_target:
            self.update_target_net()
            self.k_count = 0
        return loss


class EADAgent(Agent):
    def __init__(self, config, nr_actions, device):
        """

        :param config:
        :param nr_actions:
        :param device:
        """
        super(EADAgent, self).__init__()
        self.device = device

        # DQN parameter
        self.action_space = [_ for _ in range(nr_actions)]
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['gamma']
        self.batch_size = config['batch_size']
        self.k_target = config['k_target']
        self.k_count = 0
        self.memory = deque(maxlen=config['mem_size'])

        self.policy_net = models.EADModel(config, nr_actions, self.device).to(self.device)
        self.target_net = models.EADModel(config, nr_actions, self.device).to(self.device)
        params = list(self.policy_net.parameters()) + list(self.target_net.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(self.device)

    def append_sample(self, state_seq, action, reward, next_state_seq, done):
        """
        stores transitions in memory buffer - used for experience replay
        :param state_seq: sequence (length=4) of consecutive states
        :param action: executed action
        :param reward: reward achieved by executing action in last state of state_sequence
        :param next_state_seq: sequence (length=4) of consecutive successor states
        :param done: flag that marks if next_state is terminal state
        :return:
        """
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def policy(self, state_sequence):
        """
        policy of the agent (epsilon-greedy)
        :param state_sequence: sequence (length=3) of consecutive states
        :return: action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = self.policy_net.forward(state_sequence)
            action = torch.argmax(q_values[0]).item()
            return action

    def minimize_epsilon(self):
        """

        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        """

        :return:
        """
        self.optimizer.zero_grad()
        self.policy_net.train()
        self.target_net.train()
        mini_batch = random.sample(self.memory, self.batch_size)
        mini_batch = np.array(mini_batch)
        state_sequences = mini_batch[:, 0]
        actions = mini_batch[:, 1]
        rewards = mini_batch[:, 2]
        next_state_sequences = mini_batch[:, 3]
        dones = mini_batch[:, 4]
        loss = 0
        for i in range(self.batch_size):
            q_old = self.policy_net.forward(state_sequences[i])
            prediction = q_old[0][actions[i]]
            # prediction = prediction.clone().detach().requires_grad_(True)
            if dones[i]:
                target = rewards[i]
            else:
                q_new = self.target_net.forward(next_state_sequences[i])
                target = rewards[i] + self.discount_factor * torch.max(q_new[0]).item()
            target = torch.tensor(target, requires_grad=True, device=self.device)
            loss += self.criterion(prediction, target)
            self.k_count += 1
        loss.backward()
        self.optimizer.step()

        if self.k_count == self.k_target:
            self.update_target()
            self.k_count = 0
        self.policy_net.eval()
        self.target_net.eval()
        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
