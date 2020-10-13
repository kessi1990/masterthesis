import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
import random

from collections import deque
from itertools import starmap
from abc import ABC, abstractmethod

from agents import memory
from models import models

from datetime import datetime


class Agent(ABC):
    """
    abstract agent class
    """
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
    def update_target(self, *args):
        pass

    @abstractmethod
    def train(self, *args):
        pass


class DQN(Agent):
    """
    implementation of basic DQN approach -> Q-learning with known update rule, policy and target neural networks and
    experience replay. policy and target networks are either cead- or darqn-models.
    """
    def __init__(self, model_type, nr_actions, device, num_layers):
        """

        :param model_type: cead or darqn model
        :param nr_actions: number of possible actions in environment, defines number of output neurons
        :param device: evice which is in charge of computations (CPU / GPU)
        :param num_layers: number of LSTM layers
        """
        super().__init__()
        self.nr_actions = nr_actions
        self.action_space = [_ for _ in range(self.nr_actions)]
        print(f'nr_actions: {self.nr_actions}, action_space: {self.action_space}')
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 9e-07
        self.epsilon_min = 0.1
        self.discount_factor = 0.99
        self.batch_size = 32
        self.memory = memory.ReplayMemory(maxlen=500000)
        self.k_count = 0
        self.k_target = 10000
        self.reward_clipping = False
        self.gradient_clipping = False

        self.device = device
        self.policy_net = models.CEADModel(nr_actions, device, num_layers).to(device) if model_type == 'cead' \
            else models.DARQNModel(nr_actions, device, num_layers).to(device)
        self.target_net = models.CEADModel(nr_actions, device, num_layers).to(device) if model_type == 'cead' \
            else models.DARQNModel(nr_actions, device, num_layers).to(device)

        # copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set target_net in evaluation mode
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)  # RMSProp instead of Adam
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer)

    def append_sample(self, state_seq, action, reward, next_state_seq, done):
        """
        saves experience tuple to internal memory buffer of the agent
        :param state_seq: sequence of 4 consecutive states
        :param action: executed action in last state of state_seq
        :param reward: obtained reward by executing action
        :param next_state_seq: sequence of 4 consecutive next_states
        :param done: terminal flag
        :return:
        """
        # TODO: check if legit: len >= 4
        if len(state_seq) >= 4:
            self.memory.append((state_seq, action, reward, next_state_seq, done))

    def policy(self, state_sequence, mode):
        """
        policy pi defines behaviour of agent. with probability epsilon a random action is selected. with complementary
        probability 1 - epsilon, a prediction of Q-values based on state_sequence is made and the corresponding action
        to the highest predicted Q-value is chosen for execution. epsilon value decays during training mode, whilst
        being stationary during evaluation mode.
        :param state_sequence: sequence of 4 consecutive states
        :param mode: training or evaluation mode
        :return: action
        """
        if np.random.rand() <= (self.epsilon if mode == 'training' else 0.05):
            return np.random.choice(self.action_space)
        else:
            q_values = self.policy_net.forward(state_sequence)
            action = torch.argmax(q_values[0]).item()
            return action

    def minimize_epsilon(self):
        """
        minimizes epsilon value which is used for greedy action selection
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def update_target(self):
        """
        updates target network by copying weights from policy network (hard update)
        :return:
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        """
        trains policy network by sampling a mini batch of already experienced transitions from memory buffer and
        constructing a loss which is propagated backwards through the network
        :return: accumulated loss
        """
        if len(self.memory) < self.batch_size:
            return torch.zeros(1).item()

        # set policy_net to train mode
        self.policy_net.train()

        # sample and slice mini_batch
        mini_batch = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch)
        state_sequences = mini_batch[:, 0]
        actions = mini_batch[:, 1]
        rewards = np.clip(mini_batch[:, 2], a_min=-1, a_max=1) if self.reward_clipping else mini_batch[:, 2]
        next_state_sequences = mini_batch[:, 3]
        dones = mini_batch[:, 4]
        loss = 0

        # compute targets and loss
        for i in range(self.batch_size):
            self.policy_net.init_hidden()
            self.target_net.init_hidden()
            q_old = self.policy_net.forward_old(state_sequences[i])
            prediction = q_old[0][actions[i]]
            if dones[i]:
                target = rewards[i]
            else:
                q_new = self.target_net.forward_old(next_state_sequences[i]).detach()
                target = rewards[i] + self.discount_factor * torch.max(q_new[0]).item()
            target = torch.tensor(target, requires_grad=False, device=self.device)
            loss += functional.smooth_l1_loss(prediction, target)
        self.k_count += 1

        # zero gradients
        self.optimizer.zero_grad()

        start = datetime.utcnow()
        loss.backward()
        end = datetime.utcnow()
        print(f'loss backward {end - start}')
        if self.gradient_clipping:
            for param in self.policy_net.parameters():  # only clamp lstm gradients (ltpwtl paper)
                param.grad.data.clamp_(min=-1, max=1)
        start = datetime.utcnow()
        self.optimizer.step()
        end = datetime.utcnow()
        print(f'optimizer step {end - start}')

        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0

        # set policy_net to evaluation mode
        self.policy_net.eval()

        # set init hidden
        self.policy_net.init_hidden()
        self.target_net.init_hidden()
        print(f'k_count: {self.k_count}')
        return loss.item()

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return torch.zeros(1).item()

        self.policy_net.train()

        self.policy_net.init_hidden(batch_size=self.batch_size)
        self.target_net.init_hidden(batch_size=self.batch_size)

        mini_batch = np.array(self.memory.sample(self.batch_size))
        state_sequences = mini_batch[:, 0]
        actions = mini_batch[:, 1]
        reward = torch.tensor(list(mini_batch[:, 2])).unsqueeze(dim=1)
        next_state_sequences = mini_batch[:, 3]
        dones = mini_batch[:, 4]

        state_batch = [torch.stack([seq[i] for seq in state_sequences], dim=0) for i in range(4)]
        next_state_batch = [torch.stack([seq[i] for seq in next_state_sequences], dim=0) for i in range(4)]

        state_q = self.policy_net.forward(state_batch)

        with torch.no_grad():
            next_state_q = self.target_net.forward(next_state_batch)

        target = torch.max(next_state_q, dim=-1, keepdim=True)[0]
        target = (target * self.discount_factor) + reward
        target = target.clone().detach().requires_grad_(False)
        # TODO: terminal / non temrinal states

        prediction = torch.stack([state_q[i][a] for i, a in enumerate(actions)]).unsqueeze(dim=1)

        start = datetime.utcnow()
        loss = functional.smooth_l1_loss(prediction, target)
        end = datetime.utcnow()
        print(f'loss {end - start}')

        self.optimizer.zero_grad()

        start = datetime.utcnow()
        loss.backward()
        end = datetime.utcnow()
        print(f'loss backward {end - start}')
        if self.gradient_clipping:
            for param in self.policy_net.parameters():  # only clamp lstm gradients (ltpwtl paper)
                param.grad.data.clamp_(min=-1, max=1)

        start = datetime.utcnow()
        self.optimizer.step()
        end = datetime.utcnow()
        print(f'optimizer step {end - start}')
        self.k_count += 1
        self.policy_net.eval()
        print(f'k_count: {self.k_count}')
        return loss.item()


class DQNRaw(Agent):
    """
    implementation of basic DQN approach -> Q-learning with known update rule, policy and target neural networks and
    experience replay. policy and target networks are either cead- or darqn-models.
    """
    def __init__(self, model_type, nr_actions, device, num_layers):
        """

        :param model_type: cead or darqn model
        :param nr_actions: number of possible actions in environment, defines number of output neurons
        :param device: evice which is in charge of computations (CPU / GPU)
        :param num_layers: number of LSTM layers
        """
        super().__init__()
        self.nr_actions = nr_actions
        self.action_space = [_ for _ in range(self.nr_actions)]
        print(f'nr_actions: {self.nr_actions}, action_space: {self.action_space}')
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 9e-07
        self.epsilon_min = 0.1
        self.discount_factor = 0.99
        self.batch_size = 32
        self.memory = deque(maxlen=5000000)  # memory.ReplayMemory(maxlen=500000)
        self.k_count = 0
        self.k_target = 10000

        self.device = device
        self.policy_net = models.DQNModel(nr_actions, device).to(device)
        self.target_net = models.DQNModel(nr_actions, device).to(device)

        # copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set target_net in evaluation mode
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)  # RMSProp instead of Adam
        # self.criterion = functional.smooth_l1_loss()

    def append_sample(self, state_seq, action, reward, next_state_seq, done):
        """
        saves experience tuple to internal memory buffer of the agent
        :param state_seq: sequence of 4 consecutive states
        :param action: executed action in last state of state_seq
        :param reward: obtained reward by executing action
        :param next_state_seq: sequence of 4 consecutive next_states
        :param done: terminal flag
        :return:
        """
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def policy(self, state, mode):
        """
        policy pi defines behaviour of agent. with probability epsilon a random action is selected. with complementary
        probability 1 - epsilon, a prediction of Q-values based on state_sequence is made and the corresponding action
        to the highest predicted Q-value is chosen for execution. epsilon value decays during training mode, whilst
        being stationary during evaluation mode.
        :param state_sequence: sequence of 4 consecutive states
        :param mode: training or evaluation mode
        :return: action
        """
        if np.random.rand() <= (self.epsilon if mode == 'training' else 0.05):
            return np.random.choice(self.action_space)
        else:
            q_values = self.policy_net.forward(state.unsqueeze(dim=0))
            action = torch.argmax(q_values[0]).item()
            return action

    def minimize_epsilon(self):
        """
        minimizes epsilon value which is used for greedy action selection
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def update_target(self):
        """
        updates target network by copying weights from policy network (hard update)
        :return:
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        """
        trains policy network by sampling a mini batch of already experienced transitions from memory buffer and
        constructing a loss which is propagated backwards through the network
        :return: accumulated loss
        """
        if len(self.memory) < self.batch_size:
            return torch.zeros(1)

        # set policy_net to train mode
        self.policy_net.train()

        # sample and slice mini_batch
        mini_batch = random.sample(self.memory, self.batch_size) # self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        # mini_batch = [data for data in mini_batch]
        states = [data[0] for data in mini_batch] # mini_batch[:, 0]
        actions = [data[1] for data in mini_batch] # mini_batch[:, 1]
        rewards = [data[2] for data in mini_batch] # mini_batch[:, 2]
        next_states = [data[3] for data in mini_batch] # mini_batch[:, 3]
        dones = [data[4] for data in mini_batch] # mini_batch[:, 4]
        loss = 0

        # compute targets and loss
        for i in range(self.batch_size):
            q_old = self.policy_net.forward(states[i].unsqueeze(dim=0))
            prediction = q_old[0][actions[i]]
            if dones[i]:
                target = rewards[i]
            else:
                q_new = self.target_net.forward(next_states[i].unsqueeze(dim=0)).detach()
                target = rewards[i] + self.discount_factor * torch.max(q_new[0]).item()
            target = torch.tensor(target, requires_grad=False, device=self.device)
            loss += functional.smooth_l1_loss(prediction, target)
        self.k_count += 1

        # zero gradients
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(min=-1, max=1)
        self.optimizer.step()

        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0

        # set policy_net to evaluation mode
        self.policy_net.eval()
        return loss
