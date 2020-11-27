import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
import random

from collections import deque
from itertools import count
from abc import ABC, abstractmethod

from agents import memory
from models import models
from utils import shapes

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
    def __init__(self, model_type, nr_actions, device, num_layers, hidden_size, alignment):
        """

        :param model_type:
        :param nr_actions:
        :param device:
        :param num_layers:
        :param hidden_size:
        :param alignment:
        """
        super().__init__()
        self.nr_actions = nr_actions
        self.action_space = [_ for _ in range(self.nr_actions)]
        print(f'nr_actions: {self.nr_actions}, action_space: {self.action_space}')
        self.learning_rate = 0.001
        self.learning_rate_decay = 3.0e-09
        self.learning_rate_min = 0.00025
        self.epsilon = 1
        self.epsilon_decay = 9e-07
        self.epsilon_min = 0.1
        self.discount_factor = 0.99
        self.batch_size = 32
        self.memory = memory.DARQNReplayMemory(maxlen=200000)
        self.k_count = 0
        self.k_target = 10000
        self.reward_clipping = True
        self.gradient_clipping = True
        self.clip_value = 10

        self.device = device
        self.policy_net = models.CEADModel(nr_actions, device, num_layers, hidden_size, alignment).to(device) if model_type == 'cead' \
            else models.DARQNModel(nr_actions, device, num_layers, hidden_size, alignment).to(device)
        self.target_net = models.CEADModel(nr_actions, device, num_layers, hidden_size, alignment).to(device) if model_type == 'cead' \
            else models.DARQNModel(nr_actions, device, num_layers, hidden_size, alignment).to(device)

        # copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set target_net in evaluation mode
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate, momentum=0.95)
        self.criterion = nn.MSELoss()

        shapes.count_parameters(self.policy_net)

    def append_sample(self, state, action, reward, next_state, done):
        """
        saves experience tuple to internal memory buffer of the agent
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def policy(self, state, mode):
        """
        policy pi defines behaviour of agent. with probability epsilon a random action is selected. with complementary
        probability 1 - epsilon, a prediction of Q-values based on state is made and the corresponding action
        to the highest predicted Q-value is chosen for execution. epsilon value decays during training mode, whilst
        being stationary during evaluation mode.
        :param state: state x of n consecutive states
        :param mode: training or evaluation mode
        :return: action
        """
        if np.random.rand() <= (self.epsilon if mode == 'training' else 0.05):
            return np.random.choice(self.action_space)
        else:
            # q_values, context, weights = self.policy_net.forward(state)
            q_values = self.policy_net.forward(state)
            action = torch.argmax(q_values[0]).item()
            return action  # , context, weights

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
        :return: loss
        """
        if len(self.memory) < self.batch_size:
            return torch.zeros(1).item()

        # set policy_net to train mode
        self.policy_net.train()

        # init hidden & cell states of both networks with batch_size
        self.policy_net.init_hidden(batch_size=self.batch_size)
        self.target_net.init_hidden(batch_size=self.batch_size)

        # sample mini_batch from memory buffer
        mini_batch = self.memory.sample(self.batch_size)

        # unzip / inverse zip
        state_seq, actions, rewards, next_state_seq, dones = list(zip(*mini_batch))

        # construct network inputs
        state_seq_batch = torch.stack(state_seq, dim=0).transpose(dim0=0, dim1=1)  # state_batches
        next_state_seq_batch = torch.stack(next_state_seq, dim=0).transpose(dim0=0, dim1=1)  # next_state_batch

        # construct tensors target computation
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.int64)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.int8)
        final_mask = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # clip rewards if True
        if self.reward_clipping:
            reward_batch.clamp_(min=-1, max=1)

        # predict on state_batch and gather q_values for action_batch
        for state_batch in state_seq_batch:
            prediction = self.policy_net.forward(state_batch)
        prediction = prediction.gather(1, action_batch.unsqueeze(dim=1))

        # compute target according to q-learning update rule
        for next_state_batch in next_state_seq_batch:
            target = self.target_net.forward(next_state_batch)

        # get max q-value and detach from computational graph
        target = target.max(dim=-1)[0].detach()
        target[final_mask] = 0
        target = (target * self.discount_factor) + reward_batch

        # compute loss
        loss = self.criterion(prediction, target.unsqueeze(dim=1))

        # zero gradients
        self.optimizer.zero_grad()

        # backpropagate loss
        loss.backward()

        # clip gradients if True
        if self.gradient_clipping:
            parameters = [param for name, param in self.policy_net.named_parameters() if 'lstm' in name]
            torch.nn.utils.clip_grad_value_(parameters, self.clip_value)

        # perform optimizer step
        self.optimizer.step()

        # decay learning rate
        if self.learning_rate > self.learning_rate_min:
            self.optimizer.param_groups[0]['lr'] -= self.learning_rate_decay

        # increment counter for target_net update
        self.k_count += 1

        # update target network if True
        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0

        # set policy_net to eval mode
        self.policy_net.eval()

        # init hidden & cell states of both networks with default (batch=1)
        self.policy_net.init_hidden()
        self.target_net.init_hidden()

        return loss.item()


class DQNRaw(Agent):
    """
    implementation of basic DQN approach -> Q-learning with known update rule, policy and target neural networks and
    experience replay. policy and target networks are either cead- or darqn-models.
    """
    def __init__(self, in_channels, nr_actions, device):
        """

        :param nr_actions: number of possible actions in environment, defines number of output neurons
        :param device: evice which is in charge of computations (CPU / GPU)
        """
        super().__init__()
        self.nr_actions = nr_actions
        self.action_space = [_ for _ in range(self.nr_actions)]
        print(f'nr_actions: {self.nr_actions}, action_space: {self.action_space}')
        self.learning_rate = 0.001
        self.learning_rate_decay = 1.95e-08
        self.learning_rate_min = 0.00025
        self.lr_lambda = lambda epoch: self.learning_rate - (epoch * self.learning_rate_decay)  # linear decay per epoch
        self.epsilon = 1
        self.epsilon_decay = 9e-07
        self.epsilon_min = 0.1
        self.discount_factor = 0.99
        self.batch_size = 32
        print(f'in_channels {in_channels}')
        self.memory = memory.DQNReplayMemory(maxlen=200000) if in_channels == 1 else memory.DQNFSReplayMemory(maxlen=200000)
        self.k_count = 0
        self.k_target = 10000
        self.reward_clipping = True
        self.gradient_clipping = False
        self.gradient_threshold = 10

        self.device = device
        self.policy_net = models.DQNModel(in_channels, nr_actions, device).to(device)
        self.target_net = models.DQNModel(in_channels, nr_actions, device).to(device)

        # copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set target_net in evaluation mode
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate, momentum=0.95)  # optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)  #
        self.lr_scheduler = None  # optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

        self.criterion = nn.MSELoss()

        shapes.count_parameters(self.policy_net)


    def append_sample(self, state, action, reward, next_state, done):
        """
        saves experience tuple to internal memory buffer of the agent
        :param state_seq: sequence of 4 consecutive states
        :param action: executed action in last state of state_seq
        :param reward: obtained reward by executing action
        :param next_state_seq: sequence of 4 consecutive next_states
        :param done: terminal flag
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

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
            q_values = self.policy_net.forward(state)
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
        :return: loss
        """
        if len(self.memory) < self.batch_size:
            return torch.zeros(1).item()

        # set policy_net to train mode
        self.policy_net.train()

        # sample mini_batch from memory buffer
        mini_batch = self.memory.sample(self.batch_size)

        # unzip / inverse zip
        states, actions, rewards, next_states, dones = list(zip(*mini_batch))

        # construct network inputs
        state_batch = torch.cat(states)
        next_state_batch = torch.cat(next_states)

        # construct tensors target computation
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.int64)
        reward_batch = torch.tensor(rewards, device=self.device)
        final_mask = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # clip rewards if True
        if self.reward_clipping:
            reward_batch.clamp_(min=-1, max=1)

        # zero gradients
        # self.optimizer.zero_grad()

        # predict on state_batch and gather q_values for action_batch
        prediction = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(dim=1))

        # compute target according to q-learning update rule
        target = self.target_net.forward(next_state_batch).max(dim=1)[0].detach()
        target[final_mask] = 0
        target = (target * self.discount_factor) + reward_batch

        # compute loss
        loss = self.criterion(prediction, target.unsqueeze(dim=1))

        # zero gradients
        self.optimizer.zero_grad()

        # backpropagate loss
        loss.backward()

        # clip gradients if True
        if self.gradient_clipping:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(min=self.gradient_threshold, max=self.gradient_threshold)

        # perform optimizer step
        self.optimizer.step()

        # decay learning rate if scheduler exists
        if self.lr_scheduler:
            if self.optimizer.param_groups[0]["lr"] > self.learning_rate_min:
                self.lr_scheduler.step()

        # increment counter for target_net update
        self.k_count += 1

        # update target network if True
        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0

        # set policy_net to eval mode
        self.policy_net.eval()

        return loss.item()


class DQNNew(Agent):
    """
    implementation of basic DQN approach -> Q-learning with known update rule, policy and target neural networks and
    experience replay. policy and target networks are either cead- or darqn-models.
    """
    def __init__(self, in_channels, out_channels, nr_actions, device):
        """

        :param nr_actions: number of possible actions in environment, defines number of output neurons
        :param device: evice which is in charge of computations (CPU / GPU)
        """
        super().__init__()
        self.nr_actions = nr_actions
        self.action_space = [_ for _ in range(self.nr_actions)]
        print(f'nr_actions: {self.nr_actions}, action_space: {self.action_space}')
        self.learning_rate = 0.00025
        self.learning_rate_decay = 1.95e-08
        self.learning_rate_min = 0.00025
        self.lr_lambda = lambda epoch: self.learning_rate - (epoch * self.learning_rate_decay)  # linear decay per epoch
        self.epsilon = 1
        self.epsilon_decay = 9e-07
        self.epsilon_min = 0.1
        self.discount_factor = 0.99
        self.batch_size = 32
        print(f'in_channels {in_channels}')
        self.memory = memory.DQNReplayMemory(maxlen=200000) if in_channels == 1 else memory.DQNFSReplayMemory(maxlen=200000)
        self.k_count = 0
        self.k_target = 10000
        self.reward_clipping = True
        self.gradient_clipping = False
        self.gradient_threshold = 10

        self.device = device
        self.policy_net = models.NoLSTM(nr_actions, in_channels, out_channels=out_channels, alignment='concat', hidden_size=out_channels, device=device).to(device)  # nr_actions, in_channels, out_channels, alignment, hidden_size, device
        self.target_net = models.NoLSTM(nr_actions, in_channels, out_channels=out_channels, alignment='concat', hidden_size=out_channels, device=device).to(device)

        # copy initial weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # set target_net in evaluation mode
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)  # optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)  #
        self.lr_scheduler = None  # optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

        self.criterion = nn.MSELoss()

        shapes.count_parameters(self.policy_net)


    def append_sample(self, state, action, reward, next_state, done):
        """
        saves experience tuple to internal memory buffer of the agent
        :param state_seq: sequence of 4 consecutive states
        :param action: executed action in last state of state_seq
        :param reward: obtained reward by executing action
        :param next_state_seq: sequence of 4 consecutive next_states
        :param done: terminal flag
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

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
        if np.random.rand() <= (self.epsilon if mode == 'training' else 0.00):
            return np.random.choice(self.action_space)
        else:
            q_values, context, weights = self.policy_net.forward(state)
            action = torch.argmax(q_values[0]).item()
            return action, context, weights

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
        :return: loss
        """
        if len(self.memory) < self.batch_size:
            return torch.zeros(1).item()

        self.policy_net.init_hidden(batch_size=self.batch_size)
        self.target_net.init_hidden(batch_size=self.batch_size)

        # set policy_net to train mode
        self.policy_net.train()

        # sample mini_batch from memory buffer
        mini_batch = self.memory.sample(self.batch_size)

        # unzip / inverse zip
        states, actions, rewards, next_states, dones = list(zip(*mini_batch))

        # construct network inputs
        state_batch = torch.cat(states)
        next_state_batch = torch.cat(next_states)

        # construct tensors target computation
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.int64)
        reward_batch = torch.tensor(rewards, device=self.device)
        final_mask = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # clip rewards if True
        if self.reward_clipping:
            reward_batch.clamp_(min=-1, max=1)

        # zero gradients
        # self.optimizer.zero_grad()

        # predict on state_batch and gather q_values for action_batch
        prediction = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(dim=1))

        # compute target according to q-learning update rule
        target = self.target_net.forward(next_state_batch).max(dim=1)[0].detach()
        target[final_mask] = 0
        target = (target * self.discount_factor) + reward_batch

        # compute loss
        loss = self.criterion(prediction, target.unsqueeze(dim=1))  # functional.smooth_l1_loss(prediction, target.unsqueeze(dim=1))

        # zero gradients
        self.optimizer.zero_grad()

        # backpropagate loss
        loss.backward()

        # clip gradients if True
        if self.gradient_clipping:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(min=self.gradient_threshold, max=self.gradient_threshold)

        # perform optimizer step
        self.optimizer.step()

        # decay learning rate if scheduler exists
        if self.lr_scheduler:
            if self.optimizer.param_groups[0]["lr"] > self.learning_rate_min:
                self.lr_scheduler.step()

        # increment counter for target_net update
        self.k_count += 1

        # update target network if True
        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0

        # set policy_net to eval mode
        self.policy_net.eval()

        self.policy_net.init_hidden()
        self.target_net.init_hidden()

        return loss.item()
