import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from collections import deque
from abc import ABC, abstractmethod

from models import models
from utils import transformation
from utils import viz
from utils import captain

from datetime import datetime


class Agent(ABC):
    """
    abstract agent class from which other agent classes inherit
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
    def train(self, *args):
        pass


class DQN(Agent):
    """
    basic DQN agent implementation according to Mnih et al. "Playing Atari with Deep Reinforcement Learning"
    """
    def __init__(self, in_channels, input_size, nr_actions, kernel_size, stride, padding, epsilon, epsilon_min,
                 epsilon_decay, learning_rate, discount_factor, batch_size, k_target, config):
        """

        :param in_channels: number of channels from input image
        :param input_size: size of input image
        :param nr_actions: number of possible actions in some atari environment
        :param kernel_size: size of filters / kernels
        :param stride: kernel stepping, moves n=stride steps further
        :param padding: padding zeros
        :param epsilon: numerical value for  exploitation / exploration trade-off
        :param epsilon_min: minimal value of epsilon
        :param epsilon_decay: decay factor for minimizing epsilon value
        :param learning_rate: initial learning rate, defines how strong weight updates are taken into account
        :param discount_factor: discounting factor, diminishing future rewards
        :param batch_size: size of batch
        :param k_target: counting parameter, defines the time of weight update for target network
        :param config: config file containing necessary parameters
        """
        super(Agent).__init__()
        self.memory = deque()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
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

    def policy(self, state):
        """
        policy pi defining behaviour of agent. with probability epsilon a random action is selected. with complementary
        probability 1 - epsilon, a prediction of Q-values based on current state is made and the corresponding action to
        the highest predicted Q-value is chosen for execution
        :param state: current state
        :return: action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            net_input = self.transformation.transform(state)
            net_input.unsqueeze_(dim=0)
            q_values = self.policy_net.forward(net_input)
            q_values = q_values.detach().numpy()
            return np.argmax(q_values[0])

    def minimize_epsilon(self):
        """
        minimizes epsilon value which is used for greedy action selection
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
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
                q_new = self.target_net.forward(next_states[i])
                target = rewards[i] + self.discount_factor * torch.max(q_new[0]).item()
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

        :param config: config file containing parameters
        :param nr_actions: number of possible actions in some atari environment
        :param device: device which is in charge of computations (CPU / GPU)
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

        # Networks
        self.policy_net = models.EADModel(config, nr_actions, self.device).to(self.device)
        self.target_net = models.EADModel(config, nr_actions, self.device).to(self.device)

        # Network parameter
        params = list(self.policy_net.parameters()) + list(self.target_net.parameters())

        # Optimizer and Loss
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(self.device)

        # Visualization
        self.visor = viz.Visualizer(config)
        self.viz_data = None

        self.captain = captain.Captain()
        self.register_hooks()

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
        policy pi defining behaviour of agent. with probability epsilon a random action is selected. with complementary
        probability 1 - epsilon, a prediction of Q-values based on current state sequence is made and the corresponding
        action to the highest predicted Q-value is chosen for execution
        :param state_sequence: sequence of states
        :return: action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values, _ = self.policy_net.forward(state_sequence)
            action = torch.argmax(q_values[0]).item()
            return action

    def minimize_epsilon(self):
        """
        minimizes epsilon value which is used for greedy action selection
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        """
        trains policy network by sampling a mini batch of already experienced transitions from memory buffer and
        constructing a loss which is propagated backwards through the network
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
        visualization_data = None
        for i in range(self.batch_size):
            q_old, visualization_data = self.policy_net.forward(state_sequences[i])
            prediction = q_old[0][actions[i]]
            if dones[i]:
                target = rewards[i]
            else:
                q_new, _ = self.target_net.forward(next_state_sequences[i])
                target = rewards[i] + self.discount_factor * torch.max(q_new[0]).item()
            target = torch.tensor(target, requires_grad=True, device=self.device)
            loss += self.criterion(prediction, target)
        else:
            self.viz_data = visualization_data
        self.k_count += 1
        loss.backward()
        self.optimizer.step()

        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0
            # self.call_visor()
        self.policy_net.eval()
        self.target_net.eval()
        self.minimize_epsilon()
        return loss

    def update_target(self):
        """
        updates target network by copying weights from policy network (hard update)
        :return:
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def call_visor(self):
        self.visor.start(self.captain.data, self.policy_net.conv_net, self.viz_data)

    def register_hooks(self):
        self.policy_net.conv_net.conv_1.register_forward_hook(self.captain.hook('conv_1'))
        self.policy_net.conv_net.conv_2.register_forward_hook(self.captain.hook('conv_2'))
        self.policy_net.conv_net.conv_3.register_forward_hook(self.captain.hook('conv_3'))
