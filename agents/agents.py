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

        """
        self.conv_net = conv_first_model.ConvNet(self.in_channels)  # conv3 out shape torch.Size([3, 128, 4, 3])
        self.encoder = conv_first_model.Encoder(input_size=self.input_size_enc,
                                                hidden_size=self.hidden_size_enc,
                                                nr_layers=self.nr_layers_enc).to(self.device)
        self.attention_layer = conv_first_model.Attention(hidden_size=self.input_size_enc,
                                                          alignment_mechanism=config['alignment_function']).to(
            self.device)
        self.decoder = conv_first_model.Decoder(input_size=self.input_size_dec,
                                                hidden_size=self.hidden_size_dec,
                                                nr_layers=self.nr_layers_dec,
                                                output_size=nr_actions).to(self.device)
        self.q_net = conv_first_model.QNet(input_size=self.input_size_fc,
                                           nr_actions=len(self.action_space)).to(self.device)

        # Networks optimizer
        params = list(self.conv_net.parameters()) + list(self.encoder.parameters()) + \
                 list(self.decoder.parameters()) + list(self.q_net.parameters())
        if config['alignment_function'] != 'dot':
            params += list(self.attention_layer.parameters())
        """
        self.policy_net = models.EADModel(config, nr_actions).to(self.device)
        self.target_net = models.EADModel(config, nr_actions).to(self.device)
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
            print(q_values.shape)
            action = np.argmax(q_values[-1].detach().numpy())
            print(action)
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
            prediction = prediction.clone().detach().requires_grad_(True)
            if dones[i]:
                target = rewards[i]
            else:
                q_new = self.target_net.forward(next_state_sequences[i])
                target = rewards[i] + self.discount_factor * np.argmax(q_new[0].detach().numpy())
            target = torch.tensor(target, requires_grad=True)
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

    """
    def forward(self, state_sequence):
        # conv_in 4 x 1 x 84 x 72
        conv_in = torch.stack(list(state_sequence), dim=0)
        # conv_out 4 x 128 x 4 x 3
        conv_out = self.conv_net.forward(conv_in)
        # encoder_in 12, 1, 384
        input_sequence = self.build_vector(conv_out)
        # input_sequence 12, 512 --> 12, 1, 512
        input_sequence.unsqueeze_(dim=1)
        encoder_out, (encoder_h_n, encoder_c_n) = self.encoder.forward(input_sequence)
        h_n, c_n = encoder_h_n, encoder_c_n
        # h_n.unsqueeze_(dim=0)
        # c_n.unsqueeze_(dim=0)
        q_values = []
        for _ in encoder_out:
            decoder_out, (decoder_h_n, decoder_c_n), attentional_hidden = self.decoder.forward(input_sequence,
                                                                                               encoder_out, h_n, c_n)
            h_n = decoder_h_n
            c_n = decoder_c_n
            q_values.append(self.q_net.forward(attentional_hidden))
        return q_values
    

    def build_vector(self, conv_out):
        batch, _, height, width = conv_out.shape
        vectors = [torch.stack([conv_out[i][:, y, x] for y in range(height) for x in range(width)], dim=0)
                   for i in range(batch)]
        if self.vector_combination == 'mean':
            return reduce(lambda t1, t2: t1 + t2, vectors) / len(vectors)
        elif self.vector_combination == 'sum':
            return reduce(lambda t1, t2: t1 + t2, vectors)
        elif self.vector_combination == 'concat':
            return reduce(lambda t1, t2: torch.cat((t1, t2), dim=-1), vectors)
        else:
            pass
    """
