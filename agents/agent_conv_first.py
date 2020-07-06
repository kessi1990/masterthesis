import torch
import torch.nn as nn
import numpy as np
import random
from torch.optim import Adam
from torch.autograd import Variable
from collections import deque
from functools import reduce
from utils import transformation
from models import conv_first_model


class Agent:
    """
    Agent class, builds network architecture from different parts and handles training
    """
    def __init__(self, config, nr_actions, device):
        """

        :param config: configuration parameters obtained from argument parser and configuration loader
        :param nr_actions: number of actions available in the evnironment
        :param device: used for computations, cpu is used if gpu(s) not available
        """

        self.device = device
        self.transformation = transformation.Transformation(config)

        # DQN parameter
        self.action_space = [_ for _ in range(nr_actions)]
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.train_start = config['train_start']
        self.mem_buffer = deque(maxlen=config['mem_size'])

        # LSTM parameter
        self.input_size_enc = config['input_size_enc']
        self.hidden_size_enc = config['hidden_size_enc']
        self.nr_layers_enc = config['nr_layers_enc']
        self.input_size_dec = config['input_size_dec']
        self.hidden_size_dec = config['hidden_size_dec']
        self.nr_layers_dec = config['nr_layers_dec']

        # Attention parameter
        self.attention_mechanism = config['attention_mechanism']

        # Convolutional parameter
        self.in_channels = config['in_channels']

        # Fully connected parameter
        self.input_size_fc = config['input_size_fc']

        # Networks initialization
        self.conv_net = conv_first_model.ConvNet(self.in_channels)  # conv3 out shape torch.Size([3, 128, 4, 3])
        self.encoder = conv_first_model.Encoder(input_size=self.input_size_enc,
                                                hidden_size=self.hidden_size_enc,
                                                nr_layers=self.nr_layers_enc).to(self.device)
        self.attention_layer = conv_first_model.Attention(hidden_size=self.input_size_enc,
                                                          alignment_mechanism='location').to(self.device)
        self.decoder = conv_first_model.Decoder(input_size=self.input_size_dec,
                                                hidden_size=self.hidden_size_dec,
                                                nr_layers=self.nr_layers_dec).to(self.device)
        self.q_net = conv_first_model.QNet(input_size=self.input_size_fc,
                                           nr_actions=len(self.action_space)).to(self.device)

        # Networks optimizer
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.conv_optimizer = Adam(self.conv_net.parameters(), lr=self.learning_rate)
        self.q_optimizer = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.attention_optimizer = Adam(self.attention_layer.parameters(), lr=self.learning_rate)

        self.criterion_lstm = nn.BCELoss().to(self.device)
        self.criterion_q = nn.MSELoss().to(self.device)

        self.overall_lstm_loss = []
        self.overall_q_loss = []

    def append_sample(self, state_sequence, action, reward, next_state, done):
        """
        stores transitions in memory buffer - used for experience replay
        :param state_sequence: sequence (length=3) of consecutive states
        :param action: executed action
        :param reward: reward achieved by executing action in last state of state_sequence
        :param next_state: successor state
        :param done: flag that marks if next_state is terminal state
        :return:
        """
        self.mem_buffer.append((state_sequence, action, reward, next_state, done))

    def policy(self, state_sequence):
        """
        policy of learning agent -> epsilon-greedy
        :param state_sequence: sequence (length=3) of consecutive states
        :return: action
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            self.set_eval()
            # transform image sequence (crop -> resize -> normalize -> grayscale) to tensor, unsqueeze along dim=0
            # and concat along dim=0 --> in: 3 x (210, 160, 3) --> out tensor(3, 1, 84, 72)
            input_sequence = reduce((lambda t1, t2: torch.cat((t1, t2), dim=0)),
                                    list(map((lambda img: self.transformation.transform(img).unsqueeze(dim=0)),
                                             state_sequence)))
            conv_out = self.conv_net.forward(input_sequence)
            conv_out_reshaped = conv_out.reshape(3, 1, 1536)
            encoder_out, (encoder_h_n, encoder_c_n) = self.encoder.forward(conv_out_reshaped)
            decoder_out, (decoder_h_n, decoder_c_n), context = self.decoder.forward(encoder_out, encoder_h_n[-1],
                                                                                    encoder_c_n[-1])
            q_in = decoder_out[-1]
            q_values = self.q_net.forward(q_in)
            action = torch.argmax(q_values[0]).item()
            return action

    def minimize_epsilon(self):
        """
        minimizes epsilon value which is used for epsilon-greedy action selection
        :return:
        """
        self.epsilon *= self.epsilon_decay

    def set_train(self):
        """
        zeros gradients and sets models to train mode
        :return:
        """
        # Zero the gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.conv_net.zero_grad()
        self.q_net.zero_grad()
        self.attention_layer.zero_grad()
        # Set to train mode
        self.encoder.train()
        self.decoder.train()
        self.conv_net.train()
        self.q_net.train()
        self.attention_layer.train()

    def set_eval(self):
        """
        sets models to evaluation mode
        :return:
        """
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.conv_net.eval()
        self.q_net.eval()
        self.attention_layer.eval()

    def train(self):
        """
        trains neural models
        :return:
        """
        self.set_eval()
        mini_batch = random.sample(self.mem_buffer, self.batch_size)
        lstm_loss = 0
        q_loss = 0

        for i, sample in enumerate(mini_batch):
            state_sequence = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state = self.transformation.transform(sample[3])
            done = sample[4]

            # transform image sequence (crop -> resize -> normalize -> grayscale) to tensor, unsqueeze along dim=0
            # and concat along dim=0 --> in: 3 x (210, 160, 3) --> out tensor(3, 1, 84, 72)
            input_sequence = reduce((lambda t1, t2: torch.cat((t1, t2), dim=0)),
                                    list(map((lambda img: self.transformation.transform(img).unsqueeze(dim=0)),
                                             state_sequence)))

            conv_out = self.conv_net.forward(input_sequence)
            conv_out_reshaped = conv_out.reshape(3, 1, 1536)
            encoder_out, (encoder_h_n, encoder_c_n) = self.encoder.forward(conv_out_reshaped)
            decoder_out, (decoder_h_n, decoder_c_n), context = self.decoder.forward(encoder_out, encoder_h_n[-1],
                                                                                    encoder_c_n[-1])

            lstm_loss += self.train_lstm(decoder_out, context)
            q_loss += self.train_q(next_state, conv_out, action, reward, done)

        q_loss = Variable(q_loss, requires_grad=True)
        q_loss.backward()
        lstm_loss.backward()
        self.train_step()

        return lstm_loss, q_loss

    def train_q(self, next_state, conv_out, action, reward, done):
        """
        trains Q-Net part of neural network
        :param next_state: successor state
        :param conv_out: output of convulutional network part
        :param action: performed action in state
        :param reward: reward achieved by executing action in state
        :param done: flag if next_state is terminal state
        :return:
        """
        next_state.unsqueeze_(dim=0)
        state = conv_out[-1]
        state = state.reshape(1, 1536)
        q_old = self.q_net.forward(state)[0][action].item()
        if done:
            y_hat = reward
        else:
            conv_out_next = self.conv_net.forward(next_state)
            next_state = conv_out_next[-1]
            next_state = next_state.reshape(1, 1536)
            q_new = torch.max(self.q_net.forward(next_state)[0]).item()
            y_hat = reward + self.gamma * q_new - q_old
        y = torch.tensor([q_old])
        y_hat = torch.tensor([y_hat])
        # criterion_q() --> q_loss
        return self.criterion_q(y, y_hat)

    def train_lstm(self, decoder_out, context):
        decoder_out = decoder_out[-1].squeeze()
        context = context[-1].squeeze()
        return self.criterion_lstm(torch.sigmoid(decoder_out), context.detach())

    def train_step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.conv_optimizer.step()
        self.q_optimizer.step()
        self.attention_optimizer.step()
