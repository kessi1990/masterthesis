import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
import random

from collections import deque


class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.device = device
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1)

    def forward(self, state):
        out = state.to(device=self.device)
        out = functional.relu(self.conv_1(out))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        return out


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.fc_1 = nn.Linear(in_features=256, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=256)

    def forward(self, feature_maps, last_hidden_state):
        input_vectors = self.build_vector(feature_maps)
        context = []
        for vector in input_vectors:
            # print(f'vector {vector.shape}')
            # print(f'last_hidden_state {last_hidden_state.shape}')
            attention_weighted_vector = self.fc_1(vector) + last_hidden_state
            # print(f'attention_weighted_vector {attention_weighted_vector.shape}')
            attention_weighted_vector = torch.tanh(attention_weighted_vector)
            attention_weighted_vector = self.fc_2(attention_weighted_vector)
            attention_weighted_vector = functional.softmax(attention_weighted_vector, dim=-1)  # / normalizing_const
            c = attention_weighted_vector * vector
            context.append(c.squeeze())
        context = torch.sum(torch.stack(context, dim=0), dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        return context

    @staticmethod
    def build_vector(feature_maps):
        batch, _, height, width = feature_maps.shape
        vectors = torch.stack([feature_maps[-1][:, y, x] for y in range(height) for x in range(width)], dim=0)
        return vectors


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)

    def forward(self, input_sequence, hidden_state, hidden_cell):
        output, (hidden_state, hidden_cell) = self.lstm(input_sequence, (hidden_state, hidden_cell))
        return output.squeeze(dim=0), (hidden_state, hidden_cell)


class QNet(nn.Module):
    def __init__(self, nr_actions):
        super(QNet, self).__init__()
        self.fc_1 = nn.Linear(in_features=256, out_features=nr_actions)

    def forward(self, decoder_out):
        return self.fc_1(decoder_out)


class DARQNModel(nn.Module):
    def __init__(self, nr_actions, device):
        super(DARQNModel, self).__init__()
        self.device = device
        self.cnn = CNN(device)
        self.attention = Attention()
        self.decoder = Decoder()
        self.q_net = QNet(nr_actions)

        self.hidden_state = None
        self.cell_state = None

    def forward(self, input_frames):
        for input_frame in input_frames:
            if self.hidden_state is None:
                # print(f'hidden_state is {self.hidden_state}')
                self.init_hidden()
            cnn_out = self.cnn.forward(input_frame.unsqueeze(dim=0))
            # print(f'cnn_out {cnn_out.shape}')
            context = self.attention.forward(cnn_out, self.hidden_state)
            # print(f'context {context.shape}')
            decoder_out, (self.hidden_state, self.cell_state) = self.decoder.forward(context, self.hidden_state, self.cell_state)
            # print(f'decoder_out {decoder_out.shape}')
            q_values = self.q_net(decoder_out)
            # print(f'q_values {q_values.shape}')
        return q_values

    def init_hidden(self, batch_size=1):
        self.hidden_state = torch.zeros(1, batch_size, 256, device=self.device)
        self.cell_state = torch.zeros(1, batch_size, 256, device=self.device)


class DARQNAgent:
    def __init__(self, nr_actions, device):
        self.nr_actions = nr_actions
        self.action_space = [_ for _ in range(self.nr_actions)]
        print(f'nr_actions: {self.nr_actions}, action_space: {self.action_space}')
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.000001
        self.epsilon_min = 0.05
        self.discount_factor = 0.99
        self.batch_size = 32
        self.memory = deque(maxlen=500000)
        self.k_count = 0
        self.k_target = 10000

        self.device = device

        self.policy_net = DARQNModel(self.nr_actions, self.device).to(self.device)
        self.target_net = DARQNModel(self.nr_actions, self.device).to(self.device)

        params = list(self.policy_net.parameters()) + list(self.target_net.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(self.device)

    def append_sample(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def training_policy(self, state_sequence):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = self.policy_net.forward(state_sequence)
            action = torch.argmax(q_values[0]).item()
            return action

    def evaluation_policy(self, state_sequence):
        if np.random.rand() <= 0.05:
            return np.random.choice(self.action_space)
        else:
            q_values = self.policy_net.forward(state_sequence)
            action = torch.argmax(q_values[0]).item()
            return action

    def minimize_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.optimizer.zero_grad()
        self.policy_net.train()
        self.target_net.train()
        self.policy_net.init_hidden()
        self.target_net.init_hidden()
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

        if self.k_count >= self.k_target:
            print(f'updating target network')
            self.update_target()
            self.k_count = 0
        self.policy_net.eval()
        self.target_net.eval()
        self.minimize_epsilon()
        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())