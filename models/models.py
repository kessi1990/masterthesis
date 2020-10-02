import torch
import torch.nn as nn
import torch.nn.functional as functional

from utils import visualization as v

################################################################################
#                                  Model parts                                 #
################################################################################


class CNN(nn.Module):
    """
    convolutional neural network (CNN), extracts spatial features
    """
    def __init__(self, device):
        """

        :param device: cpu / gpu
        """
        super(CNN, self).__init__()
        self.device = device
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1)

    def forward(self, state):
        """
        forwards state through CNN and outputs feature maps
        :param state: obtained from environment
        :return: feature maps
        """
        out = state.to(device=self.device)
        out = functional.relu(self.conv_1(out))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        return out


class Attention(nn.Module):
    """
    attention layer, applies attention mechanism
    """
    def __init__(self):
        """

        """
        super(Attention, self).__init__()
        self.fc_1 = nn.Linear(in_features=256, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=256)

    def forward(self, input_vectors, last_hidden_state):
        """
        aligns input_vectors and last hidden state, outputs linear combination (context vector)
        :param input_vectors: transformed feature maps
        :param last_hidden_state: last hidden state of decoder
        :return: context vector, weights
        """
        context = []
        weights = []
        for vector in input_vectors:
            attention_weights = functional.softmax(self.fc_2(torch.tanh(self.fc_1(vector) + last_hidden_state)), dim=-1)
            c = attention_weights * vector
            context.append(c.squeeze())
            weights.append(attention_weights.squeeze())
        context = torch.sum(torch.stack(context, dim=0), dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        weights = torch.stack(weights, dim=0)
        return context, weights


class Encoder(nn.Module):
    """
    encoder (LSTM), encodes input sequence
    """
    def __init__(self, num_layers):
        """

        :param num_layers: number of LSTM layers
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=num_layers)

    def forward(self, input_sequence, hidden_state, hidden_cell):
        """
        forwards input sequence through LSTM and outputs encoded sequence of same length
        :param input_sequence: feature maps from CNN as sequence
        :param hidden_state: current hidden state of LSTM
        :param hidden_cell: current cell state of LSTM
        :return: encoded sequence, last hidden state and last cell state of LSTM
        """
        output, (hidden_state, hidden_cell) = self.lstm(input_sequence, (hidden_state, hidden_cell))
        return output.squeeze(dim=0), (hidden_state, hidden_cell)


class Decoder(nn.Module):
    """
    decoder (LSTM), decodes input sequence
    """
    def __init__(self, num_layers):
        """

        :param num_layers: number of LSTM layers
        """
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=num_layers)

    def forward(self, input_sequence, hidden_state, hidden_cell):
        """

        :param input_sequence: attention weighted input sequence / context vector
        :param hidden_state: current hidden state of LSTM
        :param hidden_cell: current cell state of LSTM
        :return: decoded sequence, last hidden and cell state
        """
        output, (hidden_state, hidden_cell) = self.lstm(input_sequence, (hidden_state, hidden_cell))
        return output.squeeze(dim=0), (hidden_state, hidden_cell)


class QNet(nn.Module):
    """
    q-net, predicts q-values
    """
    def __init__(self, nr_actions):
        """

        :param nr_actions: number of actions
        """
        super(QNet, self).__init__()
        self.fc_1 = nn.Linear(in_features=256, out_features=nr_actions)

    def forward(self, decoder_out):
        """
        takes decoder output and predicts q-values
        :param decoder_out: decoder output
        :return: q-values
        """
        return self.fc_1(decoder_out)


################################################################################
#                                   Models                                     #
################################################################################


class DARQNModel(nn.Module):
    """
    darqn model according to "deep attention recurrent q-network" paper
    """
    def __init__(self, nr_actions, device, num_layers):
        """

        :param nr_actions: number of actions
        :param device: cpu / gpu
        :param num_layers: number of LSTM cells
        """
        super(DARQNModel, self).__init__()
        self.device = device
        self.directory = None
        self.num_layers = num_layers
        self.cnn = CNN(device)
        self.attention = Attention()
        self.decoder = Decoder(num_layers)
        self.q_net = QNet(nr_actions)

        self.dec_h_t = None
        self.dec_c_t = None
        self.init_hidden()

        self.show = True

    def forward(self, input_frames):
        """
        propagates consecutive input frames through cnn, attention layer, decoder and q-net
        :param input_frames: consecutive input frames from environment
        :return: q-values
        """
        q_values = None
        for i, input_frame in enumerate(input_frames):
            feature_maps = self.cnn.forward(input_frame.unsqueeze(dim=0))
            input_vector = self.build_vector(feature_maps)
            input_vector.unsqueeze_(dim=1)
            context, weights = self.attention.forward(input_vector, self.dec_h_t[-1])
            decoder_out, (self.dec_h_t, self.dec_c_t) = self.decoder.forward(context, self.dec_h_t, self.dec_c_t)
            q_values = self.q_net(decoder_out)
            if self.show and self.directory is not None:
                v.visualize_attention(weights, input_frame, self.directory, i)
        self.show = False
        return q_values

    def init_hidden(self, batch_size=1):
        """
        initializes hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.dec_h_t = torch.zeros(self.num_layers, batch_size, 256, device=self.device)
        self.dec_c_t = torch.zeros(self.num_layers, batch_size, 256, device=self.device)

    @staticmethod
    def build_vector(feature_maps):
        """
        builds input vector for lstm from cnn feature maps
        :param feature_maps: cnn output
        :return: input vector for lstm
        """
        batch, _, height, width = feature_maps.shape
        vectors = torch.stack([feature_maps[-1][:, y, x] for y in range(height) for x in range(width)], dim=0)
        return vectors


class CEADModel(nn.Module):
    """
    cead model, extends darqn model with additional lstm encoder
    """
    def __init__(self, nr_actions, device, num_layers):
        """

        :param nr_actions: number of actions
        :param device: cpu / gpu
        :param num_layers: number of LSTM cells
        """
        super(CEADModel, self).__init__()
        self.device = device
        self.directory = None
        self.num_layers = num_layers
        self.cnn = CNN(device)
        self.encoder = Encoder(num_layers)
        self.attention = Attention()
        self.decoder = Decoder(num_layers)
        self.q_net = QNet(nr_actions)

        self.dec_h_t = None
        self.dec_c_t = None
        self.enc_h_t = None
        self.enc_c_t = None
        self.init_hidden()

        self.show = True

    def forward(self, input_frames):
        """
        propagates consecutive input frames through cnn, encoder, attention layer, decoder and q-net
        :param input_frames: consecutive input frames from environment
        :return: q-values
        """
        q_values = None
        for i, input_frame in enumerate(input_frames):
            feature_maps = self.cnn.forward(input_frame.unsqueeze(dim=0))
            input_vector = self.build_vector(feature_maps)
            input_vector.unsqueeze_(dim=1)
            encoder_out, (self.enc_h_t, self.enc_c_t) = self.encoder.forward(input_vector, self.enc_h_t, self.enc_c_t)
            context, weights = self.attention.forward(encoder_out, self.dec_h_t[-1])
            decoder_out, (self.dec_h_t, self.dec_c_t) = self.decoder.forward(context, self.dec_h_t, self.dec_c_t)
            q_values = self.q_net(decoder_out)
            if self.show and self.directory is not None:
                v.visualize_attention(weights, input_frame, self.directory, i)
        self.show = False
        return q_values

    def init_hidden(self, batch_size=1):
        """
        initializes hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.enc_h_t = torch.zeros(self.num_layers, batch_size, 256, device=self.device)
        self.enc_c_t = torch.zeros(self.num_layers, batch_size, 256, device=self.device)
        self.dec_h_t = torch.zeros(self.num_layers, batch_size, 256, device=self.device)
        self.dec_c_t = torch.zeros(self.num_layers, batch_size, 256, device=self.device)

    @staticmethod
    def build_vector(feature_maps):
        """
        builds input vector for lstm from cnn feature maps
        :param feature_maps: cnn output
        :return: input vector for lstm
        """
        batch, _, height, width = feature_maps.shape
        vectors = torch.stack([feature_maps[-1][:, y, x] for y in range(height) for x in range(width)], dim=0)
        return vectors
