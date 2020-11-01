import torch
import torch.nn as nn
import torch.nn.functional as functional


################################################################################
#                                  Model parts                                 #
################################################################################


class CNN(nn.Module):
    """
    convolutional neural network (CNN), extraction of spatial features
    """
    def __init__(self, device, hidden_size):
        """
        init of CNN layer
        :param device: cpu / gpu
        :param hidden_size: hidden_size of LSTM -> out_channels / num_feature_maps of last convolutional layer
        """
        super(CNN, self).__init__()
        self.device = device
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=hidden_size, kernel_size=3, stride=1)

    def forward(self, state):
        """
        forward state through CNN and outputs feature maps
        :param state: state -> observation obtained from environment
        :return: feature maps
        """
        state = state.to(device=self.device)
        out = functional.relu(self.conv_1(state))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        return out


class Attention(nn.Module):
    """
    attention layer, applies attention mechanism
    """
    def __init__(self, alignment, hidden_size):
        """

        :param alignment: alignment function for computation of alignment_score
        :param hidden_size: hidden_size of LSTM -> feature_size of attention layers
        """
        super(Attention, self).__init__()
        self.alignment = alignment
        # TODO bias=False?
        if self.alignment == 'add':
            self.fc_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        else:
            self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        # TODO bias=False?
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, input_vectors, last_hidden_state):
        """
        compute attention based on all input_vectors and last_hidden_state of decoder LSTM
        :param input_vectors: transformed feature maps
        :param last_hidden_state: last hidden state of decoder
        :return: context vector, weights
        """
        # b = batch
        # last_hidden_state (b, 256) -> unsqueeze(dim=1) -> (b, 1, 256)
        # input_vectors (b, 49, 256) -> fc_1 / learnable weights -> (b, 49, 256)
        # input_vectors (b, 49, 256) + last_hidden_state (b, 1, 256) -> (b, 49, 256)  |  add last_hidden_state to every input_vector
        # apply hyperbolic tangent function -> aligned input_vectors (b, 49, 256)
        # aligned input_vectors (b, 49, 256) -> fc_2 / learnable weights -> alignment_scores (b, 49, 1)  |  alignment_score / energy for each input_vector regarding last_hidden_state
        # apply softmax function to dim=1 -> importance of each input_vector / attention_weights (b, 49, 1)
        # pointwise multiplication of input_vectors (b, 49, 256) and their corresponding attention value (b, 49, 1)  -> (b, 49, 256)
        # compute sum of these products (b, 49, 256) along dim=1 to obtain context_vector z (b, 1, 256)  |  == linear combination
        if self.alignment == 'add':
            # add
            alignment_scores = self.fc_2(torch.tanh(self.fc_1(input_vectors) + last_hidden_state.unsqueeze(dim=1)))
        else:
            # concat
            alignment_scores = self.fc_2(torch.tanh(self.fc_1(torch.cat((input_vectors, last_hidden_state.unsqueeze(dim=1)), dim=-1))))
        attention_weights = functional.softmax(alignment_scores, dim=1)
        context = input_vectors * attention_weights
        z = torch.sum(context, dim=1, keepdim=True)
        return z, attention_weights


class Encoder(nn.Module):
    """
    encoder (LSTM), encodes input sequence
    """
    def __init__(self, num_layers, hidden_size):
        """

        :param num_layers: number of LSTM layers
        :param hidden_size: hidden_size of LSTM
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input_sequence, hidden_state, hidden_cell):
        """
        forwards input sequence through LSTM and outputs encoded sequence of same length
        :param input_sequence: feature maps from CNN as sequence
        :param hidden_state: current hidden state of LSTM
        :param hidden_cell: current cell state of LSTM
        :return: encoded sequence, last hidden state and last cell state of LSTM
        """
        output, (hidden_state, hidden_cell) = self.lstm(input_sequence, (hidden_state, hidden_cell))
        return output, (hidden_state, hidden_cell)


class Decoder(nn.Module):
    """
    decoder (LSTM), decodes input sequence
    """
    def __init__(self, num_layers, hidden_size):
        """

        :param num_layers: number of LSTM layers
        :param hidden_size: hidden_size of LSTM
        """
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input_sequence, hidden_state, cell_state):
        """

        :param input_sequence: attention weighted input sequence / context vector
        :param hidden_state: current hidden state of LSTM
        :param cell_state: current cell state of LSTM
        :return: decoded sequence, last hidden and cell state
        """
        output, (hidden_state, cell_state) = self.lstm(input_sequence, (hidden_state, cell_state))
        return output, (hidden_state, cell_state)


class QNet(nn.Module):
    """
    q-net, predicts q-values
    """
    def __init__(self, nr_actions, hidden_size):
        """

        :param nr_actions: number of actions
        :param hidden_size: hidden_size of LSTM -> feature_size / input_size of fully connected layer / q-net
        """
        super(QNet, self).__init__()
        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=nr_actions)

    def forward(self, decoder_out):
        """
        takes decoder output and predicts q-values
        :param decoder_out: decoder output
        :return: q-values
        """
        out = self.fc_1(decoder_out)
        return out


################################################################################
#                                   Models                                     #
################################################################################


class DARQNModel(nn.Module):
    """
    darqn model according to "deep attention recurrent q-network" paper
    """
    def __init__(self, nr_actions, device, num_layers, hidden_size, alignment):
        """

        :param nr_actions: number of possible actions in environment
        :param device: cpu / gpu
        :param num_layers: number of stacked LSTM layers
        :param hidden_size: hidden_size / feature_size of decoder LSTM
        :param alignment: alignment function for attention layer
        """
        super(DARQNModel, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.cnn = CNN(device, hidden_size)
        self.attention = Attention(alignment, hidden_size)
        self.decoder = Decoder(num_layers, hidden_size)
        self.q_net = QNet(nr_actions, hidden_size)

        self.hidden_size = hidden_size
        self.dec_h_t = None
        self.dec_c_t = None
        self.init_hidden()

    def forward(self, input_frame):
        """
        forward input_frames through darqn architecture parts. q-value prediction based on last input_frame
        :param input_frame: consecutive input_frames (len=4)
        :return: returns q-values for each action
        """
        # b = batch
        # input_frames = sequence of 4 frames for evaluation mode, sequence of 4 batches of states for training mode
        # evaluation (b =  1): input_frames: list of tensors: 4 x (b, 1, 84, 84) ->  4 x (batch, channels, height, width)
        # training   (b = 32): input_frames: 1 tensor: (4 x b x 1 x 84 x 84) -> (input_frames, batch, channels, height, width)
        feature_maps = self.cnn.forward(input_frame)  # -> (b, 256, 7, 7)
        input_vector = self.build_vector(feature_maps)  # -> (b, 49, 256)
        context, weights = self.attention.forward(input_vector, self.dec_h_t[-1])  # -> (b, 1, 256), (b, 49, 1)
        decoder_out, (self.dec_h_t, self.dec_c_t) = self.decoder.forward(context, self.dec_h_t, self.dec_c_t)  # -> (b, 1, 256), (n, b, 256), (n, b, 256);  n = num_layers)
        q_values = self.q_net.forward(decoder_out.squeeze(dim=1))  # (b, a);  a = nr_actions
        return q_values

    def init_hidden(self, batch_size=1):
        """
        initialize hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.dec_h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.dec_c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

    @staticmethod
    def build_vector(feature_maps):
        """
        transform feature_maps from CNN to sequence of pixel vectors / input_vectors for LSTM input
        (batch, channels / num_feature_maps, height, width) --> (batch, seq_len, features); e.g.
        (32, 256, 7, 7) --> (32, 49, 256)
        :param feature_maps: cnn output
        :return: input vector for lstm
        """
        batch, _, height, width = feature_maps.shape
        vectors = [torch.stack([feature_maps[i][:, y, x] for y in range(height) for x in range(width)], dim=0)
                   for i in range(batch)]  # slice pixel vectors -> dimension of single vector = number of feature maps, e.g. 256
        return torch.stack(vectors, dim=0)  # stack vectors at dim=0


class CEADModel(nn.Module):
    """
    cead model, extends darqn model with additional lstm encoder
    """
    def __init__(self, nr_actions, device, num_layers, hidden_size, alignment):
        """

        :param nr_actions: number of actions
        :param device: cpu / gpu
        :param num_layers: number of LSTM cells
        """
        super(CEADModel, self).__init__()
        self.device = device
        self.directory = None
        self.num_layers = num_layers
        self.cnn = CNN(device, hidden_size)
        self.encoder = Encoder(num_layers, hidden_size)
        self.attention = Attention(alignment, hidden_size)
        self.decoder = Decoder(num_layers, hidden_size)
        self.q_net = QNet(nr_actions, hidden_size)

        self.hidden_size = hidden_size
        self.dec_h_t = None
        self.dec_c_t = None
        self.enc_h_t = None
        self.enc_c_t = None
        self.init_hidden()

    def forward(self, input_frames):
        """
        propagates consecutive input frames through cnn, encoder, attention layer, decoder and q-net
        :param input_frames: consecutive input frames from environment
        :return: q-values
        """
        # TODO: obsolete --> implement modified cead model
        """
        q_values = None
        context = None
        ws = []
        for input_frame in input_frames:
            feature_maps = self.cnn.forward(input_frame)
            input_vector = self.build_vector(feature_maps)
            encoder_out, (self.enc_h_t, self.enc_c_t) = self.encoder.forward(input_vector, self.enc_h_t, self.enc_c_t)
            context, weights = self.attention.forward(encoder_out, self.dec_h_t[-1])
            ws.append(weights)
            decoder_out, (self.dec_h_t, self.dec_c_t) = self.decoder.forward(context, self.dec_h_t, self.dec_c_t)
            q_values = self.q_net.forward(decoder_out)"""
        return None  # q_values, context, ws

    def init_hidden(self, batch_size=1):
        """
        initializes hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.enc_h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.enc_c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.dec_h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.dec_c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

    @staticmethod
    def build_vector(feature_maps):
        """
        builds input vector for lstm from cnn feature maps
        :param feature_maps: cnn output
        :return: input vector for lstm
        """
        batch, _, height, width = feature_maps.shape
        vectors = [torch.stack([feature_maps[i][:, y, x] for y in range(height) for x in range(width)], dim=0)
                   for i in range(batch)]  # slice pixel vectors -> dimension of single vector = number of feature maps, e.g. 256
        return torch.stack(vectors, dim=0)  # stack vectors at dim=0


class DQNModel(nn.Module):
    def __init__(self, nr_actions, device):
        super(DQNModel, self).__init__()
        self.device = device
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(in_features=12544, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=nr_actions)

    def forward(self, state):
        state = state.to(device=self.device)
        out = functional.relu(self.conv_1(state))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        batch, *shape = out.shape
        out = functional.relu(self.fc_1(out.view(batch, -1)))
        return self.fc_2(out)
