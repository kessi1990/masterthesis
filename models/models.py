import torch
import torch.nn as nn
import torch.nn.functional as functional

from utils import shapes


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
        bias = True
        if self.alignment == 'general':
            self.fc_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)
        elif self.alignment == 'concat':
            self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=bias)
        elif self.alignment == 'concat_fc':
            self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=bias)
            self.fc_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)
        else:  # self.alignment == 'dot'
            pass
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=1, bias=bias)

    def forward(self, input_vectors, last_hidden_state):
        """
        compute attention based on all input_vectors and last_hidden_state of decoder LSTM
        :param input_vectors: transformed feature maps
        :param last_hidden_state: last hidden state of decoder
        :return: context vector, weights
        """
        # b = batch
        # last_hidden_state (b, 128) -> unsqueeze(dim=1) -> (b, 1, 128)
        if self.alignment == 'general':
            """
            # general
            # align(v_it, h_t−1) = h^T_t−1 * (W_a(v_it) + b_a)
            # --------------------------------------------------------------------------------------------------------
            # 1. weights matrix with bias (fc_1) -> (b, 49, 128) 
            # 2. dot product transposed last_hidden_state (b, 1, 128)^T * input_vectors (b, 49, 128)
            # --------------------------------------------------------------------------------------------------------
            """
            alignment_scores = torch.bmm(self.fc_1(input_vectors), last_hidden_state.unsqueeze(dim=1).permute(0, 2, 1))
        elif self.alignment == 'concat':
            """
            # concat
            # align(v_it, h_t−1) = W_s(tanh(W_a[v_it ; h_t−1] + b_a)) + b_s
            # --------------------------------------------------------------------------------------------------------
            # 1. concat input_vectors (b, 49, 128) and last_hidden_state (b, 1, 128) -> (b, 49, 256)
            # 2. weights matrix with bias (fc_1) -> (b, 49, 128) 
            # 3. apply hyperbolic tangent function -> aligned input_vectors (b, 49, 128)
            # 4. alignment_score for each input_vector regarding last_hidden_state:
            # -> aligned input_vectors (b, 49, 128) -> weights matrix with bias (fc_2) -> alignment_scores (b, 49, 1)
            # --------------------------------------------------------------------------------------------------------
            """
            # batch, seq_len, features
            _, seq_len, _ = input_vectors.shape
            alignment_scores = self.fc_2(torch.tanh(self.fc_1(torch.cat((input_vectors, last_hidden_state.unsqueeze(dim=1).expand(-1, seq_len, -1)), dim=-1))))
        elif self.alignment == 'concat_fc':
            """
            # concat_fc
            # align(v_it, h_t−1) = W_s(tanh(W_a[v_it ; W_h(h_t−1) + b_h] + b_a)) + b_s
            # --------------------------------------------------------------------------------------------------------
            # 1. weights matrix with bias (fc_3) to last_hidden_state -> (b, 1, 128) 
            # 2. concat input_vectors (b, 49, 128) and last_hidden_state (b, 1, 128) -> (b, 49, 256)
            # 3. weights matrix with bias (fc_1) -> (b, 49, 128) 
            # 4. apply hyperbolic tangent function -> aligned input_vectors (b, 49, 128)
            # 5. alignment_score for each input_vector regarding last_hidden_state:
            # -> aligned input_vectors (b, 49, 128) -> weights matrix with bias (fc_2) -> alignment_scores (b, 49, 1)
            # --------------------------------------------------------------------------------------------------------
            """
            # batch, seq_len, features
            _, seq_len, _ = input_vectors.shape
            alignment_scores = self.fc_2(torch.tanh(self.fc_1(torch.cat((input_vectors, self.fc_3(last_hidden_state).unsqueeze(dim=1).expand(-1, seq_len, -1)), dim=-1))))
        else:
            """
            # dot
            # align(v_it, h_t−1) = h^T_t−1 * v_it
            # --------------------------------------------------------------------------------------------------------
            # 1. dot product transposed last_hidden_state (b, 1, 128)^T * input_vectors (b, 49, 128)
            # --------------------------------------------------------------------------------------------------------
            """
            alignment_scores = torch.bmm(input_vectors, last_hidden_state.unsqueeze(dim=1).permute(0, 2, 1))
        """
        # softmax + linear combination
        # --------------------------------------------------------------------------------------------------------
        # apply softmax function to dim=1 -> importance of each input_vector -> attention_weights (b, 49, 1)
        # pointwise multiplication of input_vectors (b, 49, 128) and their corresponding attention value (b, 49, 1)  -> (b, 49, 128)
        # compute sum of these products (b, 49, 128) along dim=1 to obtain context_vector z (b, 1, 128)  |  == linear combination
        # --------------------------------------------------------------------------------------------------------
        """
        attention_weights = functional.softmax(alignment_scores, dim=1)
        context = input_vectors * attention_weights
        z = torch.sum(context, dim=1, keepdim=True)
        """
        z = torch.bmm(attention_weights.permute(0, 2, 1), input_vectors)
        """
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
        feature_maps = self.cnn.forward(input_frame)  # -> (b, 128, 7, 7)
        input_vector = shapes.build_vector(feature_maps)  # -> (b, 49, 128)
        context, weights = self.attention.forward(input_vector, self.dec_h_t[-1])  # -> (b, 1, 128), (b, 49, 1)
        decoder_out, (self.dec_h_t, self.dec_c_t) = self.decoder.forward(context, self.dec_h_t, self.dec_c_t)  # -> (b, 1, 128), (n, b, 128), (n, b, 128);  n = num_layers)
        q_values = self.q_net.forward(decoder_out.squeeze(dim=1))  # (b, a);  a = nr_actions
        return q_values, context, weights

    def init_hidden(self, batch_size=1):
        """
        initialize hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.dec_h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.dec_c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)


class CEADModel(nn.Module):
    """

    """
    def __init__(self, nr_actions, device, num_layers, hidden_size, alignment):
        """

        :param nr_actions:
        :param device:
        :param num_layers:
        :param hidden_size:
        :param alignment:
        """
        super(CEADModel, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.cnn = CNN(device, hidden_size)
        self.attention = Attention(alignment, hidden_size)
        self.att_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = Decoder(num_layers, hidden_size)
        self.q_net = QNet(nr_actions, hidden_size)

        self.hidden_size = hidden_size
        self.dec_h_t = None
        self.dec_c_t = None
        self.attentional_hidden = None
        self.init_hidden()

    def forward(self, input_frame):
        """

        :param input_frame:
        :return:
        """
        feature_maps = self.cnn.forward(input_frame)  # -> (b, 128, 7, 7)
        input_vector = shapes.build_vector(feature_maps)  # -> (b, 49, 128)
        decoder_out, (self.dec_h_t, self.dec_c_t) = self.decoder.forward(self.attentional_hidden, self.dec_h_t, self.dec_c_t)  # -> (b, 1, 128), (n, b, 128), (n, b, 128);  n = num_layers
        context, weights = self.attention.forward(input_vector, self.dec_h_t[-1])  # -> (b, 1, 128), (b, 49, 1)
        self.attentional_hidden = torch.tanh(self.att_concat(torch.cat((context, self.dec_h_t.permute(1, 0, 2)), dim=-1)))
        q_values = self.q_net.forward(self.attentional_hidden.squeeze(dim=1))  # (b, a);  a = nr_actions
        return q_values, context, weights

    def init_hidden(self, batch_size=1):
        """
        initialize hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.dec_h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.dec_c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        self.attentional_hidden = torch.zeros(batch_size, self.num_layers, self.hidden_size, device=self.device)


class DQNModel(nn.Module):
    """

    """
    def __init__(self, nr_actions, device, in_channels=4):
        """

        :param in_channels:
        :param nr_actions:
        :param device:
        """
        super(DQNModel, self).__init__()
        self.device = device
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(in_features=3136, out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=nr_actions)

    def forward(self, state):
        """

        :param state:
        :return:
        """
        state = state.to(device=self.device)
        out = functional.relu(self.conv_1(state))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        batch, *shape = out.shape
        out = functional.relu(self.fc_1(out.view(batch, -1)))
        return self.fc_2(out)


class NoLSTM(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels, alignment, hidden_size, nr_actions, device):
        """

        :param in_channels:
        :param out_channels:
        :param alignment:
        :param hidden_size:
        :param nr_actions:
        :param device:
        """
        super(NoLSTM, self).__init__()
        self.device = device
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.attention = Attention(alignment, hidden_size)
        self.q_net = nn.Linear(in_features=hidden_size, out_features=nr_actions)

        self.hidden_size = hidden_size
        self.last_hidden = None
        self.init_hidden()

    def forward(self, state):
        """

        :param state:
        :return:
        """
        state = state.to(device=self.device)
        feature_maps = functional.relu(self.conv.forward(state))
        input_vectors = shapes.build_vector(feature_maps)
        context, weights = self.attention.forward(input_vectors, self.last_hidden.squeeze(dim=1))
        q_values = self.q_net.forward(context.squeeze(dim=1))
        self.last_hidden = context
        return q_values, weights

    def init_hidden(self, batch_size=1):
        """
        initialize hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.last_hidden = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)


class Identity(nn.Module):
    """

    """
    def __init__(self, alignment, hidden_size, nr_actions, device):
        """

        :param alignment:
        :param hidden_size:
        :param nr_actions:
        :param device:
        """
        super(Identity, self).__init__()
        self.device = device
        self.attention = Attention(alignment, hidden_size)
        self.q_net = nn.Linear(in_features=hidden_size, out_features=nr_actions)

        self.hidden_size = hidden_size
        self.last_hidden = None
        self.init_hidden()

    def forward(self, state):
        """

        :param state:
        :return:
        """
        state = state.to(device=self.device)
        input_vectors = shapes.build_vector(state)
        context, weights = self.attention.forward(input_vectors, self.last_hidden.squeeze(dim=1))
        q_values = self.q_net.forward(context.squeeze(dim=1))
        self.last_hidden = context
        return q_values, weights

    def init_hidden(self, batch_size=1):
        """
        initialize hidden and cell state of lstm with zeros
        :param batch_size: size of mini batch
        :return:
        """
        self.last_hidden = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)

