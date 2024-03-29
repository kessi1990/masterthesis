import torch
import torch.nn as nn
import torch.nn.functional as functional

from functools import reduce

################################################################################
#                                  Model parts                                 #
################################################################################


class CNN(nn.Module):
    """
    convolutional neural network (CNN)
    """
    def __init__(self, in_channels, device):
        """

        :param in_channels: number of channels
        """
        super(CNN, self).__init__()
        self.device = device
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=2, stride=1)

    def forward(self, input_sequence):
        """
        forwards input sequence through CNN and outputs feature maps
        :param input_sequence: consecutive states from environment
        :return: returns feature maps
        """
        out = input_sequence.to(device=self.device)
        out = functional.relu(self.conv_1(out))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        return out


class Encoder(nn.Module):
    """
    encoder (LSTM), encodes input sequence
    """
    def __init__(self, input_size, hidden_size, nr_layers, device):
        """

        :param input_size: size of input features
        :param hidden_size: size hidden features
        :param nr_layers: number of stacking layers
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.nr_layers = nr_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, nr_layers)

    def forward(self, input_sequence):
        """
        forwards input sequence through LSTM and outputs encoded sequence of same length
        :param input_sequence: feature maps from CNN as sequence
        :return: encoded sequence, last hidden state and last cell state of LSTM
        """
        (h_0, c_0) = self.init_hidden(self.device)
        output_sequence, (h_n, c_n) = self.lstm(input_sequence, (h_0, c_0))
        return output_sequence, (h_n, c_n)

    def init_hidden(self, device, batch_size=1):
        """
        initializes first hidden state and cell state of LSTM
        :param batch_size: batch size
        :param device: device on which returned tensor is stored (CPU / GPU)
        :return: initial hidden state and cell state tensor
        """
        return (torch.zeros(self.nr_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.nr_layers, batch_size, self.hidden_size, device=device))


class Attention(nn.Module):
    """
    attention layer, applies attention mechanism
    """
    def __init__(self, hidden_size, alignment_mechanism):
        """

        :param hidden_size: size of input features
        :param alignment_mechanism: defines which alignment method is used
        """
        super(Attention, self).__init__()
        self.alignment_mechanism = alignment_mechanism

        if self.alignment_mechanism == 'dot':
            return
        elif self.alignment_mechanism == 'general':
            self.alignment_function = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.alignment_mechanism == 'concat':
            self.weight = nn.Parameter(torch.rand([1, hidden_size]))
            self.alignment_function = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        """
        aligns current target hidden state (decoder) with all source hidden state (encoder)
        :param encoder_out: encoded sequence, contains all source hidden states
        :param decoder_hidden: current target hidden states
        :return: alignment score (scalar)
        """
        if self.alignment_mechanism == 'dot':
            return torch.matmul(encoder_out.squeeze(), decoder_hidden[-1].squeeze())
        elif self.alignment_mechanism == 'general':
            return torch.matmul(encoder_out.squeeze(), self.alignment_function(decoder_hidden[-1].squeeze()))
        elif self.alignment_mechanism == 'concat':
            concat = torch.stack([torch.cat((member, decoder_hidden[-1]), dim=-1) for member in encoder_out], dim=0)
            aligned = torch.tanh(self.alignment_function(concat))
            return torch.matmul(aligned.squeeze(), self.weight.squeeze())


class Decoder(nn.Module):
    """
    decoder (LSTM), decodes input sequence
    """
    def __init__(self, input_size, hidden_size, nr_layers, alignment):
        """

        :param input_size: size of input features
        :param hidden_size: size hidden features
        :param nr_layers: number of stacking layers
        :param alignment: defines which alignment method is used
        """
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_size=hidden_size, alignment_mechanism=alignment)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nr_layers)
        self.concat_layer = nn.Linear(input_size*2, input_size)

    def forward(self, input_sequence, encoder_out, hidden_state, cell_state):
        """
        forwards input sequence through decoder (LSTM), applies attention to input
        :param input_sequence: feature maps from CNN as sequence
        :param encoder_out: encoded sequence, contains all source hidden states
        :param hidden_state: last hidden state
        :param cell_state: last cell state
        :return:
        """

        output = []
        context_vectors = []
        applied_attention = []
        weights = []

        input_vector = hidden_state[-1]
        for _ in input_sequence:
            decoder_out, (decoder_hidden_s, decoder_hidden_c) = self.lstm(input_vector.unsqueeze(dim=0),
                                                                          (hidden_state, cell_state))
            alignment_vector = self.attention.forward(encoder_out, decoder_hidden_s)
            attention_weights = functional.softmax(alignment_vector, dim=0)
            attention_applied = torch.mul(encoder_out.squeeze(), attention_weights.unsqueeze(dim=1))
            context = torch.sum(attention_applied, dim=0)
            context_concat_hidden = torch.cat((context.unsqueeze(dim=0), decoder_hidden_s[-1]), dim=-1)
            attentional_hidden = torch.tanh(self.concat_layer(context_concat_hidden))

            # next iteration
            input_vector = attentional_hidden
            hidden_state = decoder_hidden_s
            cell_state = decoder_hidden_c

            output.append(attentional_hidden)

            # for visualization / plotting reasons
            context_vectors.append(context)
            applied_attention.append(attention_applied)
            weights.append(attention_weights)

        output = torch.stack(output, dim=0)

        # for visualization / plotting reasons
        context_vectors = torch.stack(context_vectors, dim=0).unsqueeze(dim=1)  # --> why unsqueeze() ?
        applied_attention = torch.stack(applied_attention, dim=0)
        weights = torch.stack(weights, dim=0)
        visualization_data = {'context_vectors': context_vectors, 'applied_attention': applied_attention,
                              'attentional_hidden': output, 'weights': weights}

        return output, visualization_data


class QNet(nn.Module):
    """
    q-net, predicts q-values
    """
    def __init__(self, input_size, nr_actions):
        """

        :param input_size: size of input features
        :param nr_actions: number of actions
        """
        super(QNet, self).__init__()
        self.fully_connected_4 = nn.Linear(input_size, 512)
        self.fully_connected_5 = nn.Linear(512, nr_actions)

    def forward(self, state):
        """

        :param state:
        :return:
        """
        out = functional.relu(self.fully_connected_4(state))
        q_values = self.fully_connected_5(out)
        return q_values


################################################################################
#                                   Models                                     #
################################################################################


class DQNModel(nn.Module):
    """

    """
    def __init__(self, in_channels, input_size, nr_actions, kernel_size, stride):
        """

        :param in_channels:
        :param input_size:
        :param nr_actions:
        :param kernel_size:
        :param stride:
        """
        super(DQNModel, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=stride)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=kernel_size / 2, stride=stride / 2)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=kernel_size / 4, stride=stride / 4)
        self.fc_1 = nn.Linear(input_size, 512)
        self.fc_2 = nn.Linear(512, nr_actions)

    def forward(self, state):
        """

        :param state:
        :return:
        """
        out = functional.relu(self.conv_1(state))
        out = functional.relu(self.conv_2(out))
        out = functional.relu(self.conv_3(out))
        out = functional.relu(self.fc_1(out.view(1, -1)))
        out = self.fc_2(out)
        return out


class EADModel(nn.Module):
    """
    Model 2 according to architecture sketches
    """
    def __init__(self, config, nr_actions, device):
        """

        :param config:
        :param nr_actions:
        """
        super(EADModel, self).__init__()
        self.conv_net = CNN(config['in_channels'], device)

        self.encoder = Encoder(input_size=config['input_size_enc'],
                               hidden_size=config['hidden_size_enc'],
                               nr_layers=config['nr_layers_enc'], device=device)

        self.decoder = Decoder(input_size=config['input_size_dec'],
                               hidden_size=config['hidden_size_dec'],
                               nr_layers=config['nr_layers_dec'],
                               alignment=config['alignment_function'])

        self.q_net = QNet(input_size=config['input_size_q'],
                          nr_actions=nr_actions)

        self.config = config
        self.vector_combination = config['vector_combination']

        self.q_prediction = config['q_prediction'] if 'q_prediction' in config else None

        if self.vector_combination == 'layer':
            self.concat_layer = nn.Linear(config['hidden_size_enc'] * config['input_length'], config['hidden_size_enc'])

    def forward(self, state_sequence):
        """

        :param state_sequence:
        :return:
        """
        conv_in = torch.stack(list(state_sequence), dim=0)
        conv_out = self.conv_net.forward(conv_in)
        input_sequence = self.build_vector(conv_out)
        input_sequence.unsqueeze_(dim=1)
        encoder_out, (encoder_h_n, encoder_c_n) = self.encoder.forward(input_sequence)
        decoder_out, visualization_data = self.decoder.forward(input_sequence, encoder_out, encoder_h_n, encoder_c_n)
        q_in = decoder_out.squeeze(dim=1)
        if self.vector_combination == 'concat':
            q_in = torch.transpose(q_in, dim0=0, dim1=1)
            q_in = q_in.reshape(self.config['input_length'], self.config['max_filters'], self.config['cnn_out'] ** 2)
            q_in = q_in.reshape(self.config['input_length'], self.config['max_filters'], self.config['cnn_out'],
                                self.config['cnn_out'])
            if self.q_prediction == 'last':
                q_values = self.q_net.forward(q_in[-1].reshape(1, -1))
            else:
                q_values = self.q_net.forward(q_in.reshape(self.config['input_length'], -1))
        else:
            q_values = self.q_net.forward(q_in.reshape(1, -1))
        return q_values, visualization_data

    def build_vector(self, conv_out):
        """

        :param conv_out:
        :return:
        """
        batch, _, height, width = conv_out.shape
        vectors = [torch.stack([conv_out[i][:, y, x] for y in range(height) for x in range(width)], dim=0)
                   for i in range(batch)]
        if self.vector_combination == 'mean':
            length = len(vectors)
            return reduce(lambda t1, t2: t1 + t2, vectors) / length
        elif self.vector_combination == 'sum':
            return reduce(lambda t1, t2: t1 + t2, vectors)
        elif self.vector_combination == 'concat':
            return reduce(lambda t1, t2: torch.cat((t1, t2), dim=-1), vectors)
        else:
            return self.concat_layer(reduce(lambda t1, t2: torch.cat((t1, t2), dim=-1), vectors))
