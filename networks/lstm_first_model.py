import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvNet(nn.Module):
    """

    """
    def __init__(self, in_channels):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

    def forward(self, input_sequence):
        """

        :param input_sequence:
        :return:
        """
        out = functional.relu(self.conv_1(input_sequence))
        out = functional.relu(self.conv_2(out))
        out = functional.relu((self.conv_3(out)))
        return out


class Encoder(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, nr_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, nr_layers)

    def forward(self, sequence_input):
        """

        :param sequence_input:
        :return:
        """
        print(f'encoder forward: sequence_input shape {sequence_input.shape}')
        output_sequence, (h_n, c_n) = self.lstm(sequence_input)
        return output_sequence, (h_n, c_n)


class Attention(nn.Module):
    """
    Attention layer, takes LSTM output from encoder as input and outputs attention weights and matrix
    """

    def __init__(self, hidden_size, alignment_mechanism):
        super(Attention, self).__init__()
        self.alignment_mechanism = alignment_mechanism
        self.alignment_function = nn.Linear(hidden_size, hidden_size, bias=False)

        if self.alignment_mechanism == 'dot':
            # dot does not use a weighted matrix
            # dot computes alignment score according to h_t^T \bar{h_s}
            del self.alignment_function
        elif self.alignment_mechanism == 'location' or 'general':
            # location_based computes alignment score according to a_t = softmax(W_a, h_t)
            # general computes alignment score according to score(h_t, \bar{h_s}) = h_t^T W_a \bar{h_s}
            pass
        elif self.alignment_mechanism == 'concat':
            # TODO
            pass

    def forward(self, encoder_out, decoder_hidden):
        """

        :param encoder_out:
        :param decoder_hidden:
        :return:
        """
        if self.alignment_mechanism == 'dot':
            return torch.matmul(decoder_hidden, encoder_out)
        elif self.alignment_mechanism == 'general':
            aligned = self.alignment_function(decoder_hidden)
            return torch.matmul(encoder_out, aligned)
        elif self.alignment_mechanism == 'location':
            # location_based computes alignment score according to a_t = softmax(W_a, h_t)
            return functional.softmax(self.alignment_function(decoder_hidden), dim=0)
        elif self.alignment_mechanism == 'concat':
            pass


class Decoder(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, nr_layers):
        """

        :param input_size:
        :param hidden_size:
        :param nr_layers:
        """
        super(Decoder, self).__init__()
        self.attention = Attention(hidden_size=hidden_size, alignment_mechanism='location')
        self.lstm = nn.LSTM(input_size=input_size*2, hidden_size=hidden_size, num_layers=nr_layers)

    def forward(self, encoder_out, hidden_state, cell_state):
        """

        :param encoder_out:
        :param hidden_state:
        :param cell_state:
        :return:
        """
        """output_sequence, (h_n, c_n) = self.lstm(attention_applied, (encoder_h_n, encoder_c_n))"""
        # print (f'decoder output_sequence shape {output_sequence.shape}')
        # alignment score
        alignment_score = self.attention.forward(encoder_out, hidden_state)
        print(f'alignment_score shape {alignment_score.shape}')
        # encoder_out.squeeze_()
        print(f'encoder_out squeezed shape {encoder_out.shape}')
        # softmax alignment score to obtain attention weights
        attention_weights = functional.softmax(alignment_score.squeeze(), dim=0)
        print(f'attention_weights squeezed shape {attention_weights.shape}')

        # multiply attention with encoder output
        context = torch.mul(encoder_out.squeeze(), attention_weights)
        context.unsqueeze_(dim=1)
        print(f'context shape {context.shape}')
        # concat context with encoder_out
        decoder_in = torch.cat((context, encoder_out), dim=2)
        print(f'decoder_in shape {decoder_in.shape}')
        # feed into decoder
        output_sequence, (h_n, c_n) = self.lstm(decoder_in, (hidden_state.unsqueeze(dim=0), cell_state.unsqueeze(dim=0)))

        return output_sequence, (h_n, c_n), context


class QNet(nn.Module):
    """

    """
    def __init__(self, input_size, nr_actions):
        """

        :param nr_actions:
        """
        super(QNet, self).__init__()
        # print (f'input size fc {input_size}')
        self.fully_connected_4 = nn.Linear(input_size, 512)
        self.fully_connected_5 = nn.Linear(512, nr_actions)

    def forward(self, state):
        """

        :param state:
        :return:
        """
        out = functional.relu(self.fully_connected_4(state))
        q_values = self.fully_connected_5(out)
        # print (f'q_net q_values shape {q_values.shape}')
        return q_values
