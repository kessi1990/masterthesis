import torch
import torch.nn as nn
import torch.nn.functional as functional
from functools import reduce


class ConvNet(nn.Module):
    """
    Convolutional Network
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
        # print(f'CNN FORWARD: input_sequence shape {input_sequence.shape}')
        out = functional.relu(self.conv_1(input_sequence))
        out = functional.relu(self.conv_2(out))
        out = functional.relu((self.conv_3(out)))
        # print(f'CNN FORWARD: out shape {out.shape}')
        return out


class Encoder(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, nr_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, nr_layers)

    def forward(self, input_sequence):
        """

        :param input_sequence:
        :return:
        """
        # print(f'ENCODER FORWARD: input_sequence shape {input_sequence.shape}')
        output_sequence, (h_n, c_n) = self.lstm(input_sequence)
        # print(f'ENCODER FORWARD: output_sequence shape {output_sequence.shape}')
        # print(f'ENCODER FORWARD: h_n shape {h_n.shape}')
        # print(f'ENCODER FORWARD: c_n shape {c_n.shape}')
        return output_sequence, (h_n, c_n)


class Attention(nn.Module):
    """
    Attention layer, takes LSTM output from encoder as input and outputs attention weights and matrix
    """

    def __init__(self, hidden_size, alignment_mechanism):
        super(Attention, self).__init__()
        self.alignment_mechanism = alignment_mechanism

        if self.alignment_mechanism == 'dot':
            # dot does not use a weighted matrix
            # dot computes alignment score according to h_t^T \bar{h_s}
            return
        elif self.alignment_mechanism == 'location' or 'general':
            # location_based computes alignment score according to a_t = softmax(W_a, h_t)
            # general computes alignment score according to score(h_t, \bar{h_s}) = h_t^T W_a \bar{h_s}
            self.alignment_function = nn.Linear(hidden_size, hidden_size, bias=False)
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
            # print(f'ATTENTION FORWARD: encoder_out {encoder_out.shape}')
            # print(f'ATTENTION FORWARD: decoder_hidden {decoder_hidden.shape}')
            # print('ATTENTION FORWARD: squeeze encoder_out and decoder_hidden')
            source_states = encoder_out.squeeze()
            target_state = decoder_hidden[-1].squeeze()
            # print(f'ATTENTION FORWARD: encoder_out {encoder_out.shape}')
            # print(f'ATTENTION FORWARD: decoder_hidden {decoder_hidden.shape}')
            scalars = []
            for source_state in source_states:
                # print(f'source_state shape {source_state.shape}')
                # print(f'target_state shape {target_state.shape}')
                scalar = torch.dot(source_state, target_state)
                # print(f'scalar shape {scalar.shape}')
                # print(f'scalar {scalar}')
                # # print(f'scalar {scalar}')
                scalars.append(scalar)
            return torch.tensor(scalars)

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
    def __init__(self, input_size, hidden_size, nr_layers, output_size):
        """

        :param input_size:
        :param hidden_size:
        :param nr_layers:
        """
        super(Decoder, self).__init__()
        # TODO alignment from config
        self.attention = Attention(hidden_size=hidden_size, alignment_mechanism='dot')
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=nr_layers)
        self.concat_layer = nn.Linear(input_size*2, input_size)
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, input_sequence, encoder_out, hidden_state, cell_state):
        """

        :param input_sequence:
        :param encoder_out:
        :param hidden_state:
        :param cell_state:
        :return:
        """
        # print(f'DECODER FORWARD: hidden_state {hidden_state.shape}')
        # print(f'DECODER FORWARD: cell_state {cell_state.shape}')
        # print('DECODER FORWARD: unsqueeeze both')
        # hidden_state.unsqueeze_(dim=0)
        # cell_state.unsqueeze_(dim=0)
        # print(f'DECODER FORWARD: hidden_state {hidden_state.shape}')
        # print(f'DECODER FORWARD: cell_state {cell_state.shape}')
        # print('DECODER FORWARD: feed through decoder lstm')
        # alignment score
        decoder_out, (decoder_hidden_s, decoder_hidden_c) = self.lstm(input_sequence, (hidden_state, cell_state))
        # print(f'DECODER FORWARD: decoder_hidden_s {decoder_hidden_s.shape}')
        # print(f'DECODER FORWARD: decoder_hidden_c {decoder_hidden_c.shape}')
        # print('DECODER FORWARD: feed through attention layer')
        alignment_score = self.attention.forward(encoder_out, decoder_hidden_s)
        # print(f'DECODER FORWARD: alignment_score shape {alignment_score.shape}')
        # print(f'DECODER FORWARD: alignment_score {alignment_score}')
        # print(f'DECODER FORWARD: decoder_hidden_c {decoder_hidden_c.shape}')
        # softmax alignment score to obtain attention weights
        attention_weights = functional.softmax(alignment_score, dim=0)
        # print(f'DECODER FORWARD: attention_weights shape {attention_weights.shape}')
        # print(f'DECODER FORWARD: attention_weights {attention_weights}')
        # multiply attention with encoder output
        context = []
        for i, source_state in enumerate(encoder_out):
            # print(f'source_state shape {source_state.shape}')
            # print(f'attention_weights shape {attention_weights.shape}')
            context.append(torch.mul(source_state, attention_weights[i]).unsqueeze_(dim=0))
        context = reduce(lambda t1, t2: torch.add(t1, t2), context)
        # print(f'DECODER FORWARD: context shape {context.shape}')
        # print(f'DECODER FORWARD: decoder hidden shape {decoder_hidden_s.shape}')
        foo = torch.cat((context, decoder_hidden_s[-1].unsqueeze_(dim=0)), dim=-1)
        # print(f'foo shape {foo.shape}')
        attentional_hidden = torch.tanh(self.concat_layer(foo[-1]))
        # print(f'attentional_hidden shape {attentional_hidden.shape}')
        # print(f'DECODER FORWARD: attentional_hidden shape {attentional_hidden.shape}')
        # print(f'DECODER FORWARD: attentional_hidden values {attentional_hidden}')
        # output = functional.softmax(output, dim=-1)
        # predictive_dist = functional.softmax(self.classifier(attentional_hidden.squeeze()))
        # print(f'DECODER FORWARD: predictive_dist shape {predictive_dist.shape}')
        # print(f'DECODER FORWARD: predictive_dist values {predictive_dist}')
        return decoder_out, (decoder_hidden_s, decoder_hidden_c), attentional_hidden


class QNet(nn.Module):
    """

    """
    def __init__(self, input_size, nr_actions):
        """

        :param nr_actions:
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
