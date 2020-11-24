import torch

from functools import reduce


def init_hidden(nr_layers, batch_size, hidden_size, device):
    """
    initializes first hidden state and cell state of LSTM
    :param nr_layers: number of stacked layers
    :param batch_size: batch size
    :param hidden_size: size of hidden features
    :param device: device on which returned tensor is stored (CPU / GPU)
    :return: initial hidden state and cell state tensor
    """
    return (torch.zeros(nr_layers, batch_size, hidden_size, device=device),
            torch.zeros(nr_layers, batch_size, hidden_size, device=device))


def cnn_out_size(size, filter_size, stride, padding=0, layers=3):
    """
    computes (recursively) output / feature size of CNN
    :param size: image size (quadratic -> height or width)
    :param filter_size: kernel / filter size
    :param stride: kernel stepping, moves n=stride steps further
    :param padding: padding zeros
    :param layers: number of convolutional layers
    :return: output size of current / last layer
    """
    if layers == 1:
        return int(((size - filter_size + (2 * padding)) / stride) + 1)
    else:
        size = int(((size - filter_size + (2 * padding)) / stride) + 1)
        return cnn_out_size(size, filter_size / 2, stride / 2, padding=padding, layers=(layers - 1))


def encoder_in_features(filters, size, combination, input_length=4):
    """
    computes number of features w.r.t previous applied vector combination method
    :param filters: number of features / filter maps from CNN
    :param size: (quadratic) size of single filter / feature map
    :param combination: vector combination method
    :param input_length: length of original input sequence (consecutive images to CNN)
    :return: number (=sequence length LSTM) and dimension (=features / hidden size) of vectors
    """
    if combination == 'concat':
        dim = filters * input_length
        nr_vectors = size * size
        return nr_vectors, dim
    else:
        dim = filters
        nr_vectors = size * size
        return nr_vectors, dim


def q_in_features(filters, nr_vectors, dim, config):
    """
    computes number of in features for fully connected layers / Q-Net
    :param filters: number of features / filter maps from CNN
    :param nr_vectors: sequence length LSTM
    :param dim: features / hidden size
    :param config: config file for missing parameters
    :return: batch size and number of features (flattened)
    """
    if 'q_prediction' in config:
        if config['q_prediction'] == 'last':
            shape = (1, int(dim / (dim / filters)), nr_vectors)
        else:
            shape = (dim / filters, filters, nr_vectors)
    else:
        shape = (dim / filters, filters, nr_vectors)
    batch, *features = shape
    features = reduce(lambda x, y: x * y, features)
    return batch, features


def compute_sizes(config):
    """
    computes all sizes, dimensions, hidden sizes, features and saves them to config for subsequent initialization
    of models
    :param config: config file, containing run specific parameter settings
    :return: extended config
    """
    size = config['crop_height']
    cnn_out = cnn_out_size(size, 8, 4)
    nr_vectors, dim = encoder_in_features(config['max_filters'], cnn_out, config['vector_combination'],
                                          config['input_length'])
    batch, features = q_in_features(config['max_filters'], nr_vectors, dim, config)

    config = {**config, 'input_size_enc': dim, 'hidden_size_enc': dim, 'input_size_dec': dim, 'hidden_size_dec': dim,
              'input_size_q': features, 'cnn_out': cnn_out}
    return config


def count_parameters(model):
    print(f'number of trainable parameters')
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        print(f'{name}: {param}')
        total_params += param
    print(f'total: {total_params}')
    print(f'number of all parameters')
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        print(f'{name}: {param}')
        total_params += param
    print(f'total: {total_params}')
