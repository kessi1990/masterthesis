import argparse
from utils import config as c


def parse():
    parser = argparse.ArgumentParser(prog='Attention-driven learning of Temporal Abstractions for '
                                          'Reinforcement Learning', description='Parse arguments for run script')
    parser.add_argument('--head',
                        help='specifies network architecture. --head \'cnn\'puts convolutional layers in '
                             'front, followed by encoder / decoder LSTM with attention mechanism. \n'
                             '--head \'lstm\' puts encoder / decoder LSTM with attention mechanism in '
                             'front, followed by convolutional layers. \n'
                             'if not provided, \'cnn\' is used')
    parser.add_argument('-c', '--config',
                        help='parse config file as \'*.yaml\'. \n'
                             'if not provided, default config is used.')
    parser.add_argument('-env', '--environment',
                        help='used for game selection. \n'
                             'if not provided, \'Breakout-v0\' is used.')
    parser.add_argument('-o', '--output',
                        help='output directory. ensure you have permissions to write to this directory! \n'
                             'if not provided, default-directory \'/output\' is used.')
    parser.add_argument('-m', '--mode',
                        choices=['train', 'eval'],
                        help='sets mode for run script. \'train\' is used for training mode, '
                             '\'eval\' is used for evaluation mode. if not provided, \'train\' is used.')

    args = parser.parse_args()

    if args.mode:
        config = {'mode': args.train}
    else:
        config = {'mode': 'train'}

    if args.config:
        config_provided = c.load_config_file(config['mode'])
        config = {**config, **config_provided, 'default': False}
    else:
        config = {**config, 'default': True}

    if args.environment:
        config = {**config, 'environment': args.environment}
    else:
        config = {**config, 'environment': 'Breakout-v0'}

    if args.output:
        config = {**config, 'output': args.output}
    else:
        output = '../output/'
        config = {**config, 'output': output}

    config = {**config, 'head': args.head if args.head else 'cnn'}

    return config
