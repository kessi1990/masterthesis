import argparse


def parse():
    parser = argparse.ArgumentParser(prog='Attention-driven learning of Temporal Abstractions in '
                                          'Reinforcement Learning', description='Parse arguments for run script')

    parser.add_argument('-m', '--model',
                        choices=['darqn', 'cead'],
                        help='model type. defines if darqn or cead model is used.')

    parser.add_argument('-a', '--alignment',
                        choices=['concat', 'general', 'dot'],
                        help='alignment method. defines which alignment method is used for computing attention weights.')

    parser.add_argument('-e', '--environment',
                        help='environment. defines which environment is used for training. \n'
                             'note: environments must start with a capital letter, e.g. -e Pong')

    parser.add_argument('-o', '--output',
                        help='output ID. running the script for the first time generates an output directory in the '
                             'root directory of this project. the -a argument defines the ID of the run, which is'
                             'mandatory if the training is interrupted and later continued for some reason.')

    args = parser.parse_args()

    if args.model:
        config = {'model': args.model}
    else:
        config = {'model': 'cead'}

    if args.alignment:
        config = {**config, 'alignment': args.alignment}
    else:
        config = {**config, 'alignment': 'concat'}

    if args.environment:
        config = {**config, 'environment': args.environment}
    else:
        config = {**config, 'environment': 'Pong'}

    if args.output:
        config = {**config, 'output': args.output}
    else:
        config = {**config, 'output': 0}

    return config
