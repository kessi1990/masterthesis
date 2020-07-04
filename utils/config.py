import yaml


def make_config(args):
    if args['default']:
        if args['head'] == 'cnn':
            conf_append = load_config_file('default_cnn_first')
        else:
            conf_append = load_config_file('default_lstm_first')
        args = {**args, **conf_append}
    with open(f'../config/{args["environment"]}.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = {**config, **args}
    return config


def load_config_file(file):
    with open(f'../config/{file}.yaml', 'r') as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)
