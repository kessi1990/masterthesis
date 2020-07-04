import yaml


def make_config(args):
    if args['default']:
        conf_append = load_config_file('default')
        args = {**args, **conf_append}
    with open(f'../config/{args["environment"]}.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = {**config, **args}
    return config


def load_config_file(file):
    with open(f'../config/{file}_lstm_first.yaml', 'r') as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)
