import yaml


def make_config(args):
    if args['default']:
        if args['head'] == 'cnn':
            conf_append = load_config_file('../config/default.yaml')
        else:
            conf_append = load_config_file('default_lstm_first')
        args = {**args, **conf_append}
    with open(f'../config/{args["environment"]}.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = {**config, **args}
    return config


def load_config_file(file):
    with open(file, 'r') as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


def gen_config_files(config):
    config_list = []
    alignment_functions = ['dot', 'general', 'concat']
    attention_mechanism = ['soft']
    vector_combination = ['mean', 'sum', 'concat', 'layer']
    q_prediction = ['last', 'all']
    q_shapes = ['original', None]
    for af in alignment_functions:
        for am in attention_mechanism:
            for vc in vector_combination:
                for qs in q_shapes:
                    if vc == 'concat':
                        for qp in q_prediction:
                            config_list.append({**config, 'alignment_function': af, 'attention_mechanism': am,
                                                'vector_combination': vc, 'q_prediction': qp, 'q_shape': qs})
                    else:
                        config_list.append({**config, 'alignment_function': af, 'attention_mechanism': am,
                                            'vector_combination': vc, 'q_shape': qs})
    return config_list
