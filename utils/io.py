import os
import yaml
import json
import torch
import time
import random
from datetime import datetime


def make_dir(config):
    if os.path.exists(config['output']):
        print(f'output directory {config["output"]} already exists')
    else:
        os.mkdir(config['output'], 0o755)
    while True:
        time.sleep(random.randint(1, 5))
        timestamp = datetime.strftime(datetime.utcnow(), '%Y-%m-%d__%H-%M-%S__')
        path = config['output'] + timestamp + config['id']
        if os.path.exists(path):
            continue
        else:
            os.mkdir(path, 0o755)
            break
    return path + '/'


def write_config(config, directory):
    with open(directory + 'parameters.yaml', 'x') as file:
        yaml.dump(config, file)


def write_info(config, directory):
    with open(directory + 'info.txt', 'x') as file:
        params = f'attention_mechanism={config["attention_mechanism"]}\n' \
                 f'alignment_function={config["alignment_function"]}\n' \
                 f'vector_combination={config["vector_combination"]}\n' \
                 f'q_prediction={config["q_prediction"]}\n' \
                 f'q_shape={config["q_shape"]}'
        file.write(params)


def save_model(model_p, model_t, path):
    torch.save(model_p.state_dict(), path + 'model_policy.pt')
    torch.save(model_t.state_dict(), path + 'model_target.pt')


def save_results(data, directory):
    with open(directory + 'results.json', 'w') as file:
        json.dump(data, file)
