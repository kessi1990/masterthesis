import os
import yaml
import json
import torch
from torchvision import transforms
from datetime import datetime


def make_dir(config):
    if os.path.exists(config['output']):
        print(f'output directory {config["output"]} already exists')
    else:
        os.mkdir(config['output'], 0o755)
    while True:
        timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d__%H-%M-%S__')
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
        for k, v in config.items():
            if k in ('attention_mechanism', 'alignment_function', 'vector_combination', 'q_shape', 'q_prediction'):
                file.write(f'{k}: {v}\n')


def save_model(model_p, model_t, path):
    torch.save(model_p.state_dict(), path + 'model_policy.pt')
    torch.save(model_t.state_dict(), path + 'model_target.pt')


def save_results(data, directory):
    if os.path.exists(directory + 'results.json'):
        results = load_results(directory)
        results.update(data)
    else:
        results = data
    with open(directory + 'results.json', 'w') as file:
        json.dump(results, file)


def load_results(directory):
    with open(directory + 'results.json', 'r') as file:
        results = json.load(file)
        return results


def save_image(tensor, path):
    image = transforms.ToPILImage()(tensor)
    image.save(path)
