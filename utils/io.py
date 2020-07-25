import os
import yaml
import json
import torch


def make_dir(output, test=False):
    if os.path.exists(output):
        print(f'output directory {output} already exists')
    else:
        os.mkdir(output, 777)
    if not test:
        path = output + 'ID_'
        i = 0
        while True:
            if os.path.exists(path + f'{i:04d}'):
                i += 1
                continue
            else:
                path = path + f'{i:04d}'
                break
        os.mkdir(path, 777)
        return path + '/'
    else:
        return output + '/'


def write_config(config, directory):
    with open(directory + 'parameters.yaml', 'x') as file:
        yaml.dump(config, file)


def save_model(model, path):
    torch.save(model.state_dict(), path + 'model.pt')


def save_json(data, directory):
    with open(directory + 'results.json', 'w') as file:
        json.dump(data, file)
