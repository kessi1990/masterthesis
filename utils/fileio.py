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
        path = config['output'] + timestamp + 'ID-' + config['id']
        if os.path.exists(path):
            continue
        else:
            os.mkdir(path, 0o755)
            break
    return path + '/'


def mkdir(model, env, num_layers):
    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    path = f'../output_new/{timestamp}_{model}_{env}_{num_layers}--'
    i = 0
    while True:
        if os.path.exists(path + str(i)):
            i += 1
            continue
        else:
            os.mkdir(path + str(i), 0o755)
            break
    return path + str(i) + '/'


def visual_dir(root):
    sub_dir = 'visualization/'
    full_path = os.path.join(root, sub_dir)
    if os.path.exists(root + sub_dir):
        pass
    else:
        os.mkdir(root + sub_dir)
    if not os.listdir(root + sub_dir):
        dir_count = 0
    else:
        dir_count = max(sorted(list(filter(lambda x: x is not None,
                                           list(map(lambda x: int(x) if os.path.isdir(os.path.join(full_path, x))
                                                    else None, os.listdir(full_path))))))) + 1
    os.mkdir(os.path.join(full_path, str(dir_count)), 0o755)
    return os.path.join(full_path, str(dir_count)) + '/'


def write_config(config, directory):
    with open(directory + 'parameters.yaml', 'x') as file:
        yaml.dump(config, file)


def write_info(config, directory):
    with open(directory + 'info.txt', 'x') as file:
        for k, v in config.items():
            if k in ('attention_mechanism', 'alignment_function', 'vector_combination', 'q_shape', 'q_prediction'):
                file.write(f'{k}: {v}\n')


def save_model(model_p, model_t, optimizer, path):
    torch.save(model_p.state_dict(), path + 'model_policy.pt')
    torch.save(model_t.state_dict(), path + 'model_target.pt')
    torch.save(optimizer.state_dict(), path + 'optimizer.pt')


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
