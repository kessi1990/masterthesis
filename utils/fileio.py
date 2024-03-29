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


def mkdir(model, env, num_layers, hidden_size, alignment, dir_id=None):
    if dir_id is not None:
        path = f'../output/ID_{dir_id}/{model}_{env}_{num_layers}_{alignment}_{hidden_size}'
    else:
        path = f'../output_new/{model}_{env}_{num_layers}'
    if not os.path.exists(path):
        os.mkdir(path, 0o755)
    return path + '/'


def mkdir_g(model, env, dir_id, align):
    root = '../output/'
    id_path = f'../output/ID_{dir_id}'
    path = f'../output/ID_{dir_id}/{model}_{align}_{env}'
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(id_path):
        os.mkdir(id_path, 0o755)
    if not os.path.exists(path):
        os.mkdir(path, 0o755)
    return path + '/'


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


def save_checkpoint(agent, train_counter, steps, directory, best=False):
    data = {
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'learning_rate': agent.learning_rate,
        'learning_rate_decay': agent.learning_rate_decay,
        'learning_rate_min': agent.learning_rate_min,
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min,
        'discount_factor': agent.discount_factor,
        'batch_size': agent.batch_size,
        'memory_size': agent.memory.maxlen,
        'k_count': agent.k_count,
        'k_target': agent.k_target,
        'reward_clipping': agent.reward_clipping,
        'gradient_clipping': agent.gradient_clipping,
        'train_counter': train_counter,
        'continue': steps
    }
    path = directory + 'best_model.pt' if best else directory + 'checkpoint.pt'
    torch.save(data, path)


def load_checkpoint(directory):
    path = directory + 'checkpoint.pt'
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        return checkpoint
    else:
        return None


def save_results(data, directory):
    if os.path.exists(directory + 'results.json'):
        results = load_results(directory)
        results.update(data)
    else:
        results = data
    with open(directory + 'results.json', 'w') as file:
        json.dump(results, file)


def load_results(directory):
    if os.path.exists(directory + 'results.json'):
        with open(directory + 'results.json', 'r') as file:
            results = json.load(file)
            return results
    else:
        return {}


def save_history(directory, data):
    history = load_history(directory)
    if history is not None:
        history[list(history)[-1] + 1] = data
    else:
        history = {0: data}
    torch.save(history, directory + 'history.pt')


def load_history(directory):
    path = directory + 'history.pt'
    if os.path.exists(path):
        history = torch.load(path, map_location=torch.device('cpu'))
        return history
    else:
        return None


def save_image(tensor, path):
    image = transforms.ToPILImage()(tensor)
    image.save(path)
