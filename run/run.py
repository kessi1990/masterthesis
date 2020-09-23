import sys
sys.path.append('..')

import torch
import gym
import datetime
import copy
import gc

from collections import deque

from agents import agents
from utils import config as c
from utils import fileio
from utils import transformation
from utils import plotter


model_type = sys.argv[1]
env_type = sys.argv[2]
num_layers = int(sys.argv[3])

config = c.load_config_file(f'../config/default_new.yaml')
directory = fileio.mkdir(model_type, env_type, num_layers)
env = gym.make(env_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
t = transformation.Transformation(config)

training_steps = 1000000  # 5000000
evaluation_start = 10000  # 50000
evaluation_steps = 5000   # 25000


def evaluate_model(model):
    returns_ = []
    return_ = 0
    steps = 0
    done = True
    init = True
    lives = None
    state_seq = deque(maxlen=4)
    next_state_seq = deque(maxlen=4)

    while steps < evaluation_steps:
        if done:
            if not init:
                returns_.append(return_)
            init = False
            state_seq = deque(maxlen=4)
            env.reset()
            state, reward, done, info = env.step(1)  # press fire to start breakout
            lives = info['ale.lives']
            state_seq.append(t.transform(state))
            next_state_seq = copy.deepcopy(state_seq)
            return_ = 0
            model.policy_net.init_hidden()
            model.target_net.init_hidden()
        with torch.no_grad():
            action = model.policy(state_seq, mode='evaluation')
        next_state, reward, done, info = env.step(action)

        return_ += reward
        steps += 1
        next_state_seq.append(t.transform(next_state))
        state_seq = copy.deepcopy(next_state_seq)

        if lives != info['ale.lives'] and not done:
            state, reward, done, info = env.step(1)  # press fire to start breakout
            lives = info['ale.lives']
            return_ += reward
            state_seq.append(t.transform(state))
            next_state_seq = copy.deepcopy(state_seq)
            model.policy_net.init_hidden()
            model.target_net.init_hidden()

    return sum(returns_) / len(returns_) if len(returns_) > 0 else 0


if __name__ == '__main__':
    agent = agents.DQN(model_type, env.action_space.n, device, num_layers)

    evaluation_returns = []
    training_returns = []
    losses = []
    epsilons = []
    results = {}

    train_counter = 0
    return_ = 0
    done = True
    init = True
    lives = 0
    state_seq = deque(maxlen=4)

    print('=====================================================')
    print(f'model: {model_type}')
    print('-----------------------------------------------------')
    print(f'epsilon_decay: {agent.epsilon_decay}')
    print(f'epsilon_min: {agent.epsilon_min}')
    print(f'discount_factor: {agent.discount_factor}')
    print(f'batch_size: {agent.batch_size}')
    print(f'memory-size: {agent.memory.maxlen}')
    print(f'k_target: {agent.k_target}')
    print('=====================================================')
    print('training model ...')
    start = datetime.datetime.now()
    train_start = datetime.datetime.now()

    for step in range(1, training_steps + 1):
        epsilons.append(agent.epsilon)
        if done:
            agent.policy_net.init_hidden()
            agent.target_net.init_hidden()
            if not init:
                training_returns.append(return_)
            init = False
            env.reset()
            state, reward, done, info = env.step(1)  # press fire to start breakout
            lives = info['ale.lives']
            state_seq.clear()
            state_seq.append(t.transform(state))
            next_state_seq = copy.deepcopy(state_seq)
            return_ = 0
        with torch.no_grad():
            action = agent.policy(state_seq, mode='training')
        next_state, reward, done, info = env.step(action)
        return_ += reward
        next_state_seq.append(t.transform(next_state))
        agent.append_sample(copy.deepcopy(state_seq), action, reward, copy.deepcopy(next_state_seq), done)
        state_seq = copy.deepcopy(next_state_seq)
        agent.minimize_epsilon()
        if lives != info['ale.lives'] and not done:
            state, reward, done, info = env.step(1)  # press fire to start breakout
            lives = info['ale.lives']
            return_ += reward
            state_seq.append(t.transform(state))
            next_state_seq = copy.deepcopy(state_seq)
            agent.policy_net.init_hidden()
            agent.target_net.init_hidden()
        if step % 4 == 0:
            loss = agent.train()
            losses.append(loss.item())
            train_counter += 1
        if step % evaluation_start == 0 or step == training_steps:
            train_end = datetime.datetime.now()
            print(f'training done! time: {train_end - train_start}')
            print(f'training: {train_counter} / {int(training_steps / 4)}, steps: {step} / {training_steps}')
            print('-----------------------------------------------------')
            print('evaluating model ...')
            start_time = datetime.datetime.now()
            avg_return = evaluate_model(agent)
            end_time = datetime.datetime.now()
            print(f'evaluation done! time: {end_time - start_time}')
            print('-----------------------------------------------------')
            print(f'saving model ...')
            fileio.save_model(agent.policy_net, agent.target_net, agent.optimizer, directory)
            print(f'model saved!')
            print('-----------------------------------------------------')
            evaluation_returns.append(avg_return)
            print(f'plotting intermediate results ...')
            plotter.plot_intermediate_results(directory, **results)
            print(f'plotting done!')
            print('-----------------------------------------------------')
            print(f'saving intermediate results ...')
            results = {'loss': losses, 'evaluation_returns': evaluation_returns, 'training_returns': training_returns,
                       'epsilons': epsilons}
            fileio.save_results(results, directory)
            print('=====================================================')
            print('continue training ...')
            done = True
            gc.collect()
            train_start = datetime.datetime.now()
    end = datetime.datetime.now()
    print(f'overall time: {end - start}')
    results = {'loss': losses, 'evaluation_returns': evaluation_returns, 'training_returns': training_returns,
               'epsilons': epsilons}
    print('=====================================================')
    print('=====================================================')
    print(f'saving results and final model ...')
    fileio.save_results(results, directory)
    fileio.save_model(agent.policy_net, agent.target_net, agent.optimizer, directory)
    print(f'... done!')
    print('-----------------------------------------------------')
    print(f'plotting final results ...')
    plotter.plot_intermediate_results(directory, **results)
    print(f'plotting done!')
    print('=====================================================')
