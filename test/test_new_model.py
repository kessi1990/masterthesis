import sys
sys.path.append('..')
import torch
import gym
import datetime
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

from utils import config as c
from utils import fileio
from utils import transformation
from models import models_v2


config = c.load_config_file(f'../config/{sys.argv[1]}.yaml')
directory = fileio.mkdir(sys.argv[2])
env = gym.make(sys.argv[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
t = transformation.Transformation(config)

training_steps = 300  # 5000000
evaluation_start = 100  # 50000
evaluation_steps = 500  # 5000  # 25000


def evaluate_model(model):
    returns_ = []
    return_ = 0
    steps = 0
    done = True
    init = True
    while steps < evaluation_steps:
        if done:
            if not init:
                returns_.append(return_)
            init = False
            state_seq = deque(maxlen=4)
            for _ in range(4):
                state_seq.append(t.transform(env.reset()))
            next_state_seq = copy.deepcopy(state_seq)
            return_ = 0
        with torch.no_grad():
            action = model.evaluation_policy(state_seq)
        if action == 1:
            print('FIREEEEEE')
        next_state, reward, done, info = env.step(action)
        return_ += reward
        steps += 1
        next_state_seq.append(t.transform(next_state))
        model.append_sample(copy.deepcopy(state_seq), action, reward, copy.deepcopy(next_state_seq), done)
        state_seq = next_state_seq

    return sum(returns_) / len(returns_) if len(returns_) > 0 else 0


def plot_intermediate_results(losses, avg_returns):
    plt.plot(losses)
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.savefig(directory + 'loss.png')
    plt.close()
    plt.plot(avg_returns)
    plt.xlabel('epoch')
    plt.ylabel('avg return')
    plt.savefig(directory + 'avg_return.png')
    plt.close()


if __name__ == '__main__':
    agent = models_v2.CEADNAgent(env.action_space.n, device)
    start = datetime.datetime.now()
    train_start = datetime.datetime.now()
    avg_returns = []
    losses = []
    print('====================================')
    print('training model ...')
    for i in range(training_steps):
        loss = agent.train()
        losses.append(loss.item())
        if i % evaluation_start == 0:
            train_end = datetime.datetime.now()
            print(f'training done for {i} steps! time: {train_end - train_start}')
            print('-------------------------------------')
            print('evaluating model ...')
            start_time = datetime.datetime.now()
            avg_return = evaluate_model(agent)
            end_time = datetime.datetime.now()
            print(f'evaluation done! time: {end_time - start_time}')
            print('-------------------------------------')
            print(f'saving model ...')
            fileio.save_model(agent.policy_net, agent.target_net, agent.optimizer, directory)
            print(f'model saved!')
            print('-------------------------------------')
            avg_returns.append(avg_return)
            print(f'plotting intermediate results ...')
            plot_intermediate_results(losses, avg_returns)
            print(f'plotting done!')
            print('-------------------------------------')
            print(f'saving intermediate results ...')
            results = {'loss': losses, 'avg_return': avg_returns}
            fileio.save_results(results, directory)
            print('====================================')
            print('continue training ...')
            train_start = datetime.datetime.now()
    end = datetime.datetime.now()
    print(f'overall time: {end - start}')
    results = {'loss': losses, 'avg_return': avg_returns}
    print('====================================')
    print(f'saving results and final model ...')
    fileio.save_results(results, directory)
    fileio.save_model(agent.policy_net, agent.target_net, agent.optimizer, directory)
    print(f'... done!')
    print('====================================')
