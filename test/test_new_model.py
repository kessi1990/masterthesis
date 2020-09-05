import sys
sys.path.append('..')
import torch
import gym
import datetime
import copy
import gc
import random
from collections import deque

from utils import config as c
from utils import transformation
from models import models_v2


config = c.load_config_file('../config/Seaquest-v0.yaml')
env = gym.make('Seaquest-v0')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
t = transformation.Transformation(config)

training_steps = 5000000  # 5000000
evaluation_start = 50000  # 50000
evaluation_steps = 25000  # 25000


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
        next_state, reward, done, info = env.step(action)
        return_ += reward
        steps += 1
        next_state_seq.append(t.transform(next_state))
        model.append_sample(copy.deepcopy(state_seq), action, reward, copy.deepcopy(next_state_seq), done)
        state_seq = next_state_seq

    return sum(returns_) / len(returns_) if len(returns_) > 0 else 0


if __name__ == '__main__':
    agent = models_v2.DARQNAgent(env.action_space.n, device)
    start = datetime.datetime.now()
    evaluate = True
    for i in range(training_steps):
        agent.train()
        if i % evaluation_start == 0 and evaluate:
            evaluate = False
            print('evaluating model ...')
            start_time = datetime.datetime.now()
            avg_return = evaluate_model(agent)
            end_time = datetime.datetime.now()
            print(f'evaluation done! time: {end_time - start_time}')
    end = datetime.datetime.now()
    print(f'overall time: {end - start}')
