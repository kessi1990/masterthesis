import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
import torch
import gym
import datetime
import copy
import matplotlib.pyplot as plt
from itertools import chain
from collections import deque

from agents import agent_conv_first
from agents import agent_lstm_first
from utils import args as a
from utils import config as c

args = a.parse()
config = c.make_config(args)
print(config)
env = gym.make(config['environment'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_steps = 0
input_sequence = deque(maxlen=3)
lstm_loss = []
q_loss = []
discounted_returns = []


# used for plotting loss and discounted return during training. very basic --> TODO: plotting module
def plot(data, name):
    f = plt.figure()
    plt.plot(list(data))
    plt.xlabel('Training Steps')
    plt.ylabel(f'{name}')
    f.savefig(f'{name}.png')


if __name__ == '__main__':
    if config['head'] == 'cnn':
        agent = agent_conv_first.Agent(config, env.action_space.n, device)
    else:
        agent = agent_lstm_first.Agent(config, env.action_space.n, device)

    for episode in range(config['total_episodes']):
        input_sequence.clear()
        for _ in range(3):
            input_sequence.append(env.reset())
        discounted_return = 0
        steps = 0
        done = False
        print(f'Episode {episode + 1} of {config["total_episodes"]}')
        start = datetime.datetime.utcnow()

        while not done:
            steps += 1
            total_steps += 1
            action = agent.policy(input_sequence)
            next_state, reward, done, _ = env.step(action)
            discounted_return += reward * (config['gamma'] ** steps)
            agent.append_sample(copy.deepcopy(input_sequence), action, reward, copy.deepcopy(next_state), done)
            state = next_state
            if config['mode'] == 'train':
                if total_steps > config['train_start']:
                    # loss_1, loss_2 = agent.train()
                    # lstm_loss = chain(lstm_loss, loss_1)
                    loss_2 = agent.train()
                    q_loss = chain(q_loss, loss_2)
                    agent.minimize_epsilon()
            if done:
                end = datetime.datetime.utcnow()
                print(f'done after {steps} steps, duration: {end-start}')
                discounted_returns.append(discounted_return)
                break
    plot(lstm_loss, 'encoder_decoder_loss')
    plot(q_loss, 'q_loss')
    plot(discounted_returns, 'discounted_returns')
