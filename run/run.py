import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
import torch
import gym
import datetime
import copy
import matplotlib.pyplot as plt
from collections import deque

from agents import agents as agent
from utils import args as a
from utils import config as c
from utils import transformation

args = a.parse()
config = c.make_config(args)
print(config)
env = gym.make(config['environment'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = transformation.Transformation(config)

total_steps = 0
state_seq = deque(maxlen=config['input_length'])
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
    ead_agent = agent.EADAgent(config, env.action_space.n, device)

    for episode in range(config['total_episodes']):
        state_seq.clear()
        for _ in range(config['input_length']):
            state_seq.append(t.transform(env.reset()))
        next_state_seq = copy.deepcopy(state_seq)
        discounted_return = 0
        steps = 0
        done = False
        print(f'Episode {episode + 1} of {config["total_episodes"]}')
        start = datetime.datetime.utcnow()

        while not done:
            steps += 1
            total_steps += 1
            action = ead_agent.policy(state_seq)
            next_state, reward, done, _ = env.step(action)
            next_state = t.transform(next_state)
            next_state_seq.append(next_state)
            discounted_return += reward * (config['gamma'] ** steps)
            ead_agent.append_sample(copy.deepcopy(state_seq), action, reward, copy.deepcopy(next_state_seq), done)
            state = next_state
            if config['mode'] == 'train':
                if total_steps > config['train_start']:
                    loss = ead_agent.train()
                    q_loss.append(loss)
                    ead_agent.minimize_epsilon()
            if done:
                end = datetime.datetime.utcnow()
                print(f'done after {steps} steps, duration: {end-start}')
                discounted_returns.append(discounted_return)
                break
    plot(lstm_loss, 'encoder_decoder_loss')
    plot(q_loss, 'q_loss')
    plot(discounted_returns, 'discounted_returns')
