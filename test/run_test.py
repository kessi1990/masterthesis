import sys
sys.path.append('..')
import torch
import gym
import datetime
import copy
import gc
import random
from collections import deque

from agents import agents
from utils import args as a
from utils import config as c
from utils import transformation
from utils import fileio
from utils import shapes

args = a.parse()
config = c.make_config(args)
# directory = fileio.make_dir(config)
# config = {**config, 'sub_dir': directory}
config = shapes.compute_sizes(config)
# fileio.write_config(config, directory)
# fileio.write_info(config, directory)
env = gym.make('Seaquest-v0')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
t = transformation.Transformation(config)

total_steps = 0
state_seq = deque(maxlen=config['input_length'])
q_loss = []

config['mode'] = 'eval'

if __name__ == '__main__':
    ead_agent = agents.EADAgent(config, env.action_space.n, device)

    for episode in range(config['total_episodes']):
        q_loss.clear()
        state_seq.clear()
        for _ in range(config['input_length']):
            state_seq.append(t.transform(env.reset()))
        next_state_seq = copy.deepcopy(state_seq)
        discounted_return = 0
        steps = 0
        done = False
        print('=============================')
        print(f'Episode {episode + 1} of {config["total_episodes"]}')
        start = datetime.datetime.utcnow()
        env.render()

        while not done:
            env.render()
            steps += 1
            # print(f'steps {steps}')
            total_steps += 1
            action = ead_agent.policy(state_seq)
            next_state, reward, done, info = env.step(random.choice(range(env.action_space.n)))  # env.step(action)
            print(reward)
            next_state_seq.append(t.transform(next_state))
            discounted_return += reward * (config['gamma'] ** steps)
            ead_agent.append_sample(copy.deepcopy(state_seq), action, reward, copy.deepcopy(next_state_seq), done)
            state_seq = next_state_seq
            if config['mode'] == 'train':
                if total_steps > config['train_start']:
                    loss = ead_agent.train()
                    q_loss.append(loss.item())
                    ead_agent.minimize_epsilon()
                    done = True
            if done:
                end = datetime.datetime.utcnow()
                print(f'done after {steps} steps, duration: {end-start}')
                print(f'avg loss: {sum(q_loss) / len(q_loss) if len(q_loss) > 0 else 0}')
                result = {str(episode): {'steps': steps, 'loss': q_loss,
                                         'avg_loss': sum(q_loss) / len(q_loss) if len(q_loss) > 0 else 0,
                                         'discounted_return': discounted_return}}
                # fileio.save_results(result, directory)
                if episode % 10 == 0 or True:
                    pass
                    # fileio.save_model(ead_agent.policy_net, ead_agent.target_net, directory)
                gc.collect()
                break

    # io.save_model(ead_agent.policy_net, ead_agent.target_net, directory)
