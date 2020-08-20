import sys
sys.path.append('..')
import torch
import gym
import datetime
import copy
import gc
from collections import deque

from agents import agents as agent
from utils import args as a
from utils import config as c
from utils import transformation
from utils import fileio
from utils import shapes

args = a.parse()
config = c.make_config(args)
directory = fileio.make_dir(config)
config = {**config, 'sub_dir': directory}
config = shapes.compute_sizes(config)
fileio.write_config(config, directory)
fileio.write_info(config, directory)
env = gym.make(config['environment'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
t = transformation.Transformation(config)

total_steps = 0
state_seq = deque(maxlen=config['input_length'])
q_loss = []

if __name__ == '__main__':
    ead_agent = agent.EADAgent(config, env.action_space.n, device)
    ead_agent.policy_net.eval()

    for episode in range(config['total_episodes']):
        q_loss.clear()
        state_seq.clear()
        env.reset()
        discounted_return = 0
        acc_return = 0
        steps = 0
        state, reward, done, info = env.step(1)  # press fire to start breakout
        start_game = False
        for _ in range(config['input_length']):
            state_seq.append(t.transform(state))
        next_state_seq = copy.deepcopy(state_seq)
        lives = info['ale.lives']
        print('=============================')
        print(f'Episode {episode + 1} of {config["total_episodes"]}')
        start = datetime.datetime.utcnow()

        while not done:
            if start_game:
                _, _, _, info = env.step(1)
                lives = info['ale.lives']
                start_game = False
            steps += 1
            total_steps += 1
            action = ead_agent.policy(state_seq)
            next_state, reward, done, info = env.step(action)
            if lives != info['ale.lives']:
                start_game = True
                lives = info['ale.lives']
            next_state_seq.append(t.transform(next_state))
            discounted_return += reward * (config['gamma'] ** steps)
            acc_return += reward
            ead_agent.append_sample(copy.deepcopy(state_seq), action, reward, copy.deepcopy(next_state_seq), done)
            state_seq = copy.deepcopy(next_state_seq)  # deepcopy since state_seq and next_state_seq have type deque
            if config['mode'] == 'train':
                if total_steps > config['train_start']:
                    loss = ead_agent.train()
                    q_loss.append(loss.item())
            if done:
                end = datetime.datetime.utcnow()
                print(f'done after {steps} steps, duration: {end-start}')
                print(f'avg loss: {sum(q_loss) / len(q_loss) if len(q_loss) > 0 else 0}')
                result = {str(episode): {'steps': steps, 'loss': q_loss,
                                         'avg_loss': sum(q_loss) / len(q_loss) if len(q_loss) > 0 else 0,
                                         'discounted_return': discounted_return,
                                         'acc_return': acc_return}}
                fileio.save_results(result, directory)
                if episode % 10 == 0:
                    fileio.save_model(ead_agent.policy_net, ead_agent.target_net, ead_agent.optimizer, directory)
                gc.collect()
                break

    fileio.save_model(ead_agent.policy_net, ead_agent.target_net, ead_agent.optimizer, directory)
