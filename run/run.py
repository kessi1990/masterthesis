import sys
sys.path.append('..')

import torch
import gym
import datetime
import gc
import random

# pytorch lr_scheduler.state_dict() and .load_state_dict throw a warning when being invoked --> ignore
import warnings
warnings.filterwarnings("ignore")

from agents import agents as a
from utils import config as c
from utils import fileio
from utils import transformation
from utils import plots

import time

model_type = sys.argv[1]
env_type = sys.argv[2]
num_layers = int(sys.argv[3])
dir_id = int(sys.argv[4])
alignment = sys.argv[5]
hidden_size = int(sys.argv[6])

config = c.load_config_file(f'../config/{env_type}.yaml')
directory = fileio.mkdir(model_type, env_type, num_layers, alignment, hidden_size, dir_id=dir_id)
checkpoint = fileio.load_checkpoint(directory)
env = gym.make(env_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = transformation.Transformation(config)

print(f'dir_id: {dir_id}')
print(f'path: {directory}')
print(f'device: {device}')

training_steps = 2000000  # 1000000  # 5000000
evaluation_start = 50000  # 10000    # 50000
evaluation_steps = 25000  # 5000     # 25000


def evaluate_model(model):

    # set policy_net to eval mode
    model.policy_net.eval()

    # init hidden
    model.policy_net.init_hidden()
    model.target_net.init_hidden()

    eval_returns = []
    eval_return = 0
    steps = 0
    done = True
    init = True
    lives = None
    state = None

    while steps < evaluation_steps:

        # reset env
        if done:
            if not init:
                eval_returns.append(eval_return)
            init = False
            env.reset()

            #death detection
            # press fire (1) and continue
            state, reward, done, info = env.step(1)
            state = t.transform(state)
            lives = info['ale.lives']
            eval_return = 0

            # init hidden
            model.policy_net.init_hidden()
            model.target_net.init_hidden()

        # predict, no need for gradients
        with torch.no_grad():
            action = model.policy(state, mode='evaluation')

        next_state, reward, done, info = env.step(action)
        next_state = t.transform(next_state)
        state = next_state
        eval_return += reward
        steps += 1

        # death detection
        # press fire (1) and continue
        if lives != info['ale.lives'] and not done:
            state, reward, done, info = env.step(1)
            state = t.transform(state)
            lives = info['ale.lives']
            eval_return += reward
            model.policy_net.init_hidden()
            model.target_net.init_hidden()

    return (sum(eval_returns) / len(eval_returns), min(eval_returns), max(eval_returns)) if len(eval_returns) > 0 else (0, 0, 0)


def fill_memory_buffer(model):

    lives = None
    done = True
    state = None
    next_state = None
    action_space = [_ for _ in range(env.action_space.n)]

    while len(model.memory) < 20000:

        # reset env
        if done:
            env.reset()

            # death detection
            # press fire (1) and continue
            state, reward, done, info = env.step(1)
            state = t.transform(state)
            lives = info['ale.lives']

            # init hidden
            model.policy_net.init_hidden()
            model.target_net.init_hidden()

        # random action selection
        action = random.choice(action_space)

        # execute random action
        next_state, reward, done, info = env.step(action)
        next_state = t.transform(next_state)

        # fill buffer
        model.append_sample(state, action, reward, next_state, done)
        state = next_state

        # death detection
        # press fire (1) and continue
        if lives != info['ale.lives'] and not done:
            state, reward, done, info = env.step(1)
            state = t.transform(state)
            lives = info['ale.lives']

            # init hidden
            model.policy_net.init_hidden()
            model.target_net.init_hidden()


if __name__ == '__main__':
    agent = a.DQN(model_type, env.action_space.n, device, num_layers, hidden_size, alignment)

    evaluation_returns = []
    training_returns = []
    losses = []
    epsilons = []
    results = fileio.load_results(directory)
    if results:
        evaluation_returns = results['evaluation_returns']
        training_returns = results['training_returns']
        losses = results['loss']
        epsilons = results['epsilons']

    continue_steps = 0
    train_counter = 0
    training_return = 0
    done = True
    init = True
    lives = 0
    state = None
    next_state = None

    print('=====================================================')
    print(f'model: {model_type}')
    print(f'num_layers: {num_layers}')
    print(f'hidden_size: {hidden_size}')
    print(f'env_type: {env_type}')
    print('-----------------------------------------------------')

    if checkpoint:
        print(f'found checkpoint in directory:\n{directory}')
        print(f'loading checkingpoint ...')
        train_counter = checkpoint['train_counter']
        continue_steps = checkpoint['continue']
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.policy_net.to(device)
        agent.target_net.to(device)
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        if agent.lr_scheduler:
            agent.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        agent.learning_rate = checkpoint['learning_rate']
        agent.learning_rate_decay = checkpoint['learning_rate_decay']
        agent.learning_rate_min = checkpoint['learning_rate_min']
        agent.epsilon = checkpoint['epsilon']
        agent.epsilon_decay = checkpoint['epsilon_decay']
        agent.epsilon_min = checkpoint['epsilon_min']
        agent.discount_factor = checkpoint['discount_factor']
        agent.batch_size = checkpoint['batch_size']
        agent.k_count = checkpoint['k_count']
        agent.k_target = checkpoint['k_target']
        agent.reward_clipping = checkpoint['reward_clipping']
        agent.gradient_clipping = checkpoint['gradient_clipping']
        print(f'... done!')
        print(f'continue training at: {train_counter} / {int(training_steps / 4)}, steps: {continue_steps} / {training_steps}')
        print('-----------------------------------------------------')
        fill_start = datetime.datetime.now()
        print('filling memory buffer ...')
        fill_memory_buffer(agent)
        fill_end = datetime.datetime.now()
        print(f'... done! time: {fill_end - fill_start}')
        print(f'memory_size: {len(agent.memory)}')
        print('-----------------------------------------------------')

    print(f'epsilon: {agent.epsilon}')
    print(f'epsilon_decay: {agent.epsilon_decay}')
    print(f'epsilon_min: {agent.epsilon_min}')
    print(f'learning_rate_start: {agent.learning_rate}')
    print(f'learning_rate_decay: {agent.learning_rate_decay}')
    print(f'learning_rate_min: {agent.learning_rate_min}')
    print(f'learning_rate_current: {agent.optimizer.param_groups[0]["lr"]}')
    print(f'reward_clipping: {agent.reward_clipping}')
    print(f'gradient_clipping: {agent.gradient_clipping}')
    print(f'discount_factor: {agent.discount_factor}')
    print(f'batch_size: {agent.batch_size}')
    print(f'memory_maxlen: {agent.memory.maxlen}')
    print(f'memory_size: {len(agent.memory)}')
    print(f'k_count: {agent.k_count}')
    print(f'k_target: {agent.k_target}')
    print(f'optimizer: {agent.optimizer}')
    print('=====================================================')

    print('training model ...')
    start = datetime.datetime.now()
    train_start = datetime.datetime.now()

    # instead of 0 start from 1 to max_steps to avoid .. avoid what?
    for step in range(continue_steps + 1, training_steps + 1):
        epsilons.append(agent.epsilon)

        # reset env
        if done:

            # call garbage collector to free memory
            gc.collect()

            # init hidden
            agent.policy_net.init_hidden()
            agent.target_net.init_hidden()

            if not init:
                training_returns.append(training_return)
            init = False

            # reset env
            env.reset()

            # start game
            # press fire (1) and continue
            state, reward, done, info = env.step(1)
            state = t.transform(state)
            lives = info['ale.lives']
            training_return = 0

        # predict action, no need for gradients
        with torch.no_grad():
            action = agent.policy(state, mode='training')

        # execute action
        next_state, reward, done, info = env.step(action)
        next_state = t.transform(next_state)
        training_return += reward

        # append experience tuple to memory buffer
        agent.append_sample(state, action, reward, next_state, done)
        state = next_state

        # minimize epsilon
        agent.minimize_epsilon()

        # death detection
        # press fire (1) and continue
        if lives != info['ale.lives'] and not done:
            state, reward, done, info = env.step(1)
            state = t.transform(state)
            lives = info['ale.lives']
            training_return += reward

            # init hidden
            agent.policy_net.init_hidden()
            agent.target_net.init_hidden()

        # train every 4th step
        if step % 4 == 0:
            # save last hidden and cell state of policy_net.decoder since both are zeroed during training
            h_t, c_t = agent.policy_net.dec_h_t, agent.policy_net.dec_c_t

            # train policy net with batch and obtain loss
            loss = agent.train()

            # reset last hidden and cell state of policy_net.decoder to state before training
            agent.policy_net.dec_h_t, agent.policy_net.dec_c_t = h_t, c_t

            # append loss and increment training_counter
            losses.append(loss)
            train_counter += 1

        # enter evaluation phase
        if step % evaluation_start == 0 or step == training_steps:
            train_end = datetime.datetime.now()
            print(f'... done! time: {train_end - train_start}')
            print(f'training: {train_counter} / {int(training_steps / 4)}, steps: {step} / {training_steps}')
            print('-----------------------------------------------------')
            print('evaluating model ...')
            start_time = datetime.datetime.now()
            avg_return = evaluate_model(agent)
            end_time = datetime.datetime.now()
            print(f'... done! time: {end_time - start_time}')
            print('-----------------------------------------------------')
            print(f'saving checkpoint ...')
            fileio.save_checkpoint(agent, train_counter, step, directory)
            print(f'... done!')
            print('-----------------------------------------------------')
            evaluation_returns.append(avg_return)
            results = {'loss': losses, 'evaluation_returns': evaluation_returns, 'training_returns': training_returns,
                       'epsilons': epsilons}
            print(f'saving intermediate results ...')
            fileio.save_results(results, directory)
            print(f'... done!')
            print('-----------------------------------------------------')
            print(f'plotting intermediate results ...')
            plots.plot_intermediate_results(directory, agent.optimizer, **results)
            print(f'... done!')
            print('=====================================================')
            print('continue training ...')
            done = True
            gc.collect()
            train_start = datetime.datetime.now()

    end = datetime.datetime.now()
    print(f'overall time: {end - start}')
    results = {'loss': losses, 'evaluation_returns': evaluation_returns, 'training_returns': training_returns,
               'epsilons': epsilons}
    print('=====================================================================')
    print(f'saving results and final model ...')
    fileio.save_results(results, directory)
    fileio.save_checkpoint(agent, train_counter, training_steps, directory)
    print(f'... done!')
    print('---------------------------------------------------------------------')
    print(f'plotting final results ...')
    plots.plot_intermediate_results(directory, agent.optimizer, **results)
    print(f'... done!')
    print('=====================================================================')
