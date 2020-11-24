import sys
sys.path.append('..')

import torch
import datetime
import gc
import random

# pytorch lr_scheduler.state_dict() and .load_state_dict throw warnings when being called --> ignore
import warnings
warnings.filterwarnings("ignore")

from agents import agents as a
from utils import env_loader
from utils import fileio
from utils import transformation
from utils import plots


frame_stack = True  # eval(sys.argv[1])
env_size = sys.argv[1]
dir_id = int(sys.argv[2])
out_channels = int(sys.argv[3])
model_type = 'no_lstm'
directory = fileio.mkdir_g(model_type, f'grid-{env_size}-{out_channels}', dir_id)
checkpoint = fileio.load_checkpoint(directory)
env = env_loader.load_rooms_env(size=env_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = transformation.TransformationGridNoLSTM()

print(f'frame_stack: {frame_stack}')
print(f'env_size: {env_size}')
print(f'dir_id: {dir_id}')
print(f'path: {directory}')
print(f'device: {device}')

training_steps = 10000000  # 1000000  # 5000000
evaluation_start = 5000  # 10000  # 50000
evaluation_steps = 2500  # 5000   # 25000


def evaluate_model(model):

    # set policy_net to eval mode
    model.policy_net.eval()

    eval_returns = []
    eval_return = 0
    steps = 0
    epoch_steps = 0
    eval_done = True
    eval_init = True
    eval_state = None
    eval_next_state = None
    if frame_stack:
        eval_state_stack = None
        eval_next_state_stack = None

    while steps < evaluation_steps:
        # reset env
        if eval_done:
            gc.collect()
            if not eval_init:
                eval_return = (agent.discount_factor ** epoch_steps) * eval_return
                eval_returns.append(eval_return)
            eval_init = False
            eval_state = env.reset()
            eval_state = t.transform(eval_state)

            if frame_stack:
                eval_state_stack = torch.cat([eval_state for _ in range(4)], dim=1)
                eval_next_state_stack = torch.clone(eval_state_stack)
            epoch_steps = 0
            eval_return = 0

        # predict, no need for gradients
        with torch.no_grad():
            action = model.policy(eval_state_stack if frame_stack else eval_state, mode='evaluation')

        eval_next_state, eval_reward, eval_done, eval_info = env.step(action)
        eval_next_state = t.transform(eval_next_state)

        if frame_stack:
            eval_next_state_stack = torch.cat((eval_next_state_stack[:, 1:], eval_next_state), dim=1)
            eval_state_stack = eval_next_state_stack

        eval_state = eval_next_state

        eval_return += eval_reward
        steps += 1
        epoch_steps += 1

    return (sum(eval_returns) / len(eval_returns), min(eval_returns), max(eval_returns)) if len(eval_returns) > 0 else (0, 0, 0)


def fill_memory_buffer(model):

    done = True
    state = None
    next_state = None

    nr_actions = 4
    action_space = [_ for _ in range(nr_actions)]

    while len(model.memory) < 20000:

        # reset env
        if done:
            state = env.reset()
            state = t.transform(state)

        # random action selection
        action = random.choice(action_space)

        next_state, reward, done, info = env.step(action)
        next_state = t.transform(next_state)
        model.append_sample(state, action, reward, next_state, done)

        state = next_state


if __name__ == '__main__':
    nr_actions = 4  # gridworld actions -> 0: up, 1: down, 2: left, 3: right
    # agent = a.DQNRaw(4 if frame_stack else 1, nr_actions, device)
    agent = a.DQNNew(in_channels=4, out_channels=out_channels, nr_actions=4, device=device)

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
    ep_steps = 0
    train_counter = 0
    training_return = 0
    done = True
    init = True
    state = None
    next_state = None
    if frame_stack:
        state_stack = None
        next_state_stack = None

    print('=====================================================')
    print(f'model: {model_type}')
    print(f'env_size: {env_size}')
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

    for step in range(continue_steps + 1, training_steps + 1):
        epsilons.append(agent.epsilon)

        if done:
            gc.collect()
            if not init:
                training_return = (agent.discount_factor ** ep_steps) * training_return
                training_returns.append(training_return)

            init = False

            # reset env
            state = env.reset()
            state = t.transform(state)

            if frame_stack:
                state_stack = torch.cat([state for _ in range(4)], dim=1)
                next_state_stack = torch.clone(state_stack)
            else:
                next_state = torch.clone(state)

            ep_steps = 0
            training_return = 0

        # predict, no need for gradients
        with torch.no_grad():
            action = agent.policy(state_stack if frame_stack else state, mode='training')

        next_state, reward, done, info = env.step(action)
        next_state = t.transform(next_state)
        agent.append_sample(state, action, reward, next_state, done)

        if frame_stack:
            next_state_stack = torch.cat((next_state_stack[:, 1:], next_state), dim=1)
            state_stack = next_state_stack

        state = next_state
        training_return += reward
        ep_steps += 1

        # minimize epsilon
        agent.minimize_epsilon()

        # train every 4th step
        if step % 4 == 0:
            loss = agent.train()
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
            init = True
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
    plots.plot_intermediate_results(directory, **results)
    print(f'... done!')
    print('=====================================================================')
