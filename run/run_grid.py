import sys
sys.path.append('..')

import torch
import gc
import logging
import copy

from agents import agents as a
from utils import fileio
from utils import plots
from utils import env_loader


# framestack: 1, 2, 4
# model_type: no-lstm, identity
# alignment: dot, general, concat
# env_type: small, large
# dir_id: xxx
# out_channels: 4, 8    -> no-lstm only
# random_start: True, False
# random_goal: True, False
frame_stack = int(sys.argv[1])
model_type = sys.argv[2]
out_channels = int((sys.argv[3]))
alignment = sys.argv[4]
env_type = sys.argv[5]
random_start = eval(sys.argv[6])
dir_id = int(sys.argv[7])

directory = fileio.mkdir_g(f'fs-{frame_stack}_{model_type}-{out_channels}', f'grid-{env_type}_random_s-{random_start}', dir_id, alignment)
checkpoint = fileio.load_checkpoint(directory)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = env_loader.load(size=env_type, frame_stack=frame_stack, random_start=False, random_goal=False, device=device)

logging.basicConfig(filename=f'{directory}run.log', filemode='a',
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

if device.type == 'cpu':
    logging.warning(f'torch device type: {device.type} - terminate run')
    # exit()

logging.info('run started!')
logging.info(f'frame_stack: {frame_stack}')
logging.info(f'dir_id: {dir_id}')
logging.info(f'path: {directory}')
logging.info(f'device: {device}')
logging.info(f'model: {model_type}')
logging.info(f'env_type: {env_type}')
logging.info(f'random_start: {random_start}')

# training & evaluation steps
training_steps = 400000
evaluation_start = 4000
evaluation_steps = 40000


def evaluate_model(model):
    logging.info('-----------------------------------------------------')
    logging.info('evaluating model ...')

    return_e = 0
    returns_e = []
    t_e = 0

    # set policy_net to eval mode
    model.policy_net.eval()

    # reset environment
    state_e = env.reset()

    history = {}
    starting_coord = env.start_coord
    goal_coord = env.goal_coord
    weights_history = []
    actions_history = []
    remember = True

    for step_e in range(evaluation_steps):

        # predict, no need for gradients
        with torch.no_grad():
            action_e, weights_e = model.policy(state_e, mode='evaluation')

        actions_history.append(action_e)
        weights_history.append(weights_e.tolist())

        # execute action
        next_state_e, reward_e, done_e, info_e = env.step(action_e)
        state_e = next_state_e

        # accumulate reward
        return_e += reward_e * (0.99 ** t_e)

        t_e += 1

        # reset env
        if done_e:
            if remember:
                history = {'starting_coord': starting_coord.tolist(),
                           'goal_coord': goal_coord.tolist(),
                           'actions_history': copy.deepcopy(actions_history),
                           'weights_history': copy.deepcopy(weights_history)}
                remember = False

            returns_e.append(return_e)
            return_e = 0
            t_e = 0
            state_e = env.reset()
            starting_coord = env.start_coord
            goal_coord = env.goal_coord
            actions_history.clear()
            weights_history.clear()
            gc.collect()

    fileio.save_history(directory, history)
    logging.info(f'write history ...')

    logging.info('-----------------------------------------------------')

    return (sum(returns_e) / len(returns_e), min(returns_e), max(returns_e)) if len(returns_e) > 0 else (0, 0, 0)


def restore(model, check_p):
    logging.info(f'found checkpoint in directory:\n{directory}')
    logging.info(f'loading checkpoint ...')
    model.policy_net.load_state_dict(check_p['policy_net'])
    model.target_net.load_state_dict(check_p['target_net'])
    model.policy_net.to(device)
    model.target_net.to(device)
    model.optimizer.load_state_dict(check_p['optimizer'])
    model.learning_rate = check_p['learning_rate']
    model.learning_rate_decay = check_p['learning_rate_decay']
    model.learning_rate_min = check_p['learning_rate_min']
    model.epsilon = check_p['epsilon']
    model.epsilon_decay = check_p['epsilon_decay']
    model.epsilon_min = check_p['epsilon_min']
    model.discount_factor = check_p['discount_factor']
    model.batch_size = check_p['batch_size']
    model.k_count = check_p['k_count']
    model.k_target = check_p['k_target']
    model.reward_clipping = check_p['reward_clipping']
    model.gradient_clipping = check_p['gradient_clipping']
    return model


def log_parameters(model):
    logging.info(f'frame_stack: {frame_stack}')
    logging.info(f'epsilon: {model.epsilon}')
    logging.info(f'epsilon_decay: {model.epsilon_decay}')
    logging.info(f'epsilon_min: {model.epsilon_min}')
    logging.info(f'learning_rate_start: {model.learning_rate}')
    logging.info(f'learning_rate_decay: {model.learning_rate_decay}')
    logging.info(f'learning_rate_min: {model.learning_rate_min}')
    logging.info(f'learning_rate_current: {model.optimizer.param_groups[0]["lr"]}')
    logging.info(f'reward_clipping: {model.reward_clipping}')
    logging.info(f'gradient_clipping: {model.gradient_clipping}')
    logging.info(f'discount_factor: {model.discount_factor}')
    logging.info(f'batch_size: {model.batch_size}')
    logging.info(f'memory_maxlen: {model.memory.maxlen}')
    logging.info(f'memory_size: {len(model.memory)}')
    logging.info(f'k_count: {model.k_count}')
    logging.info(f'k_target: {model.k_target}')
    logging.info(f'optimizer: {model.optimizer}')
    logging.info('=====================================================')


if __name__ == '__main__':
    agent = a.DQNFS(model_type, len(env.action_space), device, stacked_frames=frame_stack, alignment=alignment, hidden_size=out_channels, out_channels=out_channels)

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

    logging.info('=====================================================')

    if checkpoint:
        train_counter = checkpoint['train_counter']
        continue_steps = checkpoint['continue']
        agent = restore(agent, checkpoint)
        logging.info(f'continue training at: {train_counter} / {int(training_steps / 4)}, steps: {continue_steps} / {training_steps}')

    log_parameters(agent)
    logging.info('training model ...')

    t = 0

    state = env.reset()
    for step in range(continue_steps + 1, training_steps + 1):
        epsilons.append(agent.epsilon)

        # predict, no need for gradients
        with torch.no_grad():
            action, _ = agent.policy(state, mode='training')

        # execute action
        next_state, reward, done, info = env.step(action)

        # store transition in memory buffer
        agent.append_sample(state, action, reward, next_state, done)
        state = next_state

        # accumulate reward
        training_return += reward * (0.99 ** t)

        # minimize epsilon
        agent.minimize_epsilon()

        t += 1

        if done:
            training_returns.append(training_return)
            training_return = 0
            t = 0
            # reset env
            state = env.reset()
            gc.collect()

        # train every 4th step
        if step % 4 == 0:
            loss = agent.train()
            losses.append(loss)
            train_counter += 1

        # enter evaluation phase
        if step % evaluation_start == 0 or step == training_steps:
            logging.info(f'training: {train_counter} / {int(training_steps / 4)}, steps: {step} / {training_steps}')

            avg_return = evaluate_model(agent)
            logging.info(f'saving checkpoint ...')
            fileio.save_checkpoint(agent, train_counter, step, directory)

            evaluation_returns.append(avg_return)
            results = {'loss': losses, 'evaluation_returns': evaluation_returns, 'training_returns': training_returns,
                       'epsilons': epsilons}
            logging.info(f'saving intermediate results ...')
            fileio.save_results(results, directory)

            logging.info(f'plotting intermediate results ...')
            plots.plot_intermediate_results(directory, agent.optimizer, **results)

            logging.info('=====================================================')
            logging.info('continue training ...')
            training_return = 0
            state = env.reset()
            gc.collect()
            t = 0

    results = {'loss': losses, 'evaluation_returns': evaluation_returns, 'training_returns': training_returns,
               'epsilons': epsilons}
    logging.info('=====================================================================')
    logging.info(f'saving results and final model ...')
    fileio.save_results(results, directory)
    fileio.save_checkpoint(agent, train_counter, training_steps, directory)
    logging.info('---------------------------------------------------------------------')
    logging.info(f'plotting final results ...')
    plots.plot_intermediate_results(directory, agent.optimizer, **results)
