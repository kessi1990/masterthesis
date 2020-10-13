import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from utils import fileio
import numpy as np


def plot_intermediate_results(directory, **data):
    if not data:
        print('no data')
        return
    fig, axarr = plt.subplots(2, 1, figsize=(10, 10))
    axarr[0].plot(data['loss'], linewidth=1)
    # axarr[0].plot(sliding_window(data['loss'], 10), linewidth=3)
    axarr[0].set(xlabel='training steps', ylabel='loss')

    """axarr[0, 1].plot(data['epsilons'], linewidth=0.5)
    axarr[0, 1].set(xlabel='training episodes', ylabel='epsilon')"""

    axarr[1].plot(data['training_returns'], linewidth=1)
    # axarr[1].plot(sliding_window(data['training_returns'], 5), linewidth=3)
    axarr[1].set(xlabel='training episodes', ylabel='return')

    """axarr[1, 1].plot(data['evaluation_returns'], linewidth=0.5)
    axarr[1, 1].set(xlabel='evaluation episode', ylabel='avg return')"""

    plt.tight_layout()
    plt.savefig(directory + 'results.png')


def sliding_window(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg


model_type = 'darqn'
env_type = 'SpaceInvaders-v0'
num_layers = 49

# old
# directory = f'C:\\Users\\Michael\\Desktop\\results\\old\\2020-09-27_05-02-12_{model_type}_{env_type}_{num_layers}--0\\'
# target_dir = f'C:\\Users\\Michael\\Desktop\\results\\old\\{model_type}_{env_type}_{num_layers}_'

# new
directory = f'C:\\Users\\Michael\\Desktop\\results\\new\\{model_type}_{env_type}_{num_layers}\\'
target_dir = f'C:\\Users\\Michael\\Desktop\\results\\new\\{model_type}_{env_type}_{num_layers}_'

results = fileio.load_results(directory)
plot_intermediate_results(target_dir, **results)
