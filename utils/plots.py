import matplotlib
matplotlib.use('Agg')

import numpy as np

from matplotlib import pyplot as plt


def plot_intermediate_results(directory, **data):
    fig, axarr = plt.subplots(2, 1, figsize=(10, 10))
    axarr[0].plot(data['loss'], linewidth=1)
    axarr[0].set(xlabel='training steps', ylabel='loss')

    axarr[1].plot(data['training_returns'], linewidth=1)
    axarr[1].set(xlabel='training episodes', ylabel='return')

    plt.tight_layout()
    plt.savefig(directory + 'results.png')


def sliding_window(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg
