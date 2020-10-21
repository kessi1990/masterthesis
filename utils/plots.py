import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_intermediate_results(directory, **data):
    fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
    axarr[0, 0].plot(data['loss'], linewidth=1)
    axarr[0, 0].set(xlabel='training steps', ylabel='loss')

    axarr[0, 1].plot(data['epsilons'], linewidth=1)
    axarr[0, 1].set(xlabel='training steps', ylabel='epsilon')

    axarr[1, 0].plot(data['training_returns'], linewidth=1)
    axarr[1, 0].set(xlabel='training episodes', ylabel='return')

    axarr[1, 1].plot(data['evaluation_returns'], linewidth=1)
    axarr[1, 1].set(xlabel='evaluation episodes', ylabel='return')

    plt.tight_layout()
    plt.savefig(directory + 'results.png')
    plt.close()


def sliding_window(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg
