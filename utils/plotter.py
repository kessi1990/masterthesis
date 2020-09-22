from matplotlib import pyplot as plt


def plot_intermediate_results(directory, **data):
    if not data:
        return
    fig, axarr = plt.subplots(2, 2, figsize=(30, 10))
    axarr[0, 0].plot(data['loss'], linewidth=0.5)
    axarr[0, 0].set(xlabel='training steps', ylabel='loss')

    axarr[0, 1].plot(data['epsilons'], linewidth=0.5)
    axarr[0, 1].set(xlabel='training steps', ylabel='epsilon')

    axarr[1, 0].plot(data['training_returns'], linewidth=0.5)
    axarr[1, 0].set(xlabel='training steps', ylabel='return')

    axarr[1, 1].plot(data['evaluation_returns'], linewidth=0.5)
    axarr[1, 1].set(xlabel='evaluation episode', ylabel='avg return')

    plt.tight_layout()
    plt.savefig(directory + 'results.png')
