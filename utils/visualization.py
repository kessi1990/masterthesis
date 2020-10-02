import torch
import time

from matplotlib import pyplot as plt


def visualize_attention(attention_weights, input_frame, directory, i):
    upsample = torch.nn.Upsample(scale_factor=12, mode='bicubic', align_corners=False)
    # attention_weights (49, 256)
    # --> max: (49, 1)
    # --> transpose: (1, 49)
    # --> reshape: (1, 7, 7)
    # --> unsqueeze: (1, 1, 7, 7)
    # --> upsample: (1, 1, 84, 84)
    attention_weights = torch.max(attention_weights, dim=-1, keepdim=True)
    attention_weights = torch.transpose(attention_weights[0], dim0=1, dim1=0)
    attention_weights = attention_weights.reshape(1, 7, 7)
    attention_weights = attention_weights.unsqueeze(dim=0)
    attention_weights = upsample(attention_weights).detach()

    input_frame = input_frame.detach()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_frame.squeeze(), 'gray', interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(input_frame.squeeze(), 'gray', interpolation='none')
    plt.imshow(normalize(attention_weights.squeeze()), 'plasma', interpolation='none', alpha=0.5)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{directory}{time.time()}_attention_{i}.png')
    plt.close('all')


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img
