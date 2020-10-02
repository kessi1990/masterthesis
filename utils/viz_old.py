import torch
import torch.nn as nn
from torchvision.transforms import functional as f
import matplotlib.pyplot as plt
from utils import fileio


class Visualizer(nn.Module):
    def __init__(self, config):
        super(Visualizer, self).__init__()
        self.root_dir = config['sub_dir']
        self.cwd = fileio.visual_dir(self.root_dir)
        self.config = config

    def forward(self, in_features):
        out = self.conv_transposed_1(in_features)
        out = self.conv_transposed_2(out)
        out = self.conv_transposed_3(out)
        return out

    @staticmethod
    def normalize(img):
        img = img - img.min()
        img = img / img.max()
        return img

    @staticmethod
    def shape(tensor, i):
        nr_filters = tensor.size(i)
        res = nr_filters // 8
        if res > 8:
            rows = 8
            lines = res
        else:
            rows = res
            lines = 8
        return rows, lines

    def activations(self, activations):
        for k, v in activations.items():
            for seq_idx, feature_maps in enumerate(v):
                rows, columns = self.shape(feature_maps, 0)
                fig, axarr = plt.subplots(rows, columns, figsize=(15, 12))
                i = 0
                for y in range(columns):
                    for x in range(rows):
                        axarr[x, y].imshow(feature_maps[i].squeeze(), cmap=plt.get_cmap('gray'))
                        axarr[x, y].set_axis_off()
                        i += 1
                fig.tight_layout()
                plt.savefig(self.cwd + f'activations_{k}_img-id-{seq_idx}.png')
                plt.close()

    def kernels(self, layer):
        kernels = layer.weight.detach()
        rows, columns = self.shape(kernels, 0)
        fig, axarr = plt.subplots(rows, columns, figsize=(15, 12))
        i = 0
        for y in range(columns):
            for x in range(rows):
                axarr[x, y].imshow(self.normalize(kernels[i].squeeze()), cmap=plt.get_cmap('gray'))
                axarr[x, y].set_axis_off()
                i += 1
        fig.tight_layout()
        plt.savefig(self.cwd + 'kernels.png')
        plt.close()

    def context(self, context):
        context = context.squeeze().transpose(1, 0).reshape(self.config['input_size_dec'], 8, 8).detach()
        rows, columns = self.shape(context, 0)
        fig, axarr = plt.subplots(rows, columns, figsize=(15, 12))
        i = 0
        for y in range(columns):
            for x in range(rows):
                axarr[x, y].imshow(context[i].squeeze(), cmap=plt.get_cmap('plasma'))
                axarr[x, y].set_axis_off()
                i += 1
        # fig.tight_layout()
        plt.savefig(self.cwd + 'context.png')
        plt.close()

    def attention_applied(self, attention_applied):
        attention_applied = attention_applied.detach()
        rows, lines = 8, 8  # self.shape(context)
        fig, axarr = plt.subplots(rows, lines, figsize=(15, 12))
        i = 0
        for y in range(lines):
            for x in range(rows):
                axarr[x, y].imshow(attention_applied[i].squeeze())
                axarr[x, y].set_axis_off()
                i += 1
        fig.tight_layout()
        plt.savefig(self.cwd + 'attention_applied.png')
        plt.close()

    def update_cwd(self):
        self.cwd = fileio.visual_dir(self.root_dir)

    def start(self, cpt_data, cnn_model, visor_data):
        self.attention_applied(visor_data['applied_attention'])
        self.new(visor_data['attentional_hidden'], visor_data['weights'], cpt_data['conv_3'])
        self.activations(cpt_data)
        self.kernels(cnn_model.conv_1)
        self.context(visor_data['context_vectors'])
        self.update_cwd()

    def new(self, attentional_hidden, weights, feature_maps):
        att_h = attentional_hidden.squeeze().transpose(1, 0).reshape(128, 8, 8).detach()
        rows, columns = 16, 8
        fig, axarr = plt.subplots(rows, columns, figsize=(15, 12))
        i = 0
        for y in range(columns):
            for x in range(rows):
                axarr[x, y].imshow(att_h[i].squeeze(), cmap=plt.get_cmap('plasma'))
                axarr[x, y].set_axis_off()
                i += 1
        fig.tight_layout()
        plt.savefig(self.cwd + 'attentional_hidden_v1.png')
        plt.close()

        att_h = attentional_hidden.squeeze().transpose(1, 0).reshape(128, 8, 8).detach()
        rows, columns = 16, 8
        fig, axarr = plt.subplots(rows, columns, figsize=(15, 12))
        i = 0
        for y in range(columns):
            for x in range(rows):
                axarr[x, y].imshow(self.normalize(att_h[i].squeeze()), cmap=plt.get_cmap('plasma'))
                axarr[x, y].set_axis_off()
                i += 1
        fig.tight_layout()
        plt.savefig(self.cwd + 'attentional_hidden_v1_normalized.png')
        plt.close()

        att_h_2 = torch.sum(attentional_hidden, dim=2).squeeze().reshape(8, 8).detach()
        plt.imshow(att_h_2, cmap='plasma')
        fig.tight_layout()
        plt.savefig(self.cwd + 'attenional_hidden_v2.png')
        plt.close()

        att_h_2_n = torch.sum(self.normalize(attentional_hidden), dim=2).squeeze().reshape(8, 8).detach()
        plt.imshow(att_h_2_n, cmap='plasma')
        fig.tight_layout()
        plt.savefig(self.cwd + 'attenional_hidden_v2_normalized.png')
        plt.close()

        feature_maps_3 = feature_maps[-1].detach()
        rows, columns = 16, 8
        fig, axarr = plt.subplots(rows, columns, figsize=(15, 12))
        i = 0
        for y in range(columns):
            for x in range(rows):
                axarr[x, y].imshow(self.normalize(feature_maps_3[i].squeeze()), cmap=plt.get_cmap('gray'))
                axarr[x, y].imshow(self.normalize(att_h[i].squeeze()), cmap=plt.get_cmap('plasma'), alpha=0.5)
                axarr[x, y].set_axis_off()
                i += 1
        fig.tight_layout()
        plt.savefig(self.cwd + 'a_h_fm.png')
        plt.close()
