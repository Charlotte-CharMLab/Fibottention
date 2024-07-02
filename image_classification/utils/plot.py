import matplotlib.pyplot as plt
import json
import torch
import numpy as np

def plot_attention_mask_for_all_heads(attn):
    batch_index = 0
    num_heads = attn.shape[1]

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    for head_index in range(num_heads):
        attn_cpu = attn[batch_index, head_index].cpu()
        ax = axes[head_index]
        im = ax.imshow(attn_cpu, cmap='hot', interpolation='nearest')
        ax.set_title(f'Head {head_index + 1}')
        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.6)
    plt.suptitle('Heatmaps of Mask Matrices for All Heads')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/all_heads_{current_time}.png')
    plt.close()

def plot_attention_mask_for_all_batches(attn):
    for batch_index in range(attn.size(0)):
        data = attn[batch_index].cpu().detach()

        fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            heatmap = ax.imshow(data[i], cmap='viridis', interpolation='nearest')
            ax.set_title(f'Head {i + 1}')
            ax.axis('off')

        fig.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=0.95)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'plots/mask_heatmap_batch{batch_index + 1}_{current_time}.png')
        plt.close()

def plot_attention_heatmap_for_all_batches(attn):
    for batch_index in range(attn.size(0)):
        data = attn[batch_index].cpu().detach()
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            heatmap = ax.imshow(data[i], cmap='viridis', interpolation='nearest')
            ax.set_title(f'Head {i + 1}')
            ax.axis('off')

        fig.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=0.95)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'plots/cifar100/qkT_heatmap/batch{batch_index + 1}_{current_time}.png')
        plt.close()

def plot_total_aggregated_attention_heatmap(attn):
    total_mean_attn = attn.mean(dim=[0, 1]).cpu().detach()

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(total_mean_attn, cmap='viridis', interpolation='nearest')
    ax.set_title('Aggregated Attention Map')
    ax.axis('off')
    fig.colorbar(heatmap, ax=ax, shrink=0.95)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/cifar100/total_aggregated_heatmap/baseline{current_time}.png')
    plt.close()

def plot_attention_heatmap_for_all_heads(attn):
    mean_attn = attn.mean(dim=0).cpu().detach()

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        heatmap = ax.imshow(mean_attn[i], cmap='viridis', interpolation='nearest')
        ax.set_title(f'Head {i + 1}')
        ax.axis('off')

    fig.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=0.95)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'plots/cifar100/total_aggregated_heatmap/baseline{current_time}.png')
    plt.close()
