# import torch
import matplotlib.pyplot as plt
import numpy as np
# weights plot code
# def accumulate_svd(x, title, all_singular_values):
#     U, S, Vh = torch.linalg.svd(x)
#     data = S.cpu().detach().numpy()
#     all_singular_values[title] = data

# def plot_combined_svd(all_singular_values, title):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     colors = plt.cm.viridis(np.linspace(0, 1, len(all_singular_values)))
#     i=0
#     for idx, singular_values in all_singular_values.items() :
#         ax.plot(singular_values, color=colors[int(i)], label=f'Checkpoint {idx}')
#         i=i+1

#     ax.set_title(f'Singular Values of {title.capitalize()} weights')
#     ax.set_xlabel('Singular Value Index')
#     ax.set_ylabel('Singular Value')
#     ax.legend()
#     plt.show()
#     plt.savefig(f'combined_svd_{title}_kqdse_60.png')

# checkpoints = [0, 20, 40, 60,80]
# all_q_singular_values = {}
# all_k_singular_values = {}
# all_attn_singular_values = {}

# for checkpoint in checkpoints:
#     checkpoint_path = f'exp/cifar10/kqdse_60/checkpoint-{checkpoint}.pth'
#     model = torch.load(checkpoint_path)

#     weights_11 = model['model']['blocks.11.attn.qkv.weight']
#     attn = model['model']['blocks.11.attn.proj.weight']
#     q = weights_11[:768, :]
#     k = weights_11[768:1536, :]

#     accumulate_svd(q, str(checkpoint), all_q_singular_values)
#     accumulate_svd(k, str(checkpoint), all_k_singular_values)
#     accumulate_svd(attn, str(checkpoint), all_attn_singular_values)

# plot_combined_svd(all_q_singular_values, 'q')
# plot_combined_svd(all_k_singular_values, 'k')
# plot_combined_svd(all_attn_singular_values,'attn')


# values plot code
import pickle
import torch
import einops


def accumulate_svd(x, title, all_singular_values):
    shape = x.shape
    all_singular_values[title] = []
    for i in range(shape[0]):
        mat = x[i].cpu().float()
        U, S, Vh = torch.linalg.svd(mat)
        data = S.cpu().detach().numpy()
        all_singular_values[title].append(data)

def plot_combined_svd_values(all_singular_values, title , checkpoint):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_singular_values)))
    i=0
    for idx, singular_values in enumerate(all_singular_values) :
        ax.plot(singular_values, color=colors[int(i)], label=f'image {idx}')
        i=i+1

    ax.set_title(f"Singular Values of 16 {title.capitalize()}'s")
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.legend()
    plt.show()
    plt.savefig(f'svd_{title}_values_kqdse_60_{str(checkpoint)}.png')        


# Define the directory path and file name

all_q_singular_values = {}
all_k_singular_values = {}
all_attn_singular_values = {}
checkpoints = [0, 20, 40, 60,80]
# Load tensors from the file using pickle
for checkpoint in checkpoints:
    checkpoint_path = f'exp/cifar10/kqdse_60/checkpoint_{checkpoint}_mask.pkl'
    with open(checkpoint_path, 'rb') as file:
        loaded_tensors = pickle.load(file)
    # Access the loaded tensors
    for idx, tensor in enumerate(loaded_tensors):
        if idx == 0 or idx == 1 :
            batch_size, num_heads, n, C = tensor.shape
            tensor  = einops.rearrange(tensor, 'b h n d -> b n (h d)') 
        if(idx == 0):
            accumulate_svd(tensor, str(checkpoint), all_q_singular_values)
        elif(idx == 1):
            accumulate_svd(tensor, str(checkpoint), all_k_singular_values)
        else:
            accumulate_svd(tensor, str(checkpoint), all_attn_singular_values)
        print(f"Tensor {idx + 1}:")
        print(tensor.shape)
    plot_combined_svd_values(all_q_singular_values[str(checkpoint)], 'q',checkpoint)
    plot_combined_svd_values(all_k_singular_values[str(checkpoint)], 'k',checkpoint)
    plot_combined_svd_values(all_attn_singular_values[str(checkpoint)], 'attn',checkpoint)
    
