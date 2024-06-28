import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import einops
import numpy as np 

def plot_kqa(q,k):
    
    plot_hp(q, title=f'q')
    plot_hp(k , title = f'q-mask')
    # Save the figure after plotting all tensors
    plt.savefig('test_heatmap_q.png')


def plot_svd(x,title):
    batch_size, num_heads, n, C = x.shape
    x = einops.rearrange(x, 'b h n d -> (b h) n d') 
    x= x.float()
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
    colors = plt.cm.viridis(np.linspace(0, 1, num_heads))
    for head in range(num_heads) :
        U, S, Vh = torch.linalg.svd(x[head])
        data = S.cpu().detach().numpy()
        ax.plot(data, color=colors[head], label=f'Head {head + 1}')

    if title == "q_mask":
        ax.set_title('Singular Values of Q Heads')
    if title == "k_mask":
        ax.set_title('Singular Values of K Heads')
    if title == "attn_mask":
        ax.set_title('Singular Values of Attention Heads')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')

    ax.legend()
    plt.show()
    plt.savefig(f'{title}_svd.png')



def plot_hq(q, k,title=None):
    batch_size, num_heads, n, C = q.shape
    #heads_data_q_stack = einops.rearrange(q, 'b h n d -> (b h) n d') 
    heads_data_q_stack = einops.rearrange(q, 'b h n d -> b n (h d)') 
    heads_data_k_stack = einops.rearrange(k, 'b h n d -> b n (h d)')  
   
    # Plot heatmaps for each head in a single line
    fig1, ax1 = plt.subplots(2, 1, figsize=(16, 6))  # Adjusted figure size for two rows

    data_np_q = heads_data_q_stack.cpu().detach().numpy()
    im_q = ax1[0].imshow(data_np_q[0], cmap='coolwarm', aspect='auto' ,vmin= -1,vmax = 1)
    ax1[0].set_title(f'Q 12 Heads')  # Set title for query heatmap

    ax1[0].set_ylabel('Patches' , fontsize = 13)
    ax1[0].set_xlabel('Channels' , fontsize = 13)
        

    
    # cbar_q.ax.set_position([0.50, 0.88, 0.4, 0.02])
    if  title == "no":
        plt.suptitle("Heat Map of Q Before Random Masking ", fontsize=15)
    else :
        plt.suptitle("Heat Map of Q Before and After Random Masking With Ratio = 0.4", fontsize=15)    
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    data_np_k = heads_data_k_stack.cpu().detach().numpy()
    im_k = ax1[1].imshow(data_np_k[0], cmap='coolwarm', aspect='auto',vmin= -1,vmax = 1)    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # Save the plot before displaying
    # Add colorbar
    cbar = fig1.colorbar(im_k, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
    plt.subplots_adjust(right=0.85)  # Shift the plot to the left
    cbar.ax.set_position([0.9, 0.05, 0.03, 0.7])

    # if title == "no":
    #     plt.savefig('heatmap_q_stacked.png')
    # else :
    #     plt.savefig('heatmap_q_mask_stacked_40.png')    
    # plt.show()


def plot_hk(q, k,title=None):
    batch_size, num_heads, n, C = q.shape
    #heads_data_q_stack = einops.rearrange(q, 'b h n d -> (b h) n d') 
    heads_data_q_stack = einops.rearrange(q, 'b h n d -> b n (h d)') 
    heads_data_k_stack = einops.rearrange(k, 'b h n d -> b n (h d)')  
   
    # Plot heatmaps for each head in a single line
    fig1, ax1 = plt.subplots(2, 1, figsize=(16, 6))  # Adjusted figure size for two rows

    data_np_q = heads_data_q_stack.cpu().detach().numpy()
    im_q = ax1[0].imshow(data_np_q[0], cmap='coolwarm', aspect='auto' ,vmin= -1,vmax = 1)
    ax1[0].set_title(f'K 12 Heads')  # Set title for query heatmap

    ax1[0].set_ylabel('Patches' , fontsize = 13)
    ax1[0].set_xlabel('Channels' , fontsize = 13)
        

    # cbar_q = fig1.colorbar(im_q, ax=ax1[0], orientation='vertical')
    # cbar_q.ax.set_position([0.25, 0.88, 0.4, 0.02])
    if  title == "no":
        plt.suptitle("Heat Map of K Before Random Masking ", fontsize=15)
    else :
        plt.suptitle("Heat Map of K Before and After Random Masking With Ratio = 0.4", fontsize=15)    
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    data_np_k = heads_data_k_stack.cpu().detach().numpy()
    im_k = ax1[1].imshow(data_np_k[0], cmap='coolwarm', aspect='auto',vmin= -1,vmax = 1)    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    # Save the plot before displaying
    # Add colorbar
    cbar = fig1.colorbar(im_k, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
    plt.subplots_adjust(right=0.85)  # Shift the plot to the left
    cbar.ax.set_position([0.9, 0.05, 0.03, 0.7])
    if title == "no":
        plt.savefig('heatmap_k_stacked.png')
    else :
        plt.savefig('heatmap_k_mask_stacked_40.png')    
    plt.show()    


def plot_ha(q, k,title=None):
    batch_size, num_heads, n, C = q.shape
    #heads_data_q_stack = einops.rearrange(q, 'b h n d -> (b h) n d') 
    heads_data_q_stack = einops.rearrange(q, 'b h n d -> b n (h d)') 
    heads_data_k_stack = einops.rearrange(k, 'b h n d -> b n (h d)')  
   
    # Plot heatmaps for each head in a single line
    fig1, ax1 = plt.subplots(2, 1, figsize=(16, 6))  # Adjusted figure size for two rows
    print(ax1)
    data_np_q = heads_data_q_stack.cpu().detach().numpy()
    im_q = ax1[0].imshow(data_np_q[0], cmap='coolwarm', aspect='auto' ,vmin= -1,vmax = 1)
    ax1[0].set_title(f'Attention 12 Heads')  # Set title for query heatmap

    ax1[0].set_ylabel('Patches' , fontsize = 13)
    ax1[0].set_xlabel('Channels' , fontsize = 13)
        

    
    #cbar_q.ax.set_position([0.25, 0.88, 0.4, 0.02])
    if  title == "no":
        plt.suptitle("Heat Map of Attention Before Random Masking ", fontsize=15)
    else :
        plt.suptitle("Heat Map of Attention Before and After Random Masking With Ratio = 0.4", fontsize=15)    
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    data_np_k = heads_data_k_stack.cpu().detach().numpy()
    im_k = ax1[1].imshow(data_np_k[0], cmap='coolwarm', aspect='auto',vmin= -1,vmax = 1)    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    

    # Add colorbar
    cbar = fig1.colorbar(im_k, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
    plt.subplots_adjust(right=0.85)  # Shift the plot to the left
    cbar.ax.set_position([0.9, 0.05, 0.03, 0.7])
    
    if title == "no":
        plt.savefig('heatmap_attn_stacked.png')
    else :
        plt.savefig('heatmap_attn_mask_stacked_40.png')    
    plt.show()