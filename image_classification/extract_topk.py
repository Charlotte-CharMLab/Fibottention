import torch
import torch.nn as nn
import einops
from utils.perturbed_topk import PerturbedTopK

def extractK(x):

    # Assuming x is your input tensor with shape (32, 196, 768)
    # Replace this with your actual data
    print("x-shape in ek",x.shape)

    # Define the number of patches, batch size, and the desired value of k
    batch_size, n_heads , num_patches, features_per_head = x.shape  
    k = 384

    # Initialize the PerturbedTopK module
    perturbed_topk = PerturbedTopK(k=k, num_samples=5, sigma=0.05)
    
    x_reshaped = einops.rearrange(x, 'B heads N head_dim -> (B N)(heads head_dim)')
    x = einops.rearrange(x, 'B heads N head_dim -> (B) (N)(heads head_dim)')

    # Apply PerturbedTopK to find the top-k features for each patch in parallel
    topk_features = perturbed_topk(x_reshaped)
    topk_features = einops.rearrange(topk_features, '(b n) k d -> b n k d', b=batch_size, n=num_patches)
    
    print("output",topk_features.shape)
    
    topk_features =  torch.einsum('bnd,bnkd->bnk', x, topk_features)

    topk_features = einops.rearrange(topk_features, 'b n (h k)  -> b h n k', b=batch_size, h = n_heads)

    print("output",topk_features.shape)

    return topk_features 