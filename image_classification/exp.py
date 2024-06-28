import torch
import einops
import numpy as np
import math
import time
import random

import torch
import einops

def get_mask(mask_ratio,batch_size,heads,num_patches,embed_dim):
  
  n_dim = batch_size*num_patches
  di = heads*embed_dim
  k = int(mask_ratio * di)
  sample =  torch.rand(n_dim, di,device="cuda").topk(k, dim=1).indices
  mask = torch.ones(n_dim, di, dtype=torch.bool,device="cuda")
  mask.scatter_(dim=1, index=sample, value=False)
  masked_tensor = einops.rearrange(mask, '(b n) (h d) -> b h n d', b=batch_size, n=num_patches, h=heads)
  return masked_tensor 

#Helper function to generate mask among n*n tokens
def get_mask_attn(mask_ratio,batch_size,heads,num_patches):
  n_dim = batch_size*heads
  di = num_patches*num_patches
  k = int(mask_ratio * di)
  sample =  torch.rand(n_dim, di,device="cuda").topk(k, dim=1).indices
  mask = torch.ones(n_dim, di, dtype=torch.bool,device="cuda")
  mask.scatter_(dim=1, index=sample, value=False)
  masked_tensor = einops.rearrange(mask, '(b h) (n d) -> b h n d', b=batch_size, n=num_patches, h=heads , d= num_patches)
  return masked_tensor