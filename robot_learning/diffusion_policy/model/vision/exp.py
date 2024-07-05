import torch
import random
import math

"""
Fibottention
"""
def generate_head_indices(N, h, omin,modified_flag=False):
    def get_fibonacci(a, b, w):
        sequence = [a]
        if b <= w:
            sequence.append(b)
        else:
            return sequence
        while True:
            new_element = sequence[-1] + sequence[-2]
            if new_element > w:
                break
            sequence.append(new_element)
        return sequence
    headindices = [[] for _ in range(h)]
    wmax = N
    phi = (1 + math.sqrt(5)) / 2
    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * phi ** 2))
        w = omin + int((wmax - omin) / (h - 1) * (i - 1))
        headindices[i - 1] = get_fibonacci(a, b, w)
        if modified_flag:
            if i>1:
                headindices[i - 1].insert(0,a-(i-1))
                headindices[i - 1].insert(0,i-1)
    headindices = [torch.tensor(seq, dtype=torch.int64) for seq in headindices]
    return headindices

def helper_shuffle(i, shuffled_array):
    random.seed(i)
    random.shuffle(shuffled_array)
    return shuffled_array

def get_fibottention_mask(q,depth_id,modified_flag=False):
    q_adjusted = q[:, :, 1:, :]
    
    B, H, N, _ = q_adjusted.size()
    headindices = generate_head_indices(N, H, N,modified_flag)
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)
    headindices = helper_shuffle(depth_id,headindices)
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1
            indices = torch.arange(max(i, 0), min(N, N + i))
            mask[:, h, indices, indices - i] = 1

    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
    mask_extended[:, :, 1:, 1:] = mask

    return mask_extended

'''
BigBird
'''
def get_bigbird_mask(q, k):  
  w = 4 # w is the window size  for local attention along the diagonal 
  q_adjusted = q[:, :, 1:, :]
  
  B, H, N, D = q_adjusted.size()
  sample =  torch.rand(B*H, N*N,device=q.device).topk(N, dim=-1).indices
  mask = torch.zeros(B*H, N*N, device=q.device)
  mask.scatter_(dim=-1, index=sample, value=1)

  mask = mask.view(B,H,N,N)
  for i in range(-w, w + 1):
    indices = torch.arange(max(-i, 0), min(N, N - i))
    mask[:,:, indices, indices + i] = 1               

  mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device)
  mask_extended[:, :, 1:, 1:] = mask
  return mask_extended 