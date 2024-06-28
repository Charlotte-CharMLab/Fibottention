import torch 
import einops  
import numpy as n
import math  
import time    
import random

# Getting a masked attention pattern using Wythoff's sequence
def get_mask_attn_wythoff(q, k):
    q_adjusted = q[:, :, 1:, :]  # Adjust q tensor by slicing off the first column
    k_adjusted = k[:, :, 1:, :]  # Adjust k tensor similarly
    
    B, H, N, _ = q_adjusted.size()  # batch size, number of heads, sequence length, and feature size
    headindices = generate_head_indices_wythoff(N, H, 5)  # Generate head indices based on Wythoff's sequence

    # Initialize an attention mask tensor of zeros
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)

    # Iterate over each head to apply specific indices to mask
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1

            indices = torch.arange(max(i, 0), min(N, N + i))
            mask[:, h, indices, indices - i] = 1

    # Extend mask to accommodate extra dimensions
    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
    mask_extended[:, :, 1:, 1:] = mask

    return mask_extended

# Getting a masked attention pattern using Wythoff's sequence with shuffling across layers
def get_mask_attn_wythoff_shuffled(q, k, depth_id):
    q_adjusted = q[:, :, 1:, :] # Adjust q tensor by slicing off the first column
    k_adjusted = k[:, :, 1:, :] # Adjust k tensor similarly
    
    B, H, N, _ = q_adjusted.size() # batch size, number of heads, sequence length, and feature size
    headindices = generate_head_indices_wythoff(N, H, N)# Generate head indices based on Wythoff's sequence
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype) # Initialize an attention mask tensor of zeros
    headindices = shuffle(depth_id, headindices)  # Shuffle head indices based on depth_id

    # Iterate over each head to apply specific indices to mask
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1
            indices = torch.arange(max(i, 0), min(N, N + i))
            mask[:, h, indices, indices - i] = 1

    # Extend mask to accommodate extra dimensions
    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
    mask_extended[:, :, 1:, 1:] = mask

    return mask_extended

# Generate head indices using Wythoff sequence and Fibonacci numbers
def generate_head_indices_wythoff(N, h, omin):
    # Function to get Fibonacci sequence within a limit w
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

    headindices = [[] for _ in range(h)]  # Initialize a list of lists for head indices
    wmax = N // 3  # Set maximum window size
    phi = (1 + math.sqrt(5)) / 2  # Calculate the golden ratio

    # Generate indices for each head using a scaled Fibonacci sequence
    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * phi ** 2))
        w = omin + int((wmax - omin) / (h - 1) * (i - 1))
        headindices[i - 1] = get_fibonacci(a, b, w)

    # Convert list of sequences to tensors for use in PyTorch operations
    headindices = [torch.tensor(seq, dtype=torch.int64) for seq in headindices]

    return headindices
