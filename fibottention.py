# Copyright (c) Charlotte-CharMLab at UNC Charlotte.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/Charlotte-CharMLab/Fibottention
# --------------------------------------------------------

import torch
import math
import random

# Getting a masked attention pattern using Wythoff's sequence
def get_mask_attn_wythoff(q, k, is_modified, depth_id, add_class_token=True):
    # Remove the first token from the query and key
    q_adjusted = q[:, :, 1:, :]
    k_adjusted = k[:, :, 1:, :]
    
    B, H, N, _ = q_adjusted.size()  # Batch size, number of heads, number of tokens, embedding size
    headindices = generate_head_indices(N=N, h=H, wmin=5, is_modified=is_modified)
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)

    # Shuffle head indices across layers
    # headindices = shuffle(depth_id, headindices)
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            # Create diagonal masks
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1
            indices = torch.arange(max(i, 0), min(N, N + i))
            mask[:, h, indices, indices - i] = 1

        print(f'h= {h}, fib_indices= {fib_indices}')

    if add_class_token:
        # Extend mask to include the first token
        mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
        mask_extended[:, :, 1:, 1:] = mask

    return mask_extended

# Generate head indices using Wythoff sequence and Fibonacci numbers
def generate_head_indices(N, h, wmin, is_modified):
    wmax = N//3
    headindices = [[] for _ in range(h)]
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * phi ** 2))
        w = wmin + int((wmax - wmin) / (h - 1) * (i - 1))

        if is_modified:
            b_Wyt_m = b - a
            a_Wyt_m = a - b_Wyt_m
            headindices[i - 1] = get_fibonacci(a_Wyt_m, b_Wyt_m, w)
        else:
            headindices[i - 1] = get_fibonacci(a, b, w)

    headindices = [torch.tensor(seq, dtype=torch.int64) for seq in headindices]
    return headindices

# # Generate Fibonacci sequence within a given range
def get_fibonacci(a, b, w):
    fib_seq = [a, b]
    while fib_seq[-1] <= w:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return fib_seq[:-1]

# Shuffle the array of sets using a given seed
def shuffle(i, array_of_sets):
    random.seed(i)
    shuffled_array = array_of_sets[:]
    random.shuffle(shuffled_array)
    return shuffled_array