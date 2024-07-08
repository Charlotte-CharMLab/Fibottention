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

# Getting a masked attention pattern using Wythoff sequence
def get_mask_attn_wythoff(q, k, modified_flag, depth_id):
    # Remove the first token from the query and key
    q_adjusted = q[:, :, 1:, :]
    k_adjusted = k[:, :, 1:, :]
    
    B, H, N, _ = q_adjusted.size()  # Batch size, number of heads, number of tokens, embedding size
    headindices = generate_head_indices(N=N, h=H, omin=N, modified_flag=modified_flag)
    mask = torch.zeros((B, H, N, N), device=q.device, dtype=q.dtype)

    # Shuffle head indices based on depth_id
    headindices = shuffle(depth_id, headindices)
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            # Create diagonal masks
            indices = torch.arange(max(-i, 0), min(N, N - i))
            mask[:, h, indices, indices + i] = 1
            indices = torch.arange(max(i, 0), min(N, N + i))
            mask[:, h, indices, indices - i] = 1

    # Extend mask to include the first token
    mask_extended = torch.ones((B, H, N + 1, N + 1), device=q.device, dtype=mask.dtype)
    mask_extended[:, :, 1:, 1:] = mask

    return mask_extended

# Generate head indices using Wythoff sequence
def generate_head_indices(N, h, omin, modified_flag):
    wmax = N
    headindices = [[] for _ in range(h)]
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * phi ** 2))
        w = omin + int((wmax - omin) / (h - 1) * (i - 1))
        headindices[i - 1] = get_fibonacci(a, b, w)

        if modified_flag:
            if i > 1:
                headindices[i - 1].insert(0, a - (i - 1))
                headindices[i - 1].insert(0, i - 1)

    headindices = [torch.tensor(seq, dtype=torch.int64) for seq in headindices]
    return headindices

# Generate Fibonacci sequence within a given range
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

# Shuffle the array of sets using a given seed
def shuffle(i, array_of_sets):
    random.seed(i)
    shuffled_array = array_of_sets[:]
    random.shuffle(shuffled_array)
    return shuffled_array
