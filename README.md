# Fibottention
## Inceptive Visual Representation Learning with Diverse Attention Across Heads

#### [Ali Khaleghi Rahimian](https://ak811.github.io)<sup>1</sup> , [Manish Kumar Govind](https://manishgovind.github.io/)<sup>1</sup> , [Subhajit Maity](https://maitysubhajit.github.io/)<sup>2</sup> , [Dominick Reilly](https://dominickrei.github.io)<sup>1</sup> , [Christian Kümmerle](https://webpages.charlotte.edu/~ckuemme1/)<sup>1</sup>* , [Srijan Das](https://srijandas07.github.io)<sup>1</sup>* , and [Aritra Dutta](https://sciences.ucf.edu/math/person/aritra-dutta/)<sup>2</sup>*
\* Equal contribution as Project Lead

#### Affiliations:
<sup>1</sup> University of North Carolina at Charlotte  
<sup>2</sup> University of Central Florida

<br>

## Overview
This repository contains the implementation of our proposed Fibottention mechanism and related algorithms. The complete codebase for video understanding and robot learning will be provided soon.

The paper is now available on [arXiv](https://arxiv.org/abs/2406.19391).

<br>

## Abstract
Visual perception tasks are predominantly solved by Vision Transformer (ViT) architectures, which, despite their effectiveness, encounter a computational bottleneck due to the quadratic complexity of computing self-attention. This inefficiency is largely due to the self-attention heads capturing redundant token interactions, reflecting inherent redundancy within visual data. Many works have aimed to reduce the computational complexity of self-attention in ViTs, leading to the development of efficient and sparse transformer architectures. In this paper, viewing through the efficiency lens, we realized that introducing any sparse self-attention strategy in ViTs can keep the computational overhead low. However, these strategies are sub-optimal as they often fail to capture fine-grained visual details. This observation leads us to propose a general, efficient, sparse architecture, named Fibottention, for approximating self-attention with superlinear complexity that is built upon Fibonacci sequences. The key strategies in Fibottention include: it excludes proximate tokens to reduce redundancy, employs structured sparsity by design to decrease computational demands, and incorporates inception-like diversity across attention heads. This diversity ensures the capture of complementary information through non-overlapping token interactions, optimizing both performance and resource utilization in ViTs for visual representation learning. We embed our Fibottention mechanism into multiple state-of-the-art transformer architectures dedicated to visual tasks. Leveraging only 2-6% of the elements in the self-attention heads, Fibottention in conjunction with ViT and its variants, consistently achieves significant performance boosts compared to standard ViTs in nine datasets across three domains — image classification, video understanding, and robot learning tasks.

<br>

## Installation and training of Fibottention for image classification tasks:

Use the commands below to install the required packages when setting up your environment:
```
conda create --name env_name --no-default-packages python=3.7
pip install -r requirements.txt
```

To train the model, use the script.sh script which executes main_finetune.py with specified parameters.

```bash
./script.sh [id] [out_dir] [model] [dataset] [classes] [device] [batch] [mask_ratio]
```

- `id`: An identifier for the execution.
- `out_dir`: The output directory for saving results.
- `model`: The type of model for finetuning.
- `dataset`: The name of the dataset to be used.
- `classes`: The number of classes in the dataset.
- `device`: The GPU device number or ID.
- `batch`: The batch size for training.
- `mask_ratio`: The ratio for masking during training.

For example, to train a model on the CIFAR-10 dataset, use the following command:

```bash
./script.sh 1 exp/cifar10/test base c10 10 0 16 0.4
```

This command will trigger the script with the specified parameters, initiating the training process with the chosen settings.

<br>

## Algorithms

### Algorithm 1: Generating Fibonacci Sequence with Constraint

```python
def getFibonacci(a, b, w_i):
    fib_seq = [a, b]
    while fib_seq[-1] < w_i:
        next_num = fib_seq[-1] + fib_seq[-2]
        fib_seq.append(next_num)
    return fib_seq
```

### Algorithm 2: Generating Mask for All Heads

```python
def generate_fibomask(L, N, wmin, wmax, is_modified):
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio, used for Fibonacci calculations
    a = int((i * phi) * phi)  # First Fibonacci-like number for masking
    b = int((i * phi) * (phi ** 2))  # Second Fibonacci-like number for masking
    Omega = np.zeros((h, N+1, N+1))  # Initialize the Omega tensor to store masks for all heads
    
    for i in range(h):  # Loop through each head
        w_i = wmin + int(((i - 1) * (wmax - wmin)) / (h - 1))  # Calculate the window size for current head
        Theta = np.zeros((N, N))  # Initialize Theta matrix for storing the mask
        I = getFibonacci(a, b, w_i)  # Generate Fibonacci sequence with constraints
        
        if is_modified and i > 1:  # Modify the sequence if required
            I.extend([0, (a-i)])  # Extend with additional values based on 'i'
            I.extend([0, (i-1)])
        
        for o in I:  # Loop through Fibonacci numbers
            for j in range(N-o):
                Theta[j, j+1] = 1  # Set forward connections
            for k in range(o, N):
                Theta[k+1, k] = 1  # Set backward connections
        
        Omega_i = np.ones((N+1, N+1))  # Create an Omega mask filled with ones
        for j in range(1, N):  # Adjust the mask using Theta values
            for k in range(1, N):
                Omega_i[j+1, k+1] = Theta[j, k]
        
        Omega[i, :, :] = Omega_i  # Assign the computed mask for current head
    
    Omega = np.prod(Omega, axis=0)  # Combine masks across all heads
    Omega = random_shuffle(L, Omega)  # Randomly shuffle the mask for more variance
    
    return Omega  # Return the final mask
```

### Algorithm 3: Fibottention in a Single Vision Transformer Block

```python
def fibottention(X, W_Q, W_K, W_V, d_h, wmin, wmax, is_modified):
    N, d = X.shape  # Get the dimensions of input X
    Omega = getMask(L, N, h, wmin, wmax, is_modified)  # Generate the mask using the given parameters
    
    Z = []  # Initialize a list to store the output of each head
    for i in range(h):  # Loop through each head
        Q_i = X @ W_Q[i]  # Calculate the query matrix
        K_i = X @ W_K[i]  # Calculate the key matrix
        V_i = X @ W_V[i]  # Calculate the value matrix
        
        A_i = Q_i @ K_i.T  # Compute attention scores
        A_i_Omega = np.sign(A_i) * (np.abs(A_i) * Omega[i, :, :])  # Apply the mask
        A_i_Omega = softmax(A_i_Omega)  # Normalize scores with softmax
        
        Z_i = A_i_Omega @ V_i  # Compute the output of attention mechanism
        Z.append(Z_i)  # Add the result to the list
    
    Z = np.concatenate(Z, axis=-1)  # Concatenate outputs from all heads
    O = Z @ W_Z  # Final linear transformation
    
    return O  # Return the final output
```

<br>

## Citation
```
@article{rahimian2024fibottention,
    title={Inceptive Visual Representation Learning with Diverse Attention Across Heads},
    author={Rahimian, Ali K. and Govind, Manish K. and Maity, Subhajit and Reilly, Dominick and Kümmerle, Christian and Das, Srijan and Dutta, Aritra},
    journal={arXiv preprint},
    archivePrefix={arXiv},
    year={2024},
    eprint={2211.01410}
}
```

<br>

## License
This project is licensed under the Creative Commons Attribution 4.0 International - see the [LICENSE](https://creativecommons.org/licenses/by/4.0/deed.en) website for details.
