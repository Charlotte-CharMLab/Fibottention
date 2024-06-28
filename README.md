# Fibottention
## Inceptive Visual Representation Learning with Diverse Attention Across Heads

#### [Ali Khaleghi Rahimian](https://ak811.github.io)<sup>1</sup> , [Manish Kumar Govind](https://manishgovind.github.io/)<sup>1</sup> , [Subhajit Maity](https://maitysubhajit.github.io/)<sup>2</sup> , [Dominick Reilly](https://dominickrei.github.io)<sup>1</sup> , [Christian Kümmerle](https://webpages.charlotte.edu/~ckuemme1/)<sup>1</sup>* , [Srijan Das](https://srijandas07.github.io)<sup>1</sup>* , and [Aritra Dutta](https://sciences.ucf.edu/math/person/aritra-dutta/)<sup>2</sup>*
\* Equal contribution as Project Lead

#### Affiliations:
<sup>1</sup> University of North Carolina at Charlotte  
<sup>2</sup> University of Central Florida

This repository contains the implementation of our proposed Fibottention mechanism and related algorithms. The complete codebase for video understanding and robot learning will be provided soon.

The paper is now available on [arXiv](https://arxiv.org/abs/2406.19391).

<br>

## Abstract
Visual perception tasks are predominantly solved by Vision Transformer (ViT) architectures, which, despite their effectiveness, encounter a computational bottleneck due to the quadratic complexity of computing self-attention. This inefficiency is largely due to the self-attention heads capturing redundant token interactions, reflecting inherent redundancy within visual data. Many works have aimed to reduce the computational complexity of self-attention in ViTs, leading to the development of efficient and sparse transformer architectures. In this paper, viewing through the efficiency lens, we realized that introducing any sparse self-attention strategy in ViTs can keep the computational overhead low. However, these strategies are sub-optimal as they often fail to capture fine-grained visual details. This observation leads us to propose a general, efficient, sparse architecture, named Fibottention, for approximating self-attention with superlinear complexity that is built upon Fibonacci sequences. The key strategies in Fibottention include: it excludes proximate tokens to reduce redundancy, employs structured sparsity by design to decrease computational demands, and incorporates inception-like diversity across attention heads. This diversity ensures the capture of complementary information through non-overlapping token interactions, optimizing both performance and resource utilization in ViTs for visual representation learning. We embed our Fibottention mechanism into multiple state-of-the-art transformer architectures dedicated to visual tasks. Leveraging only 2-6% of the elements in the self-attention heads, Fibottention in conjunction with ViT and its variants, consistently achieves significant performance boosts compared to standard ViTs in nine datasets across three domains — image classification, video understanding, and robot learning tasks.

<br>

## Installation:

#### Use the commands below to install stable versions of Python and PyTorch when setting up your environment:
```
conda create --name env_name --no-default-packages python=3.7
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

<br>

## Training the model:

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
  
<br>

## Example Usage:

For example, to train a model on the CIFAR-10 dataset, use the following command::

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
    phi = (1 + 5 ** 0.5) / 2
    a = int((i * phi) * phi)
    b = int((i * phi) * (phi ** 2))
    Omega = np.zeros((h, N+1, N+1))
    
    for i in range(h):
        w_i = wmin + int(((i - 1) * (wmax - wmin)) / (h - 1))
        Theta = np.zeros((N, N))
        I = getFibonacci(a, b, w_i)
        
        if is_modified and i > 1:
            I.extend([0, (a-i)])
            I.extend([0, (i-1)])
        
        for o in I:
            for j in range(N-o):
                Theta[j, j+1] = 1
            for k in range(o, N):
                Theta[k+1, k] = 1
        
        Omega_i = np.ones((N+1, N+1))
        for j in range(1, N):
            for k in range(1, N):
                Omega_i[j+1, k+1] = Theta[j, k]
        
        Omega[i, :, :] = Omega_i
    
    Omega = np.prod(Omega, axis=0)
    Omega = random_shuffle(L, Omega)
    
    return Omega
```

### Algorithm 3: Fibottention in a Single Vision Transformer Block

```python
def fibottention(X, W_Q, W_K, W_V, d_h, wmin, wmax, is_modified):
    N, d = X.shape
    Omega = getMask(L, N, h, wmin, wmax, is_modified)
    
    Z = []
    for i in range(h):
        Q_i = X @ W_Q[i]
        K_i = X @ W_K[i]
        V_i = X @ W_V[i]
        
        A_i = Q_i @ K_i.T
        A_i_Omega = np.sign(A_i) * (np.abs(A_i) * Omega[i, :, :])
        A_i_Omega = softmax(A_i_Omega)
        
        Z_i = A_i_Omega @ V_i
        Z.append(Z_i)
    
    Z = np.concatenate(Z, axis=-1)
    O = Z @ W_Z
    
    return O
```

<br>

## Citation & Acknowledgement
```
@article{rahimian2024fibottention,
    title={Inceptive Visual Representation Learning with Diverse Attention Across Heads},
    author={Ali Khaleghi Rahimian and Manish Kumar Govind and Subhajit Maity and Dominick Reilly and Christian Kümmerle and Srijan Das and Aritra Dutta},
    journal={arXiv preprint},
    archivePrefix={arXiv},
    year={2024},
    eprint={2211.01410}
}
```

<br>

## License
This project is licensed under the Creative Commons Attribution 4.0 International - see the [LICENSE](https://creativecommons.org/licenses/by/4.0/deed.en) website for details.
