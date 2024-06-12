# Fibottention
## Inceptive Visual Representation Learning with Diverse Attention Across Heads

This repository contains the implementation of our proposed Fibottention mechanism and related algorithms. The complete code base will be provided soon.

<br>

## Abstract
Visual perception tasks are predominantly solved by Vision Transformer (ViT) architectures, which, despite their effectiveness, encounter a computational bottleneck due to the quadratic complexity of computing self-attention. This inefficiency is largely due to the self-attention heads capturing redundant token interactions, a reflection of inherent redundancy within visual data. A plethora of works have aimed to reduce the computational complexity of self-attention in ViTs, leading to the development of efficient and sparse transformer architectures. In this paper, viewing through the efficiency lens, we realized that introducing any sparse self-attention strategy in ViTs can keep the computational overhead low. Still, these strategies are sub-optimal as they often fail to capture fine-grained visual details. This observation leads us to propose a general, efficient, sparse architecture, named Fibottention, for approximating self-attention with superlinear complexity that is built upon Fibonacci sequences. The key strategies in Fibottention include: it excludes proximate tokens to reduce redundancy, employs structured sparsity by design to decrease computational demands, and incorporates inception-like diversity across attention heads. This diversity ensures the capture of complementary information through non-overlapping token interactions, optimizing both performance and resource utilization in ViTs for visual representation learning. We embed our Fibottention mechanism into multiple state-of-the-art transformer architectures dedicated to visual tasks. By leveraging only 2-6% of the elements in the self-attention heads, Fibottention in conjunction with ViT and its variants, consistently achieves significant performance boosts compared to standard ViTs in nine datasets across three domains — image classification, video understanding, and robot learning tasks.

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

## License
This project is licensed under the MIT License - see the [[LICENSE](https://opensource.org/license/mit)](LICENSE) file for details.
