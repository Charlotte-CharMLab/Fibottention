# Fibottention
## Inceptive Visual Representation Learning with Diverse Attention Across Heads

#### [Ali Khaleghi Rahimian](https://ak811.github.io)<sup>1</sup> , [Manish Kumar Govind](https://manishgovind.github.io/)<sup>1</sup> , [Subhajit Maity](https://maitysubhajit.github.io/)<sup>2</sup> , [Dominick Reilly](https://dominickrei.github.io)<sup>1</sup> , [Christian Kümmerle](https://webpages.charlotte.edu/~ckuemme1/)<sup>1</sup>* , [Srijan Das](https://srijandas07.github.io)<sup>1</sup>* , and [Aritra Dutta](https://sciences.ucf.edu/math/person/aritra-dutta/)<sup>2</sup>*
\* Equal contribution as Project Lead

#### Affiliations:
<sup>1</sup> University of North Carolina at Charlotte  
<sup>2</sup> University of Central Florida

<br>

## Overview
This repository contains the implementation of our proposed Fibottention mechanism and related algorithms. 

The paper is now available on [arXiv](https://arxiv.org/abs/2406.19391).

<br>

## Abstract
Visual perception tasks are predominantly solved by Vision Transformer (ViT) architectures, which, despite their effectiveness, encounter a computational bottleneck due to the quadratic complexity of computing self-attention. This inefficiency is largely due to the self-attention heads capturing redundant token interactions, reflecting inherent redundancy within visual data. Many works have aimed to reduce the computational complexity of self-attention in ViTs, leading to the development of efficient and sparse transformer architectures. In this paper, viewing through the efficiency lens, we realized that introducing any sparse self-attention strategy in ViTs can keep the computational overhead low. However, these strategies are sub-optimal as they often fail to capture fine-grained visual details. This observation leads us to propose a general, efficient, sparse architecture, named Fibottention, for approximating self-attention with superlinear complexity that is built upon Fibonacci sequences. The key strategies in Fibottention include: it excludes proximate tokens to reduce redundancy, employs structured sparsity by design to decrease computational demands, and incorporates inception-like diversity across attention heads. This diversity ensures the capture of complementary information through non-overlapping token interactions, optimizing both performance and resource utilization in ViTs for visual representation learning. We embed our Fibottention mechanism into multiple state-of-the-art transformer architectures dedicated to visual tasks. Leveraging only 2-6% of the elements in the self-attention heads, Fibottention in conjunction with ViT and its variants, consistently achieves significant performance boosts compared to standard ViTs in nine datasets across three domains — image classification, video understanding, and robot learning tasks.

<br>

## Installation and training of Fibottention 

### Image classification

Use the commands below to install the required packages when setting up your environment:
```
conda create --name env_name --no-default-packages python=3.7
pip install -r requirements.txt
```

To train the model, use the script.sh file which executes main_finetune.py with specified parameters.

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

### Action recognition

#### Install dependencies 

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- fvcore: `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- matlotlib : `pip install matplotlib`


#### DataSet preparation
The dataset could be structured as follows:
```
├── data
    ├── Action_01
        ├── Video_01.mp4
        ├── Video_02.mp4
        ├── …
```
After all the data is prepared, resize and crop the video to person-centric to get rid of background noise. Then, prepare the CSV files for the training, validation, and testing sets as `train.csv`, `val.csv`, and `test.csv`. The format of the CSV file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```


#### Training 

We provide configs to train fibottention for action recognition  on Smarthome, NTU and NUCLA datasets  in [action_recognition/configs/](configs/). Please update the paths in the config to match the paths in your machine before using.

For example to train  on Smarthome using 8 GPUs run the following command:

`python action_recognition/tools/run_net.py --cfg configs/SMARTHOME.yaml NUM_GPUS 8`

<br>

### Robot learning

#### Install dependencies & download data
Our robot learning code is built on top of the code for ["Crossway Diffusion: Improving Diffusion-based Visuomotor Policy via Self-supervised Learning"](https://arxiv.org/abs/2307.01849). Please follow their instructions for installing dependencies and obtaining the data.
* Follow the installation instructions from [this link](https://github.com/LostXine/crossway_diffusion?tab=readme-ov-file#installation)
* Follow instructions from [this link](https://github.com/LostXine/crossway_diffusion?tab=readme-ov-file#download-datasets) to download the datasets
    * This page will direct you to [this download page](https://diffusion-policy.cs.columbia.edu/data/training/). To reproduce the results in this work, only `pusht.zip` and `robomimic_image.zip` are needed

#### Training
To perform the robot learning experiments
1. Navigate to `robot_learning/`
2. Run the following command, replacing `<DATASET>` with the desired dataset (this work uses `can_ph`, `lift_ph`, or `pusht`):
```
train.py --config-dir=config/<DATASET>/ --config-name=typea.yaml training.seed=42 hydra.run.dir=outputs/vit-b-fibottention/${now:%Y-%m-%d}/${now:%H-%M-%S}_${task_name}_${task.dataset_type}
```

<br>

## Algorithms

### Algorithm 1: Generating Fibonacci Sequence with Constraint

```python
# INPUT: a, b, w_i  # Initial two numbers and upper constraint
# OUTPUT: fib_seq  # Fibonacci sequence under constraint

fib_seq = [a, b]  # Initialize sequence with first two numbers
while fib_seq[-1] < w_i:  # Generate sequence until last number is less than w_i
    next_num = fib_seq[-1] + fib_seq[-2]  # Calculate next Fibonacci number
    fib_seq.append(next_num)  # Append new number to the sequence
return fib_seq  # Return the generated Fibonacci sequence
```

### Algorithm 2: Generating Mask for All Heads
```python
# INPUT: L, N, w_min, w_max, is_modified  # Layer, size, window min/max, and modification flag
# OUTPUT: Ω ∈ (0,1)^(h × (N+1) × (N+1))  # Output mask tensor

phi = (1 + sqrt(5)) / 2  # Golden ratio for indexing
for i in range(1, h + 1):  # Loop over each head
    a = int(i * phi * phi)  # Fibonacci starting values based on golden ratio
    b = int(i * phi * phi**2)
    w_i = w_min + int((i - 1) * (w_max - w_min) / (h - 1))  # Window size for this head
    Θ = [[0]*N for _ in range(N)]  # Initialize intermediate mask

    if is_modified:  # Modify sequence if required
        b_Wyt_m = b - a
        a_Wyt_m = a - b_Wyt_m
        I = get_fibonacci(a_Wyt_m, b_Wyt_m, w)   # Calculate Fibonacci indices using Algorithm 1
    else:
        I = get_fibonacci(a, b, w)
    
    for o in I:  # Apply Fibonacci indices to mask
        for j in range(N-o):
            Θ[j][j+1] = 1  # Upper triangular masking
        for k in range(o, N):
            Θ[k+1][k] = 1  # Lower triangular masking
    
    Ω_i = [[1]*(N+1) for _ in range(N+1)]  # Initialize output mask for head i
    for j in range(1, N+1):  # Fill in mask based on Θ
        for k in range(1, N+1):
            Ω_i[j][k] = Θ[j-1][k-1]

Ω = [Ω_i for i in range(h)]  # Combine masks from all heads
Ω = randomshuffle(L, Ω)  # Randomly shuffle masks across layers
return Ω  # Return the final mask tensor
```

### Algorithm 3: Fibottention in a Single Vision Transformer Block
```python
# INPUT: X ∈ R^(N+1 × d)  # Input feature matrix
# OUTPUT: O ∈ R^(N+1 × d)  # Output feature matrix
# PARAMETERS: W_i^Q, W_i^K, W_i^V ∈ R^(d × d_h), d_h = d / h  # Weights for Q, K, V
# HYPERPARAMETERS: w_min, w_max, is_modified  # Window sizes and modification flag

iota_Ω = getMask(L, N, h, w_min, w_max, is_modified)  # Get mask from Algorithm 2

for i in range(1, h + 1):  # Process each attention head
    Q_i = X @ W_i^Q  # Query matrix
    K_i = X @ W_i^K  # Key matrix
    V_i = X @ W_i^V  # Value matrix
    A_i = Q_i @ K_i.T  # Attention scores
    A_i_Ω = np.sign(A_i) * (np.abs(A_i) * iota_Ω[i,:,:])  # Apply mask to attention scores
    A_i_Ω = softmax(A_i_Ω)  # Softmax to normalize scores
    Z_i = A_i_Ω @ V_i  # Weighted sum to produce output for head

Z = np.concatenate([Z_i for i in range(h)], axis=1)  # Concatenate outputs from all heads
O = Z @ W^Z  # Project concatenated outputs to final dimension
return O  # Return output of Vision Transformer block
```

<!--
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
-->

<br>

## Acknowledgement

This repository is built on top of  [`MAE`](https://github.com/facebookresearch/mae), [`TimeSformer`](https://github.com/facebookresearch/TimeSformer), and [`Crossway Diffusion`](https://github.com/LostXine/crossway_diffusion). We would like to thank all the contributors for their well-organized codebases.

<br>

## Citation
```
@misc{rahimian2024fibottentioninceptivevisualrepresentation,
      title={Fibottention: Inceptive Visual Representation Learning with Diverse Attention Across Heads}, 
      author={Ali Khaleghi Rahimian and Manish Kumar Govind and Subhajit Maity and Dominick Reilly and Christian Kümmerle and Srijan Das and Aritra Dutta},
      year={2024},
      eprint={2406.19391},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.19391}, 
}
```

<br>

## License
This project is licensed under the Creative Commons Attribution 4.0 International - see the [LICENSE](https://creativecommons.org/licenses/by/4.0/deed.en) website for details.
