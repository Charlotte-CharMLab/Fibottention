import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
  def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
    super(PerturbedTopK, self).__init__()
    self.num_samples = num_samples
    self.sigma = sigma
    self.k = k

  def __call__(self, x):
    return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, k: int, num_samples: int = 100, sigma: float = 0.05):
    # b = batch size
    
    b, num_features = x.shape
    # for Gaussian: noise and gradient are the same.
    
    noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples,num_features)).to(x.device)
    
    perturbed_x = x[:, None, :] + noise * sigma # [b, num_s, num_p]
    topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=True)
   
    indices = topk_results.indices # [b, num_s, k]
    
    indices = torch.sort(indices, dim=-1).values # [b, num_s, k]
    #print(indices.shape)
    
    # b, num_s, k, num_p
    perturbed_output = F.one_hot(indices, num_classes=num_features).float()
    indicators = perturbed_output.mean(dim=1) # [b, k, num_p]
    #print(indicators.shape)
    # constants for backward
    ctx.k = k
    ctx.num_samples = num_samples
    ctx.sigma = sigma

    # tensors for backward
    ctx.perturbed_output = perturbed_output
    ctx.noise = noise

    return indicators

  @staticmethod
  def backward(ctx, grad_output):
    if grad_output is None:
      return tuple([None] * 5)

    noise_gradient = ctx.noise
    expected_gradient = (
        torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
        / ctx.num_samples
        / ctx.sigma
    )
    grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
    return (grad_input,) + tuple([None] * 5)



# class TopKFeatures(nn.Module):
#     def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
#         super(TopKFeatures, self).__init__()
#         self.num_samples = num_samples
#         self.sigma = sigma
#         self.k = k

#     def forward(self, x):
#         batch_size, num_patches, num_features = x.shape

#         # Create an empty tensor to store the top-k features for each sample in the batch
#         topk_features_list = []

#         for i in range(batch_size):
#             sample_x = x[i, :, :]  # Extract one sample from the batch

#             # Generate random noise for perturbation
#             noise = torch.normal(mean=0.0, std=1.0, size=(self.num_samples, num_patches, num_features)).to(x.device)

#             # Perturb the input features with noise
#             perturbed_x = sample_x[None, :, :] + noise * self.sigma  # [1, num_s, num_p, num_f]

#             # Select the top-k features based on perturbed values within each patch
#             topk_results = torch.topk(perturbed_x, k=self.k, dim=-1, sorted=False)
#             indices = topk_results.indices  # [1, num_s, num_p, k]
#             indices = torch.sort(indices, dim=-1).values  # [1, num_s, num_p, k]

#             # Create one-hot encoding for the selected feature indices
#             perturbed_output = F.one_hot(indices, num_classes=num_features).float()
#             indicators = perturbed_output.mean(dim=1)  # [1, num_p, k, num_f]

#             topk_features_list.append(indicators)

#         # Stack the results for all samples in the batch
#         topk_features = torch.stack(topk_features_list, dim=0)

#         return topk_features

