from robomimic.models.base_nets import Module # VisualCore, ConvBase

from diffusion_policy.model.vision.models_vit import vit_tiny, vit_small
from diffusion_policy.model.vision.vision_transformer import VisionTransformer

class ViT_Tiny(Module):
    def __init__(self, **kwargs):
        super(ViT_Tiny, self).__init__()

        self.net = vit_tiny(**kwargs)

    def output_shape(self, input_shape):
        return (self.net.embed_dim,)

    def forward(self, inputs):
        return self.net(inputs)

class ViT_Small(Module):
    def __init__(self, **kwargs):
        super(ViT_Small, self).__init__()

        self.net = vit_small(**kwargs)

    def output_shape(self, input_shape):
        return (self.net.embed_dim,)

    def forward(self, inputs):
        return self.net(inputs)

class ViT_Base(Module):
    def __init__(self, **kwargs):
        super(ViT_Base, self).__init__()

        self.net = VisionTransformer(**kwargs)

    def output_shape(self, input_shape):
        return (self.net.embed_dim,)

    def forward(self, inputs):
        return self.net(inputs)

# vis_net = VisualCore(
#   input_shape=(3, 76, 76),
#   backbone_class="ViT_Tiny",  # use ResNet18 as the visualcore backbone
#   backbone_kwargs={"img_size": 76},  # kwargs for the ResNet18Conv class
# #   pool_class="SpatialSoftmax",  # use spatial softmax to regularize the model output
# #   pool_kwargs={"num_kp": 32},  # kwargs for the SpatialSoftmax --- use 32 keypoints
# #   flatten=True,  # flatten the output of the spatial softmax layer
#   feature_dimension=64,  # project the flattened feature into a 64-dim vector through a linear layer 
# )