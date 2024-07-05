import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange

class Upsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv2dBlock(nn.Module):
    '''
        Conv2d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv2dBlock(256, 128, kernel_size=3)
    up = Upsample2d(128)
    x = torch.zeros((1, 256, 19, 19))
    o = up(cb(x))
    print(o.shape)

if __name__ == '__main__':
    test()
