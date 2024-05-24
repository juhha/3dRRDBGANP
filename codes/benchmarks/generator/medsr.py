"""
ref - https://github.com/04RR/MedSRGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from monai.networks.layers import Conv, Norm

from ..utils import make_layer, default_init_weights, PixelShuffle, PixelUnshuffle

class RWMAB(nn.Module):
    def __init__(self, spatial_dims, in_channels):
        super().__init__()
        conv_module = Conv['conv', spatial_dims]
        self.layer1 = nn.Sequential(
            conv_module(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace = True),
            conv_module(in_channels, in_channels, 3, stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            conv_module(in_channels, in_channels, 1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x_ = self.layer1(x)
        x__ = self.layer2(x_)

        return x__ * x_ + x


class ShortResidualBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels):
        super().__init__()
        
        self.layers = nn.ModuleList([RWMAB(spatial_dims, in_channels) for _ in range(16)])

    def forward(self, x):

        res = x

        for layer in self.layers:
            x = layer(x)

        return x * 0.2 + res


class MedSR(nn.Module):
    def __init__(self, spatial_dims = 3, in_channels=1, out_channels = 1, blocks=8):
        """
        fixed to 4x upscaling
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.sigmoid = False
        self.scale = 4 # this is fixed by the author of MedSR
        
        conv_module = Conv['conv', spatial_dims]

        self.conv = conv_module(in_channels, 64, 3, stride=1, padding=1)

        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(spatial_dims, 64) for _ in range(blocks)]
        )

        self.conv2 = conv_module(64, 64, 1, stride=1, padding=0)

        self.conv3 = nn.Sequential(
            conv_module(128, 256, 3, stride=1, padding=1),
            PixelShuffle(spatial_dims, [2] * spatial_dims),
            # nn.PixelShuffle(2),
            conv_module(256 // np.prod([2] * spatial_dims), 256, 3, stride=1, padding=1),
            PixelShuffle(spatial_dims, [2] * spatial_dims),
            # nn.PixelShuffle(2),
            conv_module(256 // np.prod([2] * spatial_dims), out_channels, 1, stride=1, padding=0),  # Change 64 -> 256
            # nn.Sigmoid(), # FIX. i'm not so sure about this... maybe comment this out?
        )

    def forward(self, x):

        x = self.conv(x)
        res = x

        for layer in self.short_blocks:
            x = layer(x)

        x = torch.cat([self.conv2(x), res], dim=1)

        x = self.conv3(x)

        return x
    def predict(self, x):
        x = F.interpolate(x, scale_factor = [1/self.scale for i in range(self.spatial_dims)], mode = 'nearest')
        return self.forward(x)