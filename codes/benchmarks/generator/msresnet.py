"""
ref - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
"""
from ..utils import make_layer, default_init_weights, PixelShuffle, PixelUnshuffle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import Conv


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, spatial_dims: int = 3, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        conv_module = Conv['conv', spatial_dims]
        self.res_scale = res_scale
        self.conv1 = conv_module(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = conv_module(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, spatial_dims: int = 3, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.spatial_dims = spatial_dims
        conv_module = Conv['conv', spatial_dims]
        self.interp_mode = 'bilinear' if spatial_dims == 2 else 'trilinear'
        
        self.upscale = upscale
        
        self.conv_first = conv_module(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, spatial_dims = spatial_dims)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = conv_module(num_feat, num_feat * (self.upscale ** self.spatial_dims), 3, 1, 1)
            self.pixel_shuffle = PixelShuffle(spatial_dims, self.upscale)
        elif self.upscale == 4:
            self.upconv1 = conv_module(num_feat, num_feat * (2 ** self.spatial_dims), 3, 1, 1)
            self.upconv2 = conv_module(num_feat, num_feat * (2 ** self.spatial_dims), 3, 1, 1)
            self.pixel_shuffle = PixelShuffle(spatial_dims, 2)

        self.conv_hr = conv_module(num_feat, num_feat, 3, 1, 1)
        self.conv_last = conv_module(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode=self.interp_mode, align_corners=False)
        out += base
        return out
    def predict(self, x):
        x = F.interpolate(x, scale_factor = [1/self.upscale for i in range(self.spatial_dims)], mode = 'nearest')
        return self.forward(x)