"""
ref - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
"""
from ..utils import make_layer, PixelShuffle, PixelUnshuffle

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import Conv

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, spatial_dims: int, scale, num_feat):
        conv_module = Conv['conv', spatial_dims]
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(conv_module(num_feat, (2**spatial_dims) * num_feat, 3, 1, 1))
                m.append(PixelShuffle(spatial_dims, 2))
        elif scale == 3:
            m.append(conv_module(num_feat, (3**spatial_dims) * num_feat, 3, 1, 1))
            m.append(PixelShuffle(spatial_dims, 3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, spatial_dims: int, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.spatial_dims = spatial_dims
        conv_module = Conv['conv', spatial_dims]
        pool_layer = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)
        self.attention = nn.Sequential(
            pool_layer, conv_module(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), conv_module(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, spatial_dims: int, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.spatial_dims = spatial_dims
        self.res_scale = res_scale
        
        conv_module = Conv['conv', spatial_dims]

        self.rcab = nn.Sequential(
            conv_module(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), conv_module(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(spatial_dims, num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x

class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, spatial_dims: int, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()
        self.spatial_dims = spatial_dims
        
        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale, spatial_dims = spatial_dims)
        conv_module = Conv['conv', spatial_dims]
        self.conv = conv_module(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x

class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 spatial_dims: int,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 # img_range=255.,
                 # rgb_mean=(0.4488, 0.4371, 0.4040)
        ):
        super(RCAN, self).__init__()
        # no normalization here
        # self.img_range = img_range
        # self.mean = torch.Tensor(rgb_mean).view(1, num_in_ch, 1, 1)
        self.spatial_dims = spatial_dims
        # upscale = [upscale] * spatial_dims
        self.upscale = upscale
        conv_module = Conv['conv', spatial_dims]
        
        self.conv_first = conv_module(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale,
            spatial_dims = spatial_dims
        )
        self.conv_after_body = conv_module(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(spatial_dims, upscale, num_feat)
        self.conv_last = conv_module(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        # self.mean = self.mean.type_as(x)

        # x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        # x = x / self.img_range + self.mean
        
        return x
    def predict(self, x):
        x = F.interpolate(x, scale_factor = [1/self.upscale for i in range(self.spatial_dims)], mode = 'nearest')
        return self.forward(x)