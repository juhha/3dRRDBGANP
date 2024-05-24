"""
reference - https://github.com/XPixelGroup/BasicSR
method - ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
"""
from ..utils import make_layer, default_init_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from monai.networks.layers import Conv

def pixel_shuffle(x, spatial_dims: int, scale: (int, float, list, tuple)):
    scale = [scale for _ in range(spatial_dims)] if isinstance(scale, (int, float)) else scale
    if spatial_dims == 2:
        b,inc,h,w = x.shape
        outc = inc // np.prod(scale) # this should be separable
        f1, f2 = scale
        return x.view(b, outc, f1, f2, h, w).permute(0,1,4,2,5,3).reshape(b, outc, h*f1,w*f2)
    elif spatial_dims == 3:
        b,inc,t,h,w = x.shape
        outc = inc // np.prod(scale)
        f1, f2, f3 = scale
        return x.view(b,outc, f1,f2,f3,t,h,w).permute(0,1,5,2,6,3,7,4).reshape(b,outc,t*f1,h*f2,w*f3)

def pixel_unshuffle(x, spatial_dims: int, scale: (int, float, list, tuple)):
    scale = [scale for _ in range(spatial_dims)] if isinstance(scale, (int, float)) else scale
    if spatial_dims == 2:
        b,inc,h,w = x.shape
        outc = inc * np.prod(scale)
        f1,f2 = scale
        return x.reshape(b,inc,h//f1,f1,w//f2,f2).permute(0,1,3,6,2,4).reshape(b,outc,h//f1,w//f2)
    else: # assume spatial_dims == 3
        b,inc,t,h,w = x.shape
        outc = inc * np.prod(scale)
        f1,f2,f3 = scale
        return x.reshape(b,inc,t//f1,f1,h//f2,f2,w//f3,f3).permute(0,1,3,5,7,2,4,6).reshape(b,outc,t//f1,h//f2,w//f3)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, spatial_dims: int = 3, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.spatial_dims = spatial_dims
        conv_module = Conv['conv', spatial_dims]
        self.conv1 = conv_module(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = conv_module(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = conv_module(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = conv_module(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = conv_module(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, spatial_dims, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(spatial_dims, num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
    
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, spatial_dims, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * (2 ** spatial_dims * 2)
        elif scale == 1:
            num_in_ch = num_in_ch * (2 ** spatial_dims * 2)
        conv_module = Conv['conv', spatial_dims]
        self.conv_first = conv_module(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, spatial_dims = spatial_dims)
        self.conv_body = conv_module(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = conv_module(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = conv_module(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = conv_module(num_feat, num_feat, 3, 1, 1)
        self.conv_last = conv_module(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2, spatial_dims = self.spatial_dims)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4, spatial_dims = self.spatial_dims)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
    def predict(self, x):
        x = F.interpolate(x, scale_factor = [1/self.scale for i in range(self.spatial_dims)], mode = 'nearest')
        return self.forward(x)