"""
ref - [BRAIN MRI SUPER RESOLUTION USING 3D DEEP DENSELY CONNECTED NEURAL NETWORKS](https://arxiv.org/abs/1801.02728)
        https://github.com/YunzeMan/DCSRN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from monai.networks.layers import Conv, Norm
from monai.networks.layers.convutils import same_padding

def init_weights(module_list):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):                
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

class DenseUnit(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 24,
        no_of_filters: int = 24,
        f_size: int = 3,
        bias: bool = False
    ):
        layers = []
        layers.append(Norm['batch', spatial_dims](in_channels))
        layers.append(nn.ELU()),
        layers.append(Conv['conv', spatial_dims](in_channels, no_of_filters, f_size, 1, same_padding(f_size, 1), bias = bias))
        super(DenseUnit, self).__init__(*layers)

class DCSRN(nn.Module):
    """
    ref - [BRAIN MRI SUPER RESOLUTION USING 3D DEEP DENSELY CONNECTED NEURAL NETWORKS](https://arxiv.org/abs/1801.02728)
        https://github.com/YunzeMan/DCSRN
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int =1,
        k: int = 24,
        filter_size: int = 3,
        num_units_per_block: int = 4,
        bias: bool = False
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        conv_module = Conv['conv', spatial_dims]
        self.init_conv = conv_module(in_channels, 2 * k, filter_size, 1, same_padding(filter_size, 1), bias = bias)
        layers = []
        inc = 2*k
        res = [inc]
        for i in range(num_units_per_block-1):
            layers.append(DenseUnit(spatial_dims, inc, k, filter_size, bias))
            res.append(k)
            inc = sum(res)
        self.body = nn.ModuleList(layers)
        self.out_conv = conv_module(inc, out_channels, 1, 1, same_padding(1, 1), bias = bias)
        init_weights([self.init_conv, self.body, self.out_conv])
    def forward(self, x):
        res = []
        res.append(self.init_conv(x))
        for block in self.body:
            res.append(block(torch.cat(res, dim = 1)))
        return self.out_conv(torch.cat(res, dim = 1))
    def predict(self, x):
        return self.forward(x)