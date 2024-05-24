"""
ref - [BRAIN MRI SUPER RESOLUTION USING 3D DEEP DENSELY CONNECTED NEURAL NETWORKS](https://arxiv.org/abs/1801.02728)
        https://arxiv.org/pdf/1803.01417
        
"""

import torch.nn as nn
import numpy as np

from monai.networks.layers import Conv, Norm

class SimpleDiscriminator(nn.Module):
    def __init__(self, spatial_dims: int, num_in_ch, num_feat, input_size=[96,96,96]):
        super(SimpleDiscriminator, self).__init__()
        self.spatial_dims = spatial_dims
        self.input_size = input_size

        conv_module = Conv['conv', spatial_dims]
        norm_module = Norm['layer', spatial_dims]
        
        # conv blocks
        self.convs = nn.Sequential(
            # first conv layer
            conv_module(num_in_ch, num_feat, 3, 1, 1, bias = True),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            # conv with stride=2
            conv_module(num_feat, num_feat, 3, 2, 1, bias = False), # / 2
            norm_module([num_feat, *[i//2 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            # conv-stride 1
            conv_module(num_feat, num_feat*2, 3, 1, 1, bias = False),
            norm_module([num_feat*2, *[i//2 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            conv_module(num_feat*2, num_feat*2, 3, 2, 1, bias = False), # / 4
            norm_module([num_feat*2, *[i//4 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            # conv-stride 2
            conv_module(num_feat*2, num_feat*4, 3, 1, 1, bias = False),
            norm_module([num_feat*4, *[i//4 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            conv_module(num_feat*4, num_feat*4, 3, 2, 1, bias = False), # / 8
            norm_module([num_feat*4, *[i//8 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            # conv-stride 3
            conv_module(num_feat*4, num_feat*8, 3, 1, 1, bias = False),
            norm_module([num_feat*8, *[i//8 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            conv_module(num_feat*8, num_feat*8, 3, 2, 1, bias = False), # / 16
            norm_module([num_feat*8, *[i//16 for i in input_size]]),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
        )
        # 
        inc = num_feat * 8 * np.prod([i//16 for i in input_size])
        self.linear = nn.Sequential(
            nn.Linear(inc, 1024),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        feat2d = self.convs(x)
        out = self.linear(feat2d.flatten(start_dim = 1))
        return out