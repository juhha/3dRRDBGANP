import torch
import torch.nn as nn
from torch.nn import init

import numpy as np

class PixelShuffle(nn.Module):
    def __init__(
        self,
        spatial_dims: 'int',
        scale_factors: 'list|tuple'
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.scale_factors = [scale_factors for _ in range(spatial_dims)] if isinstance(scale_factors, (float, int)) else scale_factors
    def forward(self, x):
        if self.spatial_dims == 2:
            b,inc,h,w = x.shape
            outc = inc // np.prod(self.scale_factors) # this should be separable
            f1, f2 = self.scale_factors
            return x.view(b, outc, f1, f2, h, w).permute(0,1,4,2,5,3).reshape(b, outc, h*f1,w*f2)
        else: # assume spatial_dims == 3
            b,inc,t,h,w = x.shape
            outc = inc // np.prod(self.scale_factors)
            f1, f2, f3 = self.scale_factors
            return x.view(b,outc, f1,f2,f3,t,h,w).permute(0,1,5,2,6,3,7,4).reshape(b,outc,t*f1,h*f2,w*f3)

class PixelUnshuffle(nn.Module):
    def __init__(
        self,
        spatial_dims: 'int',
        scale_factors: 'list|tuple'
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.scale_factors = [scale_factors for _ in range(spatial_dims)] if isinstance(scale_factors, (float, int)) else scale_factors
    def forward(self, x):
        if self.spatial_dims == 2:
            b,inc,h,w = x.shape
            outc = inc * np.prod(self.scale_factors)
            f1,f2 = self.scale_factors
            return x.reshape(b,inc,h//f1,f1,w//f2,f2).permute(0,1,3,6,2,4).reshape(b,outc,h//f1,w//f2)
        else: # assume spatial_dims == 3
            b,inc,t,h,w = x.shape
            outc = inc * np.prod(self.scale_factors)
            f1,f2,f3 = self.scale_factors
            return x.reshape(b,inc,t//f1,f1,h//f2,f2,w//f3,f3).permute(0,1,3,5,7,2,4,6).reshape(b,outc,t//f1,h//f2,w//f3)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)
