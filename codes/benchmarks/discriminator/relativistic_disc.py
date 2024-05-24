"""
ref - https://github.com/04RR/MedSRGAN/tree/main
        
"""

import torch
import torch.nn as nn
import numpy as np

from monai.networks.layers import Conv, Norm

class D_Block(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, stride=2):
        super().__init__()
        conv_module = Conv['conv', spatial_dims]
        norm_module = Norm['batch', spatial_dims]
        self.layer = nn.Sequential(
            conv_module(in_channels, out_channels, 3, stride=stride, padding=1),
            norm_module(out_channels),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
        )

    def forward(self, x):

        return self.layer(x)


class RelativisticDisc(nn.Module):
    def __init__(self, spatial_dims: int = 3, img_size = [96,96,96], in_channels=1, n_feats: int = 32): # 64 is original
        """
        'n_feats' is added on top of original code to reduce # of parameters for 3d models. disc getting bigger than gen.
        """
        super().__init__()
        conv_module = Conv['conv', spatial_dims]
        norm_module = Norm['batch', spatial_dims]
        
        self.relativistic = True

        self.conv_1_1 = nn.Sequential(
            conv_module(in_channels, n_feats, 3, stride=1, padding=1), nn.LeakyReLU(inplace = True, negative_slope=0.2)
        )

        self.block_1_1 = D_Block(spatial_dims, n_feats, n_feats, stride=2)  # 2
        self.block_1_2 = D_Block(spatial_dims, n_feats, n_feats*2, stride=1)
        self.block_1_3 = D_Block(spatial_dims, n_feats*2, n_feats*2) # 4

        self.conv_2_1 = nn.Sequential(
            conv_module(in_channels, n_feats, 3, stride=1, padding=1), nn.LeakyReLU(inplace = True, negative_slope=0.2)
        )

        self.block_2_2 = D_Block(spatial_dims, n_feats, n_feats*2, stride=1)

        self.block3 = D_Block(spatial_dims, n_feats*4, n_feats*4, stride=1) 
        self.block4 = D_Block(spatial_dims, n_feats*4, n_feats*4) # 8
        self.block5 = D_Block(spatial_dims, n_feats*4, n_feats*8, stride=1)
        self.block6 = D_Block(spatial_dims, n_feats*8, n_feats*8) # 16
        self.block7 = D_Block(spatial_dims, n_feats*8, n_feats*16) # 32
        # self.block8 = D_Block(spatial_dims, n_feats*16, n_feats*16) # 64 - model gets too expensive for 3d& 96 is not divisible by 64

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(n_feats*16 * np.prod([i//32 for i in img_size]), 100) # Change based on input image size
        self.fc2 = nn.Linear(100, 1)

        self.relu = nn.LeakyReLU(inplace = True, negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        x_1 = self.block_1_3(self.block_1_2(self.block_1_1(self.conv_1_1(x1))))
        x_2 = self.block_2_2(self.conv_2_1(x2))

        x = torch.cat([x_1, x_2], dim=1)
        # x = self.block8(
        #     self.block7(self.block6(self.block5(self.block4(self.block3(x)))))
        # )
        x = self.block7(self.block6(self.block5(self.block4(self.block3(x)))))

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(self.relu(x))

        return x