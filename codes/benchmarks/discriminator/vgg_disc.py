import torch.nn as nn
import numpy as np

from monai.networks.layers import Conv, Norm

class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, spatial_dims: int, num_in_ch, num_feat, input_size=[96,96,96]):
        super(VGGStyleDiscriminator, self).__init__()
        self.spatial_dims = spatial_dims
        self.input_size = input_size

        conv_module = Conv['conv', spatial_dims]
        norm_module = Norm['batch', spatial_dims]
        self.conv0_0 = conv_module(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = conv_module(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = norm_module(num_feat, affine=True)

        self.conv1_0 = conv_module(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = norm_module(num_feat * 2, affine=True)
        self.conv1_1 = conv_module(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = norm_module(num_feat * 2, affine=True)

        self.conv2_0 = conv_module(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = norm_module(num_feat * 4, affine=True)
        self.conv2_1 = conv_module(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = norm_module(num_feat * 4, affine=True)

        self.conv3_0 = conv_module(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = norm_module(num_feat * 8, affine=True)
        self.conv3_1 = conv_module(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = norm_module(num_feat * 8, affine=True)

        self.conv4_0 = conv_module(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = norm_module(num_feat * 8, affine=True)
        self.conv4_1 = conv_module(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = norm_module(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * np.prod([i//32 for i in input_size]), 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        # spatial size: (3,3) when patch_size = [96,96,96]
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out