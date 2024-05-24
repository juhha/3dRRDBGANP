import numpy as np, torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# from .train_utils import LaplacianFilter3D, SobelFilter3D, HaarWaveletTransform3D

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

def calc_mae(target, pred):
    return abs(target - pred).mean()

def calc_psnr(target, pred, data_range = None):
    data_range = target.max() - target.min() if data_range is None else data_range
    return peak_signal_noise_ratio(target, pred, data_range = data_range)

def calc_ssim(target, pred, data_range = None, channel_axis: int = None):
    data_range = target.max() - target.min() if data_range is None else data_range
    return structural_similarity(target, pred, data_range = data_range, channel_axis = channel_axis)

def calc_ldf(target, pred, channel_axis: int = 0):
    """Calculate LFD (Log Frequency Distance)
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        target (ndarray): target image
        pred (ndarray): prediction image
    """
    axes = [i for i in range(len(target.shape)) if i != channel_axis]
    fft1 = np.fft.fftn(target, axes = axes)
    fft2 = np.fft.fftn(pred, axes = axes)
    return np.log(np.mean((fft1.real - fft2.real)**2 + (fft1.imag - fft2.imag)**2) + 1.0)

def calc_ldf_torch(target, pred, channel_axis: int = 0):
    axes = [i for i in range(len(target.shape)) if i != channel_axis]
    fft1 = torch.fft.fftn(target, dim = axes)
    fft2 = torch.fft.fftn(pred, dim = axes)
    return torch.log(torch.mean((fft1.real - fft2.real)**2 + (fft1.imag - fft2.imag)**2) + 1.0)