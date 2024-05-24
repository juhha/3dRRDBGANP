import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

def make_3d_gaussian_kernel(window_size, sigma):
    from skimage._shared.filters import gaussian
    kernel = np.zeros(shape = window_size)
    if len(window_size) == 3:
        kernel[window_size[0]//2, window_size[1]//2, window_size[2]//2] = 1
    elif len(window_size) == 2:
        kernel[window_size[0]//2, window_size[1]//2] = 1
    kernel = gaussian(kernel, sigma = sigma)
    kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
    return kernel

class SSIMLoss(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        window_size: tuple = 7,
        sigma: float = 1.5,
        kernel_type: str = 'gaussian',
        channels: int = 1,
        dynamic_range: tuple = None        
    ):
        super().__init__()
        if spatial_dims == 3:
            self.loss_fn = SSIMLoss3D(window_size = window_size, sigma = sigma, kernel_type = kernel_type, channels = channels, dynamic_range = dynamic_range)
    def forward(self, out, target, mask = None, dynamic_range: tuple = None, k1: float = 0.01, k2: float = 0.03, return_map:bool = False):
        return self.loss_fn(out, target, mask, dynamic_range, k1, k2, return_map)
        
class SSIMLoss3D(nn.Module):
    def __init__(
        self,
        window_size: tuple = 7,
        sigma: float = 1.5,
        kernel_type: str = 'gaussian',
        channels: int = 1,
        dynamic_range: tuple = None
    ):
        super().__init__()
        self.channels = channels
        if kernel_type == 'uniform':
            self.kernel = torch.ones(self.channels,1,window_size,window_size,window_size) / (window_size ** 3)
        elif kernel_type == 'gaussian':
            self.kernel = make_3d_gaussian_kernel((window_size,window_size,window_size), sigma)
            self.kernel = self.kernel.repeat(self.channels, 1, 1, 1, 1)
        self.window_size = window_size
        self.device = 'cpu'
        self.dynamic_range = dynamic_range
    def to(self, device):
        self.kernel.to(device)
        self.device = device
    def pad(self, x):
        p = self.window_size//2
        return F.pad(x, pad = [p,p,p,p,p,p])
    def forward(self, out, target, mask = None, dynamic_range: tuple = None, k1: float = 0.01, k2: float = 0.03, return_map:bool = False):
        if self.device != out.device:
            self.device = out.device
            self.kernel = self.kernel.to(self.device)
        if dynamic_range is None and self.dynamic_range is None:
            dynamic_range = (target.min(), target.max())
        else:
            dynamic_range = self.dynamic_range
        out = out.clip(dynamic_range[0], dynamic_range[1])
        target = target.clip(dynamic_range[0], dynamic_range[1])
        if dynamic_range[0] < 0:
            out = out - dynamic_range[0]
            target = target - dynamic_range[0]
        # if mask is not None:
        #     out[mask] = dynamic_range[0]
        #     target[mask] = dynamic_range[0]
        data_range = dynamic_range[1] - dynamic_range[0]
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        
        mu1 = F.conv3d(self.pad(out), self.kernel, groups = self.channels)
        mu2 = F.conv3d(self.pad(target), self.kernel, groups = self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv3d(self.pad(out*out), self.kernel, groups = self.channels) - mu1_sq
        sigma2_sq = F.conv3d(self.pad(target*target), self.kernel, groups = self.channels) - mu2_sq
        sigma12 = F.conv3d(self.pad(out*target), self.kernel, groups = self.channels) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + c1)*(2*sigma12 + c2))/((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
        crop = (self.window_size - 1) // 2
        if mask is not None:
            m = mask[:,:,crop:-crop,crop:-crop,crop:-crop]
            ssim_score = (ssim_map[:,:,crop:-crop,crop:-crop,crop:-crop] * m).sum() / m.sum()
        else:
            ssim_score = ssim_map[:,:,crop:-crop,crop:-crop,crop:-crop].mean()
        if return_map:
            return 1-ssim_score, ssim_map
        return 1-ssim_score

class PerceptionLoss2D(nn.Module):
    def __init__(
        self,
        feature_extractor: "nn.Module",
        loss_fn = nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1,c,1,1,1)
            self.normalize_std = torch.Tensor(normalize_std).view(1,c,1,1,1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
    def forward(self, out, target):
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        loss = 0
        # Axial view
        b,c,h,w,z = out.shape
        # make 3D->2D (axial view)
        o = out.permute(0,4,1,2,3).reshape(b*z,c,h,w)
        t = target.permute(0,4,1,2,3).reshape(b*z,c,h,w)
        # make channel dimension feasible to perception model
        if c != self.channel_dim:
            repeat_channel = self.channel_dim // c # FIX: what if we have multi-modality?
            o = o.repeat(1,repeat_channel,1,1)
            t = t.repeat(1,repeat_channel,1,1)
        o_features = self.feature_extractor(o)
        t_features = self.feature_extractor(t)
        for key in o_features.keys():
            b_z,c,h,w = o_features[key].shape
            z = b_z // b
            o_feat = o_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
            t_feat = t_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
            loss += self.loss_fn(o_feat, t_feat)
        return loss
    
class PerceptionLoss3D(nn.Module):
    def __init__(
        self,
        feature_extractor: "nn.Module",
        loss_fn = nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        base_weight: float = 1.0,
        feature_weights: dict = None,
        views: str = ['axial', 'sagittal', 'coronal']
    ):
        super().__init__()
        self.views = views
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1,c,1,1,1)
            self.normalize_std = torch.Tensor(normalize_std).view(1,c,1,1,1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
        self.separate_channel = separate_channel
        self.base_weight = base_weight
    def forward(self, out, target):
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        # init loss variable
        loss = 0
        # seperate channel dim into batch dim if needed
        b,c,h,w,z = out.shape
        if c != self.channel_dim or self.separate_channel:
            b_orig,c_orig,h,w,z = out.shape
            out = out.view(b_orig*c_orig,1,h,w,z)
            target = target.view(b_orig*c_orig,1,h,w,z)
            out = out.repeat(1,self.channel_dim,1,1,1)
            target = target.repeat(1,self.channel_dim,1,1,1)
        # Axial view
        if 'axial' in self.views:
            # make 3D->2D (axial view)
            b,c,h,w,z = out.shape
            o = out.permute(0,4,1,2,3).reshape(b*z,c,h,w)
            t = target.permute(0,4,1,2,3).reshape(b*z,c,h,w)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_z,c,h,w = o_features[key].shape
                z = b_z // b
                o_feat = o_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,z,c,h,w).permute(0,2,3,4,1) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
        # sagittal view
        if 'sagittal' in self.views:
            # make 3D->2D (sagittal view)
            b,c,h,w,z = out.shape
            o = out.permute(0,3,1,2,4).reshape(b*w,c,h,z)
            t = target.permute(0,3,1,2,4).reshape(b*w,c,h,z)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_w,c,h,z = o_features[key].shape
                w = b_w // b
                o_feat = o_features[key].reshape(b,w,c,h,z).permute(0,2,3,1,4) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,w,c,h,z).permute(0,2,3,1,4) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
        if 'coronal' in self.views:
            # Coronal view
            # make 3D->2D (coronal view)
            b,c,h,w,z = out.shape
            o = out.permute(0,2,1,3,4).reshape(b*h,c,w,z)
            t = target.permute(0,2,1,3,4).reshape(b*h,c,w,z)
            # make channel dimension feasible to perception model
            o_features = self.feature_extractor(o)
            t_features = self.feature_extractor(t)
            for key in o_features.keys():
                b_h,c,w,z = o_features[key].shape
                h = b_h // b
                o_feat = o_features[key].reshape(b,h,c,w,z).permute(0,2,1,3,4) # make b,c,h,w,z again
                t_feat = t_features[key].reshape(b,h,c,w,z).permute(0,2,1,3,4) # make b,c,h,w,z again
                loss += self.loss_fn(o_feat, t_feat) / self.base_weight
            return loss / len(self.views)

class SpectralLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(SpectralLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # perform 2D DFT (real-to-complex, orthonormalization)
        channel_dim = 1
        batch_dim = 0
        axes = [i for i in range(1, len(x.shape)) if i != channel_dim and i != batch_dim]
        freq = torch.fft.fftn(x, dim = axes, norm = 'ortho')
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None, reduction = True):
        #### option 1
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        
        # ### option 2
        # # frequency distance using (squared) Euclidean distance
        # weight_matrix = 1
        # magnitude_x = torch.sqrt(torch.einsum('bchwzi,bchwzi->bchwz', recon_freq, recon_freq))
        # magnitude_y = torch.sqrt(torch.einsum('bchwzi,bchwzi->bchwz', real_freq, real_freq))
        # matrix_tmp = (magnitude_x * magnitude_y)
        # matrix_tmp = 1 / (matrix_tmp)
        # matrix_tmp[torch.isnan(matrix_tmp)] = 0
        # dot_product = torch.einsum('bchwzi,bchwzi->bchwz', recon_freq, real_freq)
        # theta = dot_product * matrix_tmp
        # freq_distance = (recon_freq**2).sum(dim=-1) + (real_freq**2).sum(dim=-1) - 2 * (recon_freq**2).sum(dim=-1).sqrt() * (real_freq**2).sum(dim=-1).sqrt() * theta
        # # complex_spectrum = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        if reduction:
            return torch.mean(loss)
        else:
            return loss

    def forward(self, pred, target, matrix=None, reduction = True, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix, reduction) * self.loss_weight

class DiscLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self, out, target):
        loss_fake = F.binary_cross_entropy_with_logits(out, torch.zeros_like(out))
        loss_real = F.binary_cross_entropy_with_logits(target, torch.ones_like(target))
        return loss_fake + loss_real

class AdvLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self, out, target = None):
        return F.binary_cross_entropy_with_logits(out, torch.ones_like(out))

class MultiresLossWrapper(nn.Module):
    def __init__(
        self,
        loss_module: nn.Module,
        weights: dict
    ):
        super().__init__()
        self.loss_module = loss_module
        self.weights = weights
        self.res_keys = [key for key,val in weights.items() if val > 0]
    def forward(self, out, target):
        out = {'level_0': out} if isinstance(out, torch.Tensor) else out
        target = {'level_0': target} if isinstance(target, torch.Tensor) else target
        # add this for adversarial loss
        if target is None:
            target = {key:None for key in out.keys()}
        loss = {}
        for res_key in out.keys():
            loss[res_key] = 0
            if self.weights[res_key] > 0:
                loss[res_key] = self.loss_module(out[res_key], target[res_key]) * self.weights[res_key]
        return loss

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights = torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights = torchvision.models.resnet.ResNet152_Weights)
    elif model_name == 'inception_v3':
        pretrained = torchvision.models.inception_v3(weights = torchvision.models.inception.Inception_V3_Weights)
    return pretrained
    
def define_loss_fn(loss_name):
    if loss_name == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_name in ('mse', 'l2'):
        loss_fn = nn.MSELoss()
    elif loss_name in 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn

def define_loss(loss_type, opt):
    params = opt['params']
    res_weight = opt['res_weight']
    if loss_type in 'pixel':
        # Load loss module
        loss_module = define_loss_fn(params['loss_fn'])
    elif loss_type == 'perception':
        # Load loss configurations
        model_name = 'vgg19'
        return_nodes = params['return_nodes']
        loss_fn = define_loss_fn(params['loss_fn'])
        channel_dim = 3
        spatial_dims = 3
        separate_channel = True
        base_weight = len(return_nodes)
        # Load perception feature extractor
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        if spatial_dims == 3:
            loss_module = PerceptionLoss3D(feature_extractor, loss_fn, channel_dim = channel_dim, separate_channel = separate_channel, base_weight = base_weight)
        else:
            loss_module = PerceptionLoss2D(feature_extractor, loss_fn, channel_dim, **perception_loss_params)
    elif loss_type == 'disc':
        loss_module = DiscLoss()
    elif loss_type == 'adv':
        loss_module = AdvLoss()
    elif loss_type == 'ssim':
        loss_module = SSIMLoss(**params)
    elif loss_type == 'spectral':
        loss_module = SpectralLoss(**params)
    return MultiresLossWrapper(loss_module, res_weight)