import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

class PerceptionLoss3D(nn.Module):
    def __init__(
        self,
        model_name: str,
        return_nodes: (tuple, list), 
        loss_fn = nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        views: (tuple, list) = ['axial'],#['axial', 'sagittal', 'coronal']
        weight_nodes: (tuple, list) = None
    ):
        super().__init__()
        self.views = views
        self.loss_fn = loss_fn
        self.feature_extractor = create_feature_extractor(define_pretrained(model_name).eval(), return_nodes)
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
        self.return_nodes = return_nodes
        self.weight_nodes = {n:w / len(self.return_nodes) for n, w in zip(return_nodes, weight_nodes)} if weight_nodes is not None else {node: 1/len(self.return_nodes) for node in return_nodes}
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
        if 'axial' in self.views:
            # Axial view
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
                loss += self.loss_fn(o_feat, t_feat) * self.weight_nodes[key]
        if 'sagittal' in self.views:
            # sagittal view
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
                loss += self.loss_fn(o_feat, t_feat) * self.weight_nodes[key]
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
                loss += self.loss_fn(o_feat, t_feat) * self.weight_nodes[key]
        return loss / len(self.views) # 3 views
    
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
        loss_fn: str = 'gan', # gan or wgan
        relativistic: bool = False,
        logit: bool = True
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.relativistic = relativistic
        self.logit = logit
    def forward(self, out, target):
        if self.relativistic:
            if self.logit:
                loss_fake = F.binary_cross_entropy_with_logits(out - target.mean(0, keepdim = True), torch.zeros_like(target))
                loss_real = F.binary_cross_entropy_with_logits(target - out.mean(0, keepdim = True), torch.ones_like(target))
            else:
                loss_fake = F.binary_cross_entropy(out - target.mean(0, keepdim = True), torch.zeros_like(target))
                loss_real = F.binary_cross_entropy(target - out.mean(0, keepdim = True), torch.ones_like(target))
            return loss_real + loss_fake
        elif self.loss_fn == 'gan':
            if self.logit:
                loss_fake = F.binary_cross_entropy_with_logits(out, torch.zeros_like(out))
                loss_real = F.binary_cross_entropy_with_logits(target, torch.ones_like(target))
            else:
                loss_fake = F.binary_cross_entropy(out, torch.zeros_like(out))
                loss_real = F.binary_cross_entropy(target, torch.ones_like(target))
            return loss_fake + loss_real
        elif self.loss_fn == 'wgan':
            if self.logit: # meaning sigmoid function is not applied
                return -1 * (torch.mean(torch.sigmoid(target)) - torch.mean(torch.sigmoid(out)))
            else:
                return -1 * (torch.mean(target) - torch.mean(out))

class AdvLoss(nn.Module):
    def __init__(
        self,
        loss_fn: str = 'gan', # gan or wgan
        relativistic: bool = False,
        logit: bool = True
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.relativistic = relativistic
        self.logit = logit
    def forward(self, out, target = None):
        if self.relativistic:
            if self.logit:
                return F.binary_cross_entropy_with_logits(out - target.mean(0, keepdim = True), torch.ones_like(target))
            else:
                return F.binary_cross_entropy(out - target.mean(0, keepdim = True), torch.ones_like(target))
        elif self.loss_fn == 'gan':
            if self.logit:
                return F.binary_cross_entropy_with_logits(out, torch.ones_like(out))
            else:
                return F.binary_cross_entropy(out, torch.ones_like(out))
        else:
            if self.logit:
                return - torch.mean(torch.sigmoid(out))
            else:
                return - torch.mean(out)

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
    if loss_type in 'pixel':
        # Load loss module
        loss_module = define_loss_fn(params['loss_fn'])
    elif loss_type == 'perception':
        # Load loss configurations
        model_name = 'vgg19'
        return_nodes = params['return_nodes']
        weight_nodes = params.get('weight_nodes')
        loss_fn = define_loss_fn(params['loss_fn'])
        views = params.get('views')
        channel_dim = 3
        spatial_dims = 3
        separate_channel = True
        # Load perception feature extractor
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        if spatial_dims == 3:
            loss_module = PerceptionLoss3D(model_name, return_nodes, loss_fn, channel_dim = channel_dim, separate_channel = separate_channel, views = views, weight_nodes = weight_nodes)
    elif loss_type == 'disc':
        loss_module = DiscLoss(**params)
    elif loss_type == 'adv':
        loss_module = AdvLoss(**params)
    elif loss_type == 'ssim':
        loss_module = SSIMLoss(**params)
    elif loss_type == 'spectral':
        loss_module = SpectralLoss(**params)
    return loss_module