import os, glob, nibabel as nib, numpy as np, copy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import monai
from monai.transforms import (
    Transform,
    MapTransform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    CropForegroundd,
    ForegroundMaskd,
    Spacingd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Lambdad,
    DeleteItemsd,
    Randomizable, 
    MultiSampleTrait,
    RandWeightedCropd,
    RandSpatialCropSamplesd,
    RandCropByLabelClassesd,
    Invertd
)

def kspace_crop(x, scale_factors: 'Sequence[int, float]', rescale: bool = False):
    fft = torch.fft.fftshift(torch.fft.fftn(x))
    # crop (truncate) fft
    center = [s // 2 for s in x.shape[1:]]
    scale_factors = [scale_factors for _ in range(len(center))] if isinstance(scale_factors, (int, float)) else scale_factors
    roi_starts = [c - int(c*s) for c,s in zip(center, scale_factors)]
    roi_ends = [c + int(c*s) for c,s in zip(center, scale_factors)]
    roi_ends = [roi_ends[i]+1 if roi_ends[i] == roi_starts[i] else roi_ends[i] for i in range(len(roi_ends))]
    if len(roi_starts) == 2:
        fft = fft[:,roi_starts[0]:roi_ends[0], roi_starts[1]:roi_ends[1]]
    elif len(roi_starts) == 3:
        fft = fft[:,roi_starts[0]:roi_ends[0], roi_starts[1]:roi_ends[1], roi_starts[2]:roi_ends[2]]
    x_downsized = torch.abs(torch.fft.ifftn(torch.fft.fftshift(fft)))
    if rescale:
        x_downsized = ((x_downsized - x_downsized.min()) / (x_downsized.max() - x_downsized.min() + 1e-5)) * (x.max() - x.min()) + x.min() # rescale the intensity
    return x_downsized

def kspace_zeropad(x, scale_factors: 'Sequence[int, float]', resize: bool = False):
    fft = torch.fft.fftshift(torch.fft.fftn(x))
    # crop (truncate) fft
    center = [s // 2 for s in x.shape[1:]]
    scale_factors = [scale_factors for _ in range(len(center))] if isinstance(scale_factors, (int, float)) else scale_factors
    roi_starts = [c - int(c*s) for c,s in zip(center, scale_factors)]
    roi_ends = [c + int(c*s) for c,s in zip(center, scale_factors)]
    # zero-pad
    if len(roi_starts) == 2:
        for roi_start, roi_end in zip(roi_starts[::-1], roi_ends[::-1]):
            fft = fft.permute(0,2,1)
            if roi_end - roi_start == 0:
                continue
            fft[:,:roi_start] = 0
            fft[:,roi_end:] = 0
    elif len(roi_starts) == 3:
        for roi_start, roi_end in zip(roi_starts[::-1], roi_ends[::-1]):
            fft = fft.permute(0,3,1,2)
            if roi_end - roi_start == 0:
                continue
            fft[:,:roi_start] = 0
            fft[:,roi_end:] = 0
    x_downsized = torch.abs(torch.fft.ifftn(torch.fft.fftshift(fft)))
    if resize:
        x_downsized = F.interpolate(x_downsized.unsqueeze(0), scale_factor = scale_factors, mode = 'nearest').squeeze(0)
    return x_downsized

def lowres_transform(x, scale_factors: 'Sequence[int, float]', slide_depth: bool = True, transform: str = 'kspace_zeropad', rescale: bool = False, resize: bool = True):
    c,h,w,z = x.shape
    if scale_factors[-1] != 1 and slide_depth:
        slide_depth_int = int(1/scale_factors[-1])
        x = x[:,:,:,::slide_depth_int]
        scale_factors = [*scale_factors[:2], 1]
    if transform == 'kspace_crop':
        return kspace_crop(x, scale_factors, rescale = rescale)
    elif transform == 'kspace_zeropad':
        return kspace_zeropad(x, scale_factors, resize = resize)
    elif transform == 'trilinear':
        return F.interpolate(x.unsqueeze(0), scale_factor=scale_factors, mode = transform)

class MultiresTargetTransformd(MapTransform):
    def __init__(
        self,
        keys: list,
        scale_factors: 'Sequence[list, tuple]',
        allow_missing_keys: bool = False,
        downsize_type: str = 'kspace_crop', # kspace, bilinear, trilinear, nearest
        prefix: str = 'target',
        slide_depth: bool = True
    ):
        super().__init__(keys, scale_factors)
        assert isinstance(scale_factors, (list, tuple)), 'scale_factors should be list or tuple'
        assert isinstance(scale_factors[0], (list, tuple)), 'all of the elements in scale_factors should be list or tuple'
        self.scale_factors = scale_factors
        self.prefix = prefix
        self.downsize_type = downsize_type
        self.slide_depth = slide_depth
    def __call__(self, x):
        x = dict(x)
        for key in self.keys:
            for i in range(len(self.scale_factors)):
                res_key = f'{self.prefix}_{i+1}'
                scale = self.scale_factors[i]
                x[f'{res_key}'] = lowres_transform(x[key], scale, slide_depth = self.slide_depth, transform = self.downsize_type, resize = True, rescale = True) # when downsize_type == 'kspace_zeropad', resize the image, when downsize_type == 'kspace_crop', do not rescale image
            x[f'{self.prefix}_0'] = x[key]
        return x

class InputTransform(Transform):
    def __init__(
        self,
        source_key: str = 'image',
        output_key: str = 'input',
        scale_factor: (tuple, list) = None,
        target_level: int = None
    ):
        assert scale_factor is not None, 'scale_factor should be given'
        assert target_level is not None, 'target_level should be given'
        self.source_key = source_key
        self.output_key = output_key
        self.scale_factor = scale_factor
        self.target_level = target_level
    def __call__(self, x):
        x = dict(x)
        x[f"{self.output_key}_{self.target_level}"] = lowres_transform(x[self.source_key], self.scale_factor, slide_depth = False, transform = 'kspace_zeropad', resize = True, rescale = False)
        return x

class MultiresInputTransformd(Transform):
    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_level: int,
        source_level: int,
        interp_type: str = 'trilinear', # kspace, bilinear, trilinear, nearest
    ):
        self.input_key = input_key
        self.target_key = target_key
        self.interp_type = interp_type
        self.source_level = source_level
        self.num_level = num_level
    def __call__(self, x):
        x = dict(x)
        source_key = f'{self.input_key}_{self.source_level}'
        for i in range(self.num_level):
            res_key = f'{self.input_key}_{i}'
            target_key = f'{self.target_key}_{i}'
            if res_key == source_key:
                continue
            x[res_key] = F.interpolate(x[source_key].unsqueeze(0), size = x[target_key].shape[1:], mode = self.interp_type).squeeze(0)
        return x

class CollectIntensityScaleInfo(MapTransform):
    def __init__(
        self,
        keys: list,
        b_range: (tuple, list),
        a_range: (tuple, list) = None,
        allow_missing_keys = True
    ):
        super().__init__(keys, allow_missing_keys)
        self.b_range = b_range
        self.a_range = a_range
    def __call__(self, x):
        x = dict(x)
        for key in self.keys:
            x[f'{key}_meta_dict']['min_intensity_a'] = self.a_range[0] if self.a_range is not None else x[key].min()
            x[f'{key}_meta_dict']['max_intensity_a'] = self.a_range[1] if self.a_range is not None else x[key].max()
            x[f'{key}_meta_dict']['min_intensity_b'] = self.b_range[0]
            x[f'{key}_meta_dict']['max_intensity_b'] = self.b_range[1]
        return x

class InverseIntensityScale(MapTransform):
    def __init__(
        self,
        keys: list,
        allow_missing_keys: bool = True
    ):
        super().__init__(keys, allow_missing_keys)
    def __call__(self, x):
        x = dict(x)
        for key in self.keys:
            a_min = x[f'{key}_meta_dict']['min_intensity_a']
            a_max = x[f'{key}_meta_dict']['max_intensity_a']
            b_min = x[f'{key}_meta_dict']['min_intensity_b']
            b_max = x[f'{key}_meta_dict']['max_intensity_b']
            x[key] = ((x[key] - b_min) / (b_max - b_min) * (a_max - a_min)) + a_min
        return x

class RandMultiresWeightedCropd(MapTransform, Randomizable, MultiSampleTrait):
    def __init__(
        self,
        keys: 'list',
        w_key: 'str',
        spatial_size:'Sequence[int] | int',
        inverse_scale_factors: 'Sequence[tuple, list]',
        num_samples: int = 1,
        allow_missing_keys: 'bool' = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.w_key = w_key
        # spatial_size = [patch_size * scale_factor for patch_size, scale_factor in zip(spatial_size, scale_factors[-1])]
        self.spatial_size = spatial_size
        self.spatial_dims = len(self.spatial_size)
        # self.inverse_scale_factors = inverse_scale_factors
        # accumulated inverse scale factor
        temp = inverse_scale_factors[::-1]
        self.relative_scales = [[int(temp[i][idx] * np.prod([temp[j][idx] for j in range(i)])) for idx in range(self.spatial_dims)] for i in range(len(temp))][::-1] + [[1 for i in range(self.spatial_dims)]]
        # self.relative_inverse_scale_factors = [[1/s for s in self.scale_factors[-1]]] + [[hr_s / lr_s for hr_s, lr_s in zip(list_scale_factor, self.scale_factors[-1])] for list_scale_factor in self.scale_factors]
        self.num_samples = num_samples
        # self.sampler = RandWeightedCropd(keys = self.keys, w_key = self.w_key, spatial_size = self.spatial_size, num_samples = num_samples, allow_missing_keys = allow_missing_keys)
        self.sampler = RandCropByLabelClassesd(keys = self.keys, label_key = self.w_key, spatial_size = self.spatial_size, ratios = [0,1.0], num_classes = 2, num_samples = num_samples, allow_missing_keys = allow_missing_keys)
    def select_patch(self, x, center, patch_size):
        patch_size = [p if p > 2 else 2 for p in patch_size]
        if len(patch_size) == 3:
            return x[:,max(0, center[0] - patch_size[0] // 2): center[0] + patch_size[0] // 2, max(0, center[1] - patch_size[1] // 2): center[1] + patch_size[1] // 2, max(0, center[2] - patch_size[2] // 2): center[2] + patch_size[2] // 2]
        if len(patch_size) == 2:
            return x[:,center[0] - patch_size[0] // 2: center[0] + patch_size[0] // 2, center[1] - patch_size[1] // 2: center[1] + patch_size[1] // 2]
    def __call__(self, data):
        ret: list = [dict(data) for _ in range(self.sampler.cropper.num_samples)]
        # get random centers
        # self.sampler.randomize(weight_map = data[self.w_key])
        self.sampler.randomize(label = data[self.w_key])
        centers = self.sampler.cropper.centers
        # deep copy all the unmodified data
        for i in range(self.sampler.cropper.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                ret[i][key] = copy.deepcopy(data[key])

        for key in self.key_iterator(data):
            for i, center in enumerate(centers):
                level = int(key.split('_')[-1]) # key should have for format as "{something}_{level}"
                relative_scales = self.relative_scales[level]
                patch_size = [int(p * s) for s, p in zip(relative_scales, self.spatial_size)]
                center_scaled = [int(c * s) for s, c in zip(relative_scales, center)]
                ret[i][key] = self.select_patch(data[key], center_scaled, patch_size)
        return ret

class MinimumPadd(MapTransform):
    def __init__(
        self,
        keys: list,
        roi_size: 'Sequence[list, tuple]',
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.roi_size = roi_size
    def __call__(self, x):
        x = dict(x)
        for key in self.keys:
            if any([roi > shape for roi, shape in zip(self.roi_size, x[key].shape[-3:])]):
                padsize = [[0,0] if shape >= roi else [roi-shape,0] for roi, shape in zip(self.roi_size, x[key].shape[-3:])]
                padsize = [pad for list_pad in padsize for pad in list_pad][::-1]
                x[key] = F.pad(x[key], padsize)
                x[f"{key}_meta_dict"]['min_padsize'] = padsize
            # for i in range(len(self.scale_factors)):
            #     res_key = f'{self.prefix}_{i+1}'
            #     scale = self.scale_factors[i]
            #     x[f'{res_key}'] = lowres_transform(x[key], scale, slide_depth = self.slide_depth, transform = self.downsize_type)
            # x[f'{self.prefix}_0'] = x[key]
        return x

class ValidPadMultiscale(MapTransform):
    def __init__(
        self,
        keys: list,
        scale_factors: 'Sequence[list, tuple]',
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert isinstance(scale_factors[0], (tuple, list)), 'scale_factors should be list of list'
        # define accumulated scale factors
        spatial_dims = len(scale_factors[0])
        self.acc_scale_factors = [np.prod([f[s] for f in scale_factors]) for s in range(spatial_dims)]
    def __call__(self, x):
        x = dict(x)
        for key in self.keys:
            padsize = []
            for factor, shape in zip(self.acc_scale_factors, x[key].shape[-3:]):
                if factor != 1:
                    # valid resolution
                    valid_res = shape + shape % factor
                    # make smallest resolution as even
                    small_res = int(np.ceil(valid_res / factor))
                    if small_res % 2 != 0:
                        small_res += 1
                    valid_res = small_res * factor
                    padsize.append([valid_res - shape, 0])
                else:
                    padsize.append([0,0])
            if sum([p for pad in padsize for p in pad]) == 0:
                continue
            padsize = [pad for list_pad in padsize for pad in list_pad][::-1]
            x[key] = F.pad(x[key], padsize)
            x[f"{key}_meta_dict"]['valid_padsize'] = padsize
            # for i in range(len(self.scale_factors)):
            #     res_key = f'{self.prefix}_{i+1}'
            #     scale = self.scale_factors[i]
            #     x[f'{res_key}'] = lowres_transform(x[key], scale, slide_depth = self.slide_depth, transform = self.downsize_type)
            # x[f'{self.prefix}_0'] = x[key]
        return x

class InversePad(Transform):
    def __init__(
        self,
        keys_image: list,
        keys_meta: list,
    ):
        self.keys_image = keys_image
        self.keys_meta = keys_meta
    def __call__(self, x):
        x = dict(x)
        for image_key, meta_key in zip(self.keys_image, self.keys_meta):
            for padtype in ['valid_padsize', 'min_padsize']:
                if padtype not in x[meta_key]:
                    continue
                inverse_pad = [-p for p in x[meta_key][padtype]]
                x[image_key] = F.pad(x[image_key], inverse_pad)
        return x

class Ensure3Dimd(MapTransform):
    def __init__(
        self,
        keys: list,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
    def __call__(self, x):
        x = dict(x)
        for key in self.keys:
            if len(x[key].shape) == 3: # if data is 2D (c,h,w)
                x[key] = x[key].unsqueeze(-1) # (c,h,w,1)
        return x
def define_transforms(opts: dict):
    """
    Args:
        opts (dict): configuration information
    Returns:
        train_pre_trans: pre-processing transforms for training
        eval_pre_trans: pre-processing transforms for evaluation
        eval_post_trans: post-processing transforms for evaluation
        (x) infer_pre_trans: pre-processing transforms for inference
        (x) infer_post_trans: post-processing transforms for inference
        cache_trans: base transform for cache
    """
    ### Load configurations
    start_pixdim = opts['data_opt']['start_pixdim']
    patch_size = opts['train_opt']['patch_size']
    inverse_scale_factors = opts['data_opt']['inverse_scale_factors']
    slide_depth = opts['data_opt']['slide_depth']
    image_key = opts['data_opt']['image_key']
    # downsize_type_target = opts['exp_opt']['downsize_type_target'] # this is fixed to kspace_crop
    target_scale_level = opts['data_opt']['target_scale_level']
    list_target_pixdim = []
    temp = start_pixdim
    for i in range(len(inverse_scale_factors)):
        list_scale_factor = inverse_scale_factors[i]
        temp = [t/s for t, s in zip(temp, list_scale_factor)]
        list_target_pixdim.append(temp)
    spacing_interp_mode = 'bilinear' if len(start_pixdim) == 2 else 'trilinear'
    scale_factors = [[spacing_target/spacing_start for spacing_target, spacing_start in zip(list_spacing_target, start_pixdim)] for list_spacing_target in list_target_pixdim]
    num_res_level = len(scale_factors)+1
    list_out_keys = [f'input_{i}' for i in range(num_res_level)] + [f'target_{i}' for i in range(num_res_level)]
    list_target_keys = [f'target_{i}' for i in range(num_res_level)]
    num_patch = opts['train_opt']['num_patch']
    # Define intensity preprocessing/ postprocessing
    preprop_intensity = opts['data_opt']['preprop_intensity']
    if preprop_intensity['type'] == 'minmax':
        # preprop_intensity_info = CollectIntensityScaleInfo(keys = [image_key], b_range = [0,1])
        preprop_intensity = ScaleIntensityd(keys = [image_key], minv = preprop_intensity['minv'], maxv = preprop_intensity['maxv'])
    elif preprop_intensity['type'] == 'minmax_range':
        # preprop_intensity_info = CollectIntensityScaleInfo(keys = [image_key], b_range = (preprop_intensity['b_min'], preprop_intensity['b_max']), a_range = (preprop_intensity['a_min'], preprop_intensity['a_max']))
        preprop_intensity = ScaleIntensityRanged(keys = [image_key], a_min = preprop_intensity['a_min'], a_max = preprop_intensity['a_max'], b_min = preprop_intensity['b_min'], b_max = preprop_intensity['b_max'])
    input_scale_factor = [1/np.prod([s[i] for s in inverse_scale_factors[::-1][:target_scale_level]]) for i in range(len(start_pixdim))]
    """
    Order of pre-processing transforms:
        1. Load image
        2. Orientation
        3. Crop foreground
        4. spacing
        5. inteisity scale info
        5. intensity scaling/normalization
        6. minimumpad to the patch size
        7. validpad - valid divisible size to the scale
        
    Order of post-processing transforms:
        1. Inverse pad - from validpad & minimumpad (6,7)
        2. inverse intensity scaling/normalization
        3. inverse spacing
        4. inverse foregroundcrop
        5. inverse orientation
        6. (optional) save image
    """
    ### define list of base trans modules
    list_base_trans = [
        LoadImaged(keys = [image_key]),
        EnsureChannelFirstd(keys = [image_key]),
        Ensure3Dimd(keys = [image_key]),
        Orientationd(keys = [image_key], axcodes = 'RAS'),
        CropForegroundd(keys = [image_key], source_key = image_key),
        Spacingd(keys = [image_key], pixdim = start_pixdim, mode = spacing_interp_mode),
        # preprop_intensity_info,
        preprop_intensity,
        MinimumPadd(keys = [image_key], roi_size = patch_size),
        ValidPadMultiscale(keys = [image_key], scale_factors = inverse_scale_factors),
        MultiresTargetTransformd(keys = [image_key], scale_factors = scale_factors, downsize_type='kspace_crop', slide_depth=False, prefix = 'target'),
        InputTransform(source_key = image_key, output_key = 'input', scale_factor = scale_factors[target_scale_level-1], target_level = target_scale_level),
        MultiresInputTransformd(input_key = 'input', target_key = 'target', num_level = len(scale_factors)+1, interp_type = 'trilinear', source_level = target_scale_level),
        ForegroundMaskd(keys = [f'target_{len(scale_factors)}'], new_key_prefix = 'foreground_'),
        Lambdad(keys = [f'foreground_target_{len(scale_factors)}'], func = lambda x: 1-x, inv_func = lambda x: x), # somehow foreground is masked as 0
        DeleteItemsd(keys = [image_key]),
    ]
    ### Define cache transforms
    trans_cache = Compose(list_base_trans)
    ### Define eval transforms
    trans_pre_eval = Compose(list_base_trans + [
        DeleteItemsd(keys = [f'foreground_target_{len(scale_factors)}'])
    ])
    # postprop_intensity_eval = InverseIntensityScale(keys = [image_key, 'pred'])
    trans_post_eval = Compose([
        InversePad(keys_image = [image_key, 'pred'], keys_meta = [f'{image_key}_meta_dict', 'pred_meta_dict']),
        # postprop_intensity_eval,
        Invertd(
            keys = ['pred', image_key],
            orig_keys = [image_key, image_key], 
            meta_keys = [f'{image_key}_meta_dict', f'{image_key}_meta_dict'],
            orig_meta_keys = [f'{image_key}_meta_dict', f'{image_key}_meta_dict'],
            transform = trans_pre_eval,
            nearest_interp = False,
            to_tensor = True,
            device = 'cpu'
        )
    ])
    ### Define train transforms
    lowres_patch_size = [int(p / np.prod([scale[idx] for scale in inverse_scale_factors])) for idx, p in enumerate(patch_size)]
    trans_pre_train = Compose(list_base_trans + [
        RandMultiresWeightedCropd(keys = list_out_keys, w_key = f'foreground_target_{len(scale_factors)}', spatial_size = lowres_patch_size, inverse_scale_factors = inverse_scale_factors, num_samples = num_patch),
        DeleteItemsd(keys = [f'foreground_target_{len(scale_factors)}'])
    ])
    return trans_pre_train, trans_pre_eval, trans_post_eval, trans_cache