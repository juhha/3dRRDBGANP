from codes.benchmarks.model_utils import define_network
from codes.benchmarks.loss_utils import define_loss
from codes.data_utils import define_transforms
from codes.eval_metrics import calc_mae, calc_psnr, calc_ssim, calc_ldf, calc_ldf_torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from monai.data import Dataset, PersistentDataset, DataLoader, decollate_batch, CacheDataset
from monai.inferers import sliding_window_inference

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

import os, json, yaml, glob, nibabel as nib, numpy as np, tqdm, re, argparse, copy, re, random
from PIL import Image
import matplotlib.pyplot as plt
import wandb

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type = str, default = 'brain_mice', dest = 'exp_name',
                        help = 'yaml file name (without extension)')
    parser.add_argument('--device', type = str, default = 'cuda', dest = 'device',
                        help = 'device name')
    parser.add_argument('--cv', type = int, default = 5, dest = 'cv',
                        help = '# of cross validation')
    parser.add_argument('--fold', type = int, default = 0, dest = 'fold',
                        help = 'i-th fold in cross validation')
    parser.add_argument('--num_workers', type = int, default = 8, dest = 'num_workers',
                        help = 'number of workers')
    parser.add_argument('--persistent_cache', action = 'store_true', dest = 'persistent_cache',
                        help = 'use persistent cache for data iteration')
    parser.add_argument('--memory_cache', action = 'store_true', dest = 'memory_cache',
                        help = 'use memory cache for data')
    parser.add_argument('--progress_report', type = str, default = 'pbar', dest = 'progress_report',
                        help = 'type of progress report (pbar, none, print)')
    parser.add_argument('--save_iter', type = int, default = 100, dest = 'save_iter',
                        help = 'save and evaluation iteration')
    parser.add_argument('--wandb_override', action = 'store_true', dest = 'wandb_override',
                        help = 'override wandb run or not')
    parser.add_argument('--mixed_precision', action = 'store_true', dest = 'mixed_precision',
                        help = 'use mixed precision or not')
    return parser

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ############
    # get file path
    base_dir = os.path.dirname(__file__) # base directory
    data_dir = os.path.join(base_dir, 'data') # data directory
    ############
    # Configurations
    # get user arguments
    parser = get_args()
    args = parser.parse_args()
    config_name = args.exp_name
    device = args.device
    cv = args.cv
    fold = args.fold
    mixed_precision = args.mixed_precision
    num_workers = args.num_workers
    persistent_cache = args.persistent_cache
    memory_cache = args.memory_cache
    progress_report = args.progress_report
    save_iter = args.save_iter
    # save file path
    save_dir = os.path.join(base_dir, 'checkpoint', f'{config_name}', f'fold_{fold}')
    save_dir_cp = os.path.join(save_dir, 'state_dict')
    save_dir_progress = os.path.join(save_dir, 'progress')
    save_dir_progress_train = os.path.join(save_dir, 'progress', 'train')
    save_dir_progress_monitor = os.path.join(save_dir, 'progress', 'monitor')
    save_dir_image = os.path.join(save_dir, 'progress', 'monitor_nifti')
    # get configurations
    opts = yaml.safe_load(open(os.path.join(base_dir, f'options/{config_name}.yaml')))
    save_config = copy.deepcopy(opts)
    source = opts['data_opt']['source']
    batch_size = opts['train_opt']['batch_size']
    max_epochs = opts['train_opt']['max_epochs']
    step_size = opts['train_opt']['step_size']
    image_key = opts['data_opt']['image_key']
    start_level = opts['train_opt']['start_level']
    inverse_scale_factors = opts['data_opt']['inverse_scale_factors']
    # get meta file
    meta = json.load(open(os.path.join(data_dir, 'meta', f'{source}.json'), 'r'))
    # get transforms
    trans_pre_train, trans_pre_eval, trans_post_eval, trans_cache = define_transforms(opts)
    # get files
    list_files = [{image_key: os.path.join(data_dir, f[image_key])} for f in meta['data']]
    list_files_dev = [f for idx, f in enumerate(list_files) if idx % cv != fold]
    list_files_test = [f for idx, f in enumerate(list_files) if idx % cv == fold]
    list_files_train = list_files_dev
    list_files_monitor = list_files_test[:3]
    print(f"# of files: {len(list_files):,}")
    print(f"# of training files: {len(list_files_train)}")
    ############
    # Wandb config
    if progress_report == 'wandb':
        resume = None if args.wandb_override else 'allow'
        wandb.init(
            id = f'{config_name}_{source}_{fold}',
            project = f'ISBI-SR-benchmark', 
            resume = resume,
            config = {
                'source': source,
                'config_name': config_name,
                'fold': fold,
            }
        )
    ############
    # Load dataloader
    # FIX - I don't know why this is needed, but without this, I have error
    torch.multiprocessing.set_sharing_strategy('file_system')
    if memory_cache:
        ds_train = CacheDataset(list_files_train, trans_pre_train, cache_rate = 1.0, progress = True, num_workers = num_workers)
        ds_eval = CacheDataset(list_files_test, trans_pre_eval, cache_rate = 1.0, progress = True, num_workers = num_workers)
        ds_monitor = CacheDataset(list_files_monitor, trans_pre_eval, cache_rate = 1.0, progress = True, num_workers = num_workers)
    elif persistent_cache:
        cache_dir = os.path.join(data_dir, 'persistent', f'{config_name}')
        hash_func = lambda x: x[image_key].replace(data_dir, '').strip('/').strip('.nii.gz').replace('/', '@').encode()
        if not os.path.exists(cache_dir):
            print('Generating persistent cache files')
            ds = PersistentDataset(list_files, trans_cache, cache_dir = cache_dir, hash_func = hash_func)
            dl = DataLoader(ds, num_workers = num_workers, batch_size = 1)
            length = len(dl)
            percentage = 0
            for idx, _ in enumerate(dl):
                if (idx + 1) / length > percentage:
                    percentage += 0.01
                    print(f'{int((idx + 1) / length * 100):.0f}% loaded.')
            print("Persistent cache generated.")
        ds_train = PersistentDataset(list_files_train, trans_pre_train, cache_dir = cache_dir, hash_func = hash_func)
        ds_eval = PersistentDataset(list_files_test, trans_pre_eval, cache_dir = cache_dir, hash_func = hash_func)
        ds_monitor = PersistentDataset(list_files_monitor, trans_pre_eval, cache_dir = cache_dir, hash_func = hash_func)
    else:
        ds_train = Dataset(list_files_train, trans_pre_train)
        ds_eval = Dataset(list_files_test, trans_pre_eval)
        ds_monitor = Dataset(list_files_monitor, trans_pre_eval)
    # generate random sampler
    sampler = RandomSampler(ds_train, replacement = True, num_samples = step_size)
    dl_train = DataLoader(ds_train, num_workers = num_workers, batch_size = batch_size, sampler = sampler)
    dl_eval = DataLoader(ds_eval, num_workers = num_workers, batch_size = 1)
    dl_monitor = DataLoader(ds_monitor, num_workers = num_workers, batch_size = 1)
    ############
    # Load model & optimizer
    net_g = define_network(opts['model_opt']['generator_type'], opts['model_opt']['generator']).to(device)
    net_d = define_network(opts['model_opt']['discriminator_type'], opts['model_opt']['discriminator']).to(device)
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr = 1e-4, weight_decay=0, betas=[0.9, 0.99])
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr = 1e-4, weight_decay=0, betas=[0.9, 0.99])
    scheduler_g = torch.optim.lr_scheduler.PolynomialLR(optimizer_g, total_iters = max_epochs, power = 1, verbose = False)
    scheduler_d = torch.optim.lr_scheduler.PolynomialLR(optimizer_d, total_iters = max_epochs, power = 1, verbose = False)
    precision = torch.float32
    if mixed_precision:
        scaler_g = torch.cuda.amp.GradScaler()
        scaler_d = torch.cuda.amp.GradScaler()
        precision = torch.float16
    def calc_num_params(net):
        num_params = 0 
        for p in net.parameters():
            num_params += np.prod(p.shape)
        return num_params
    print(f"# of params for generator: {calc_num_params(net_g):,}")
    print(f"# of params for discriminator: {calc_num_params(net_d):,}")
    ############
    # Load pretrained model if exists
    curr_epoch = 0
    progress = {'loss': [], 'monitor_eval': []}
    if os.path.exists(os.path.join(save_dir_cp, f'latest.pt')):
        state_dict = torch.load(os.path.join(save_dir_cp, f'latest.pt'), map_location = device)
        net_g.load_state_dict(state_dict['net_g'])
        net_d.load_state_dict(state_dict['net_d'])
        optimizer_g.load_state_dict(state_dict['optim_g'])
        optimizer_d.load_state_dict(state_dict['optim_d'])
        scheduler_g.load_state_dict(state_dict['scheduler_g'])
        scheduler_d.load_state_dict(state_dict['scheduler_d'])
        curr_epoch = state_dict['curr_epoch']
        progress = state_dict['progress']
        if mixed_precision:
            try:
                scaler_g.load_state_dict(state_dict['scaler_g'])
                scaler_d.load_state_dict(state_dict['scaler_d'])
            except:
                print('Scaler does not exist. Initializing new scaler.')
    state_dict = {
        'net_g': net_g.state_dict(),
        'net_d': net_d.state_dict(),
        'optim_g': optimizer_g.state_dict(),
        'optim_d': optimizer_d.state_dict(),
        'scheduler_g': scheduler_g.state_dict(),
        'scheduler_d': scheduler_d.state_dict(),
        'curr_epoch': curr_epoch,
        'progress': progress
    }
    if mixed_precision:
        state_dict['scaler_g'] = scaler_g.state_dict()
        state_dict['scaler_d'] = scaler_d.state_dict()
    ############
    # Define loss
    loss_opt = opts['loss_opt']
    # load weight and modules
    loss_module_dict = {}
    loss_weight_dict = {}
    for key in ['generator', 'discriminator']:
        loss_weight_dict[key] = {
            key: val['loss_weight']
            for key, val in loss_opt[key].items()        
        }
        loss_module_dict[key] = {
            key: define_loss(key, val).to(device)
            for key, val in loss_opt[key].items()        
        }
    ############
    # Train
    # configuration for inference
    num_patch = opts['train_opt']['num_patch']
    patch_size = opts['train_opt']['patch_size']
    print(f'Starting training at epoch {curr_epoch}.')
    # Make save directories
    for dirname in [save_dir_cp, save_dir_progress_train, save_dir_progress_monitor]:
        os.makedirs(dirname, exist_ok = True)
    yaml.safe_dump(save_config, open(os.path.join(save_dir, 'config.yaml'), 'w'))
    # iterate training
    for epoch in range(curr_epoch, max_epochs):
        # Train
        list_loss = {}
        if progress_report == 'pbar':
            pbar = tqdm.tqdm(total = len(dl_train), desc = f'train ({epoch}/{max_epochs})', position = 0)
        elif progress_report == 'wandb':
            print(f"Train for {epoch} / {max_epochs}")
        # dl_train.dataset.transform.transforms[-2].set_random_state(epoch)
        for batch in dl_train:
            lr = batch[f'input_{start_level}'].to(device)
            hr = batch[f'target_0'].to(device)
            with torch.cuda.amp.autocast(dtype = precision):
                ########
                # Update Discriminator
                # net_d.zero_grad()
                for p in net_d.parameters():
                    p.requires_grad = True
                if hasattr(net_g, 'sigmoid') and getattr(net_g, 'sigmoid'):
                    sr = net_g(lr).detach()
                else:
                    sr = net_g(lr).detach().clip(0,1)
                # added code for relativistic discriminator later
                if hasattr(net_d, 'relativistic') and getattr(net_d, 'relativistic'):
                    sr_out = net_d(sr, lr)
                    hr_out = net_d(hr, lr)
                else:
                    sr_out = net_d(sr)
                    hr_out = net_d(hr)
                dict_loss_d = {}
                loss_d = 0
                for loss_key in loss_module_dict['discriminator'].keys():
                    dict_loss_d[loss_key] = loss_module_dict['discriminator'][loss_key](sr_out, hr_out)
                    loss_d += dict_loss_d[loss_key] * loss_weight_dict['discriminator'][loss_key]
                    dict_loss_d[loss_key] = dict_loss_d[loss_key].item() if isinstance(dict_loss_d[loss_key], torch.Tensor) else dict_loss_d[loss_key]
                # backward
                optimizer_d.zero_grad()
                if mixed_precision:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                #########
                # Update generator
                # net_g.zero_grad()
                if hasattr(net_g, 'sigmoid') and getattr(net_g, 'sigmoid'):
                    sr = net_g(lr)
                else:
                    sr = net_g(lr).clip(0,1)
                for p in net_d.parameters():
                    p.requires_grad = False
                # forward
                if hasattr(net_d, 'relativistic') and getattr(net_d, 'relativistic'):
                    sr_out = net_d(sr, lr)
                    hr_out = net_d(hr, lr)
                else:
                    sr_out = net_d(sr)
                    hr_out = None
                # get loss
                dict_loss_g = {}
                loss_g = 0
                for loss_key in loss_module_dict['generator'].keys():
                    if loss_key != 'adv':
                        o = sr
                        t = hr
                    else:
                        o = sr_out
                        t = hr_out
                    dict_loss_g[loss_key] = loss_module_dict['generator'][loss_key](o,t)
                    loss_g += dict_loss_g[loss_key] * loss_weight_dict['generator'][loss_key]
                    dict_loss_g[loss_key] = dict_loss_g[loss_key].item() if isinstance(dict_loss_g[loss_key], torch.Tensor) else dict_loss_g[loss_key]
                # backward
                optimizer_g.zero_grad()
                if mixed_precision:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
                    optimizer_g.step()
                ###########
                # record loss
                for key in dict_loss_d.keys():
                    list_loss.setdefault(key, [])
                    list_loss[key].append(dict_loss_d[key])
                for key in dict_loss_g.keys():
                    list_loss.setdefault(key, [])
                    list_loss[key].append(dict_loss_g[key])
            if progress_report == 'pbar':
                pbar.update(1)
                pbar.set_postfix({key: f'{np.mean(val):.2f}' for key, val in list_loss.items()})
        if progress_report == 'pbar':
            pbar.close()
        sr = sr.float()
        # save train patch - image & freq
        slice_idx = batch['input_0'].shape[-1]//2
        fig, axes = plt.subplots(2,3,figsize = (10,6))
        # image
        axes[0,0].imshow(batch['input_0'][0,0,:,:,slice_idx].cpu().detach(), cmap = 'gray', vmin = 0, vmax = 1)
        axes[0,1].imshow(sr[0,0,:,:,slice_idx].cpu().detach(), cmap = 'gray', vmin = 0, vmax = 1)
        axes[0,2].imshow(batch['target_0'][0,0,:,:,slice_idx].cpu().detach(), cmap = 'gray', vmin = 0, vmax = 1)
        # freq
        mag_i = torch.log(abs(torch.fft.fftshift(torch.fft.fftn(batch['input_0'][0,0,:,:,slice_idx].cpu().detach()))))
        mag_t = torch.log(abs(torch.fft.fftshift(torch.fft.fftn(batch['target_0'][0,0,:,:,slice_idx].cpu().detach()))))
        mag_p = torch.log(abs(torch.fft.fftshift(torch.fft.fftn(sr[0,0,:,:,slice_idx].cpu().detach()))))
        axes[1,0].imshow(mag_i, cmap = 'jet')
        axes[1,1].imshow(mag_p, cmap = 'jet')
        axes[1,2].imshow(mag_t, cmap = 'jet')
        fig.savefig(os.path.join(save_dir_progress_train, f'epoch_{str(curr_epoch).zfill(4)}.png'))
        # plt.show()
        plt.close()
        # save train patch - freq
        # record progress
        if progress_report == 'wandb':
            wandb_record = {key: float(np.mean(val)) for key, val in list_loss.items()}
            wandb_record.update({
                'train_patch': wandb.Image(Image.open(os.path.join(save_dir_progress_train, f'epoch_{str(curr_epoch).zfill(4)}.png')), caption = 'train_patch')}
            )
            print(f'train at epoch {epoch}/{max_epochs}')
            print('\t', {key: f'{np.mean(val):.2f}' for key, val in list_loss.items()})
        list_loss['epoch'] = curr_epoch
        progress['loss'].append(list_loss)

        if curr_epoch % save_iter == 0 or epoch == max_epochs-1:
            with torch.no_grad():
                monitor_eval_progress = {'filename': [], 'psnr': [], 'ssim': [], 'mae': [], 'ldf': []}
                if progress_report == 'pbar':
                    pbar = tqdm.tqdm(total = len(dl_monitor), position = 0, desc = f'eval {curr_epoch}/{max_epochs}')
                for idx, batch in enumerate(dl_monitor):
                    lr = {f"level_{key.split('_')[1]}": batch[key].to(device) for key in batch.keys() if re.fullmatch('input_\d+', key)}
                    hr = {f"level_{key.split('_')[1]}": batch[key].to(device) for key in batch.keys() if re.fullmatch('target_\d+', key)}
                    # make input to the same resolution as output
                    acc_scale_inverse = [np.prod([inverse_scale_factors[i][spatial_idx] for i in range(start_level)]) for spatial_idx in range(net_g.spatial_dims)]
                    x = F.interpolate(lr[f'level_{start_level}'], scale_factor = acc_scale_inverse, mode = 'nearest')
                    sr = sliding_window_inference(x, roi_size = patch_size, sw_batch_size = batch_size * num_patch, predictor = net_g.predict, overlap = 0.625, sigma_scale = 0.125, mode = 'gaussian').clip(0,1)
                    batch['pred'] = sr
                    batch[image_key] = batch['target_0']
                    batch['pred'].meta = batch[image_key].meta
                    list_data = decollate_batch(batch)
                    for data in list_data:
                        filename = data[image_key].meta['filename_or_obj']
                        # inverted = trans_post_eval(data)
                        inverted = data

                        pred = inverted['pred'].detach().to(device)
                        target = inverted[image_key].detach().to(device)

                        # calculate the scores
                        func_psnr = PeakSignalNoiseRatio(data_range = target.max() - target.min()).to(device)
                        func_ssim = StructuralSimilarityIndexMeasure(data_range = target.max() - target.min(), channel_dim = 0).to(device)
                        ssim = func_ssim(target, pred).item()
                        psnr = func_psnr(target, pred).item()
                        mae = calc_mae(target, pred).item()
                        ldf = calc_ldf_torch(target, pred, channel_axis=0).item()
                        # record
                        monitor_eval_progress['filename'].append(filename)
                        monitor_eval_progress['psnr'].append(psnr)
                        monitor_eval_progress['ssim'].append(ssim)
                        monitor_eval_progress['mae'].append(mae)
                        monitor_eval_progress['ldf'].append(ldf)
                        # Save figure & nifti for progress report
                        p = data['pred'][0].cpu().detach()
                        t = data[image_key][0].cpu().detach()
                        # save nifti
                        nifti_orig = nib.load(filename)
                        nifti = nib.Nifti1Image(p, affine = inverted['pred'].affine, header = nifti_orig.header)
                        save_fpath = os.path.join(save_dir_image, f'epoch_{str(curr_epoch).zfill(4)}', filename.removeprefix(data_dir).strip('/'))
                        os.makedirs(os.path.dirname(save_fpath), exist_ok = True)
                        nib.save(nifti, save_fpath)
                        # save figure
                        image_id = filename.replace(data_dir, '').strip('/').strip('.nii.gz').replace('/', '@')
                        for i in range(3):
                            viewname = ['coronal', 'axial', 'sagittal'][i]
                            t = t.permute(1,2,0)
                            p = p.permute(1,2,0)
                            slice_idx = t.shape[0]//2
                            fig, axes = plt.subplots(1,2,figsize = (20,20))
                            axes[0].imshow(p[slice_idx], cmap = 'gray')
                            axes[1].imshow(t[slice_idx], cmap = 'gray')
                            fig.savefig(os.path.join(save_dir_progress_monitor, f'{str(curr_epoch).zfill(4)}-image-{viewname}-{image_id}.png'))
                            # plt.show()
                            plt.close()

                            # plot frequency
                            mag_t = torch.log(abs(torch.fft.fftshift(torch.fft.fftn(t[slice_idx]))))
                            mag_p = torch.log(abs(torch.fft.fftshift(torch.fft.fftn(p[slice_idx]))))
                            fig, axes = plt.subplots(1,2,figsize = (20,20))
                            axes[0].imshow(mag_p, cmap = 'jet')
                            axes[1].imshow(mag_t, cmap = 'jet')
                            fig.savefig(os.path.join(save_dir_progress_monitor, f'{str(curr_epoch).zfill(4)}-freq-{viewname}-{image_id}.png'))
                            # plt.show()
                            plt.close()
                            # wandb
                            if progress_report == 'wandb':
                                wandb_record.update({
                                    f'monitor_image_{idx}_{viewname}': wandb.Image(Image.open(os.path.join(save_dir_progress_monitor, f'{str(curr_epoch).zfill(4)}-image-{viewname}-{image_id}.png')), caption = 'monitor_image'), 
                                    f'monitor_freq_{idx}_{viewname}': wandb.Image(Image.open(os.path.join(save_dir_progress_monitor, f'{str(curr_epoch).zfill(4)}-freq-{viewname}-{image_id}.png')), caption = 'monitor_freq')
                                })
                    if progress_report == 'pbar':
                        pbar.update(1)
                        pbar.set_postfix({key: f"{np.mean(monitor_eval_progress[key]):.3f}" for key in ['psnr', 'ssim', 'mae', 'ldf']})
                if progress_report == 'pbar':
                    pbar.close()
                # record
                if progress_report == 'wandb':
                    wandb_record.update({f'monitor_{key}': float(np.mean(monitor_eval_progress[key])) for key in ['psnr', 'ssim', 'mae', 'ldf']})
                    print(f'eval at epoch {epoch}/{max_epochs}')
                    print('\t', {f'monitor_{key}': float(np.mean(monitor_eval_progress[key])) for key in ['psnr', 'ssim', 'mae', 'ldf']})
                monitor_eval_progress['epoch'] = curr_epoch
                progress['monitor_eval'].append(monitor_eval_progress)
                del sr, data, inverted, pred, target, lr, hr, batch, list_data
                torch.cuda.empty_cache()
        state_dict = {
            'net_g': net_g.state_dict(),
            'net_d': net_d.state_dict(),
            'optim_g': optimizer_g.state_dict(),
            'optim_d': optimizer_d.state_dict(),
            'scheduler_g': scheduler_g.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'curr_epoch': curr_epoch + 1,
            'progress': progress
        }
        if mixed_precision:
            state_dict['scaler_g'] = scaler_g.state_dict()
            state_dict['scaler_d'] = scaler_d.state_dict()

        torch.save(state_dict, os.path.join(save_dir_cp, f'latest.pt'))
        if curr_epoch % save_iter == 0:
            torch.save(state_dict, os.path.join(save_dir_cp, f'epoch_{str(curr_epoch).zfill(4)}.pt'))
        # update wandb
        if progress_report == 'wandb':
            wandb_record.update({'epoch': curr_epoch + 1})
            wandb.log(wandb_record)
        curr_epoch = epoch + 1
if __name__ == '__main__':
    main()