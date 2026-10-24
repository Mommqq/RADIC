import os
import gc
import time
import argparse
import random
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets1.kitti_dataset import PairKitti, collate_fn
from utils1.utils import instantiate_from_config
from checkpoints import save_training_checkpoint
from contextlib import contextmanager

def init_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=30))
        dist.barrier()
        return (rank, world_size, gpu)
    else:
        return (0, 1, 0)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/model/radic.yaml')
    parser.add_argument('--data_root', type=str, default='dataTVT/KITTI')
    parser.add_argument('--save_dir', type=str, default='outputs/radic')
    parser.add_argument('--vis_f', type=int, default=950)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=231)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--val_step_interval', type=int, default=1000)
    return parser

def load_model_from_config(config):
    print(f'Loading model from {config.target}')
    model = instantiate_from_config(config)
    return model

def tensor2img(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

@contextmanager
def validation_gallery(model):
    previous_split = model.gallery_split
    previous_mode = model.training
    try:
        model.set_gallery('val')
        model.eval()
        yield
    finally:
        model.set_gallery(previous_split)
        model.train(previous_mode)

def run_validation_light(model, val_loader, metric_funcs, device, global_step):
    model.eval()
    val_metrics = {'psnr': [], 'ms_ssim': [], 'lpips': [], 'dists': [], 'bpp': []}
    print(f'\n>>> Running Validation at Step {global_step} ...')
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(val_loader, desc='Validation')):
            (log, batch_bpp) = model.log_images(batch, sample_steps=model.fixed_step, bs=1, fidelity_ratio=1.0)
            pred = log['samples'].clamp(0, 1)
            target = log['target'].clamp(0, 1)
            if metric_funcs['psnr'] is not None:
                val_metrics['psnr'].append(metric_funcs['psnr'](pred, target).item())
            if metric_funcs['ms_ssim'] is not None:
                val_metrics['ms_ssim'].append(metric_funcs['ms_ssim'](pred, target).item())
            if metric_funcs['lpips'] is not None:
                val_metrics['lpips'].append(metric_funcs['lpips'](pred, target).item())
            if metric_funcs['dists'] is not None:
                val_metrics['dists'].append(metric_funcs['dists'](pred, target).item())
            val_metrics['bpp'].append(batch_bpp.item() if isinstance(batch_bpp, torch.Tensor) else batch_bpp)
    avg_metrics = {k: np.mean(v) for (k, v) in val_metrics.items()}
    print(f'\n[Validation Report - Step {global_step}]')
    print(f"BPP: {avg_metrics['bpp']:.4f} | PSNR: {avg_metrics['psnr']:.2f} dB")
    return avg_metrics
from torch.utils.data import Subset

def main():
    parser = get_parser()
    args = parser.parse_args()
    (rank, world_size, gpu) = init_dist()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    is_main = rank == 0
    device = torch.device(f'cuda:{gpu}')
    save_dir = args.save_dir
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    img_dir = os.path.join(save_dir, 'images')
    vis_dir = os.path.join(save_dir, 'train_vis')
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
    val_log_file = os.path.join(save_dir, 'val_metrics.csv')
    if is_main and (not os.path.exists(val_log_file)):
        with open(val_log_file, 'w') as f:
            f.write('Epoch,Step,BPP,PSNR,MS-SSIM,LPIPS,DISTS\n')
    model_conf = OmegaConf.load(args.model_config)
    model = load_model_from_config(model_conf)
    model.to(device)
    model.set_gallery('train')
    import torch.nn.functional as F
    from pytorch_msssim import ms_ssim as msssim_fn

    def compute_psnr(pred, target, data_range=1.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(100.0)
        return 10 * torch.log10(data_range ** 2 / mse)

    def compute_ms_ssim(pred, target, data_range=1.0):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        return msssim_fn(pred, target, data_range=data_range, size_average=True)
    import lpips as lpips_pkg
    lpips_fn = lpips_pkg.LPIPS(net='vgg', verbose=False).to(device)

    def compute_lpips(pred, target):
        p = pred * 2 - 1
        t = target * 2 - 1
        if p.dim() == 3:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)
        return lpips_fn(p, t).squeeze()
    try:
        import pyiqa
        dists_fn = pyiqa.create_metric('dists', device=device)
    except Exception:
        dists_fn = None
    metric_funcs = {'psnr': compute_psnr, 'ms_ssim': compute_ms_ssim, 'lpips': compute_lpips, 'dists': dists_fn}
    train_dataset = PairKitti(path=args.data_root, set_type='train', stereo=False, resize=(256, 512))
    val_dataset = PairKitti(path=args.data_root, set_type='val', stereo=False, resize=(256, 512))
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    if world_size > 1:
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True, broadcast_buffers=False)
    raw_model = model.module if hasattr(model, 'module') else model
    params = list(raw_model.control_model.parameters()) + list(raw_model.preprocess_model.parameters()) + list(raw_model.fusion_feature.parameters()) + list(raw_model.fusion_latent.parameters()) + list(raw_model.fusion_gate.parameters()) + list(raw_model.lgam.parameters()) + list(raw_model.fusion_feature_multi.parameters()) + list(raw_model.fusion_latent_multi.parameters())
    if not raw_model.sd_locked:
        print('training unet.......................')
        params += list(raw_model.model.diffusion_model.input_blocks.parameters())
        params += list(raw_model.model.diffusion_model.middle_block.parameters())
        params += list(raw_model.model.diffusion_model.output_blocks.parameters())
        params += list(raw_model.model.diffusion_model.out.parameters())
    total = 0
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-06)
    start_epoch = 0
    global_step = 0
    best_psnr = float('-inf')
    if args.resume and os.path.isfile(args.resume):
        if is_main:
            print(f'Loading checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        raw_model.load_state_dict(state_dict, strict=False)
        raw_model.to(device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_psnr = checkpoint.get('best_psnr', float('-inf'))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if is_main:
            print(f'Resumed from epoch {start_epoch}, step {global_step}')
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader) if is_main else train_loader
        for batch in pbar:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            current_vis_path = None
            if is_main and global_step % args.vis_f == 0:
                current_vis_path = os.path.join(vis_dir, f'step_{global_step:06d}.png')
            (x, c) = raw_model.get_input(batch, k='main', vis_path=current_vis_path)
            (loss, loss_dict) = model(x, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            if is_main:
                pbar.set_description(f'Epoch {epoch + 1} | Loss {loss.item():.4f}')
            if global_step % args.val_step_interval == 0:
                if dist.is_initialized():
                    dist.barrier()
                avg_metrics = None
                if is_main:
                    with validation_gallery(raw_model):
                        avg_metrics = run_validation_light(raw_model, val_loader, metric_funcs, device, global_step)
                    best_psnr = save_training_checkpoint(ckpt_dir, raw_model, optimizer, scheduler, epoch, global_step, best_psnr, avg_metrics)
                if dist.is_initialized():
                    best_psnr_tensor = torch.tensor([best_psnr], dtype=torch.float32, device=device)
                    dist.broadcast(best_psnr_tensor, src=0)
                    best_psnr = best_psnr_tensor.item()
                    dist.barrier()
                if is_main:
                    log_line = f"{epoch + 1},{global_step},{avg_metrics['bpp']:.6f},{avg_metrics['psnr']:.6f},{avg_metrics['ms_ssim']:.6f},{avg_metrics.get('lpips', float('nan')):.6f},{avg_metrics.get('dists', float('nan')):.6f}\n"
                    with open(val_log_file, 'a') as f:
                        f.write(log_line)
                model.train()
        scheduler.step()
        if is_main:
            save_training_checkpoint(ckpt_dir, raw_model, optimizer, scheduler, epoch, global_step, best_psnr)
    cleanup()
if __name__ == '__main__':
    main()
