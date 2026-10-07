# screen -S compress
# tar -I 'zstd --ultra -22 -T0' -cf - ./General_dic | pv -s 1652749535721 | split -b 50G - data.tar.zst.
# screen -r compress

import os
import gc
import time
import argparse
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
#import pyiqa  # patched: skipping network-dependent metrics

from datasets1.kitti_dataset import PairKitti, collate_fn
from utils1.utils import instantiate_from_config

# ---------------------------------------------------------------------------- #
# Distributed Init
# ---------------------------------------------------------------------------- #
def init_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(gpu)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )

        dist.barrier()
        return rank, world_size, gpu
    else:
        return 0, 1, 0


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="./configs/train_rdeic.yaml")
    parser.add_argument("--model_config", type=str, default="./configs/model/rdeic.yaml")
    parser.add_argument("--data_root", type=str, default="dataTVT/KITTI", help="Path to KITTI dataset root")
    parser.add_argument("--save_dir", type=str, default="logs_pro_MSI/final_2_0.1", help="Fixed directory to save logs/checkpoints")
    parser.add_argument("--vis_f", type=int, default=950) 
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter") 
    parser.add_argument("--train_distortion_decoder_only", type=bool, default=False)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--val_step_interval", type=int, default=1000, help="Validation every N steps")
    parser.add_argument("--val_epoch_interval", type=int, default=1, help="Validation every N epochs")
    return parser

def load_model_from_config(config):
    print(f"Loading model from {config.target}")
    model = instantiate_from_config(config)
    return model

def tensor2img(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

# def run_validation(model, val_loader, metric_funcs, device, epoch, global_step, 
#                    img_dir, val_log_file, ckpt_dir, best_psnr, optimizer, scheduler, train_quantizer_only=False):
#     raw_model = model.module if hasattr(model, 'module') else model
#     raw_model.eval()
#     val_metrics = {"psnr": [], "ms_ssim": [], "lpips": [], "dists": [], "bpp": []}
    
#     print(f"\n>>> Running Validation at Epoch {epoch+1} / Step {global_step} ...")
    
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(val_loader, desc="Validation")):

#             log, batch_bpp = raw_model.log_images(batch, sample_steps=raw_model.fixed_step, bs=1, fidelity_ratio=1.0)
            
#             pred = log["samples"].clamp(0, 1) 
#             target = log["target"].clamp(0, 1)

#             if metric_funcs["psnr"] is not None: val_metrics["psnr"].append(metric_funcs["psnr"](pred, target).item())
#             if metric_funcs["ms_ssim"] is not None: val_metrics["ms_ssim"].append(metric_funcs["ms_ssim"](pred, target).item())
#             if metric_funcs["lpips"] is not None: val_metrics["lpips"].append(metric_funcs["lpips"](pred, target).item())
#             if metric_funcs["dists"] is not None: val_metrics["dists"].append(metric_funcs["dists"](pred, target).item())
            
#             val_metrics["bpp"].append(batch_bpp.item() if isinstance(batch_bpp, torch.Tensor) else batch_bpp)

#             if i == 0:
#                 pred_img = tensor2img(log["samples"][0])
#                 gt_img = tensor2img(log["target"][0])
#                 Image.fromarray(pred_img).save(os.path.join(img_dir, f"step_{global_step}_pred.png"))
#                 Image.fromarray(gt_img).save(os.path.join(img_dir, f"step_{global_step}_gt.png"))
    
#     avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    
#     print(f"\n[Validation Report - Step {global_step}]")
#     print(f"BPP: {avg_metrics['bpp']:.4f} | PSNR: {avg_metrics['psnr']:.2f} dB")

#     if dist.get_rank() <= 0:
#         log_line = f"{epoch+1},{global_step},{avg_metrics['bpp']:.6f},{avg_metrics['psnr']:.6f},{avg_metrics['ms_ssim']:.6f},{avg_metrics.get('lpips', float('nan')):.6f},{avg_metrics.get('dists', float('nan')):.6f},{avg_metrics.get('lpips', float('nan')):.6f},{avg_metrics.get('dists', float(.nan.)):.6f},{avg_metrics.get(.lpips., float(.nan.)):.6f},{avg_metrics.get(.dists., float(.nan.)):.6f}\n"
#         with open(val_log_file, 'a') as f:
#             f.write(log_line)
            
#         if train_quantizer_only:
#             state_dict_to_save = raw_model.preprocess_model.stanh_y.state_dict() 
#         else:
#             state_dict_to_save = raw_model.state_dict()

#         if global_step % 2000 == 0:
#             checkpoint_dict = {
#                 "epoch": epoch,
#                 "global_step": global_step,
#                 "state_dict": state_dict_to_save,
#                 "best_psnr": best_psnr,
#                 "optimizer": optimizer.state_dict(),
#                 "scheduler": scheduler.state_dict(),
#             }   
#             if avg_metrics['psnr'] > best_psnr:
#                 best_psnr = avg_metrics['psnr']
#                 torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "best_psnr.pth"))
#             torch.save(checkpoint_dict, os.path.join(ckpt_dir, "last.pth"))
#     return best_psnr


def run_validation_light(model, val_loader, metric_funcs, device, global_step):
    """
    轻量验证函数：只计算指标并返回字典，不保存任何文件。
    该函数仅在 rank 0 主进程中调用。
    """
    model.eval()
    val_metrics = {"psnr": [], "ms_ssim": [], "lpips": [], "dists": [], "bpp": []}
    
    print(f"\n>>> Running Validation at Step {global_step} ...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
            log, batch_bpp = model.log_images(
                batch, 
                sample_steps=model.fixed_step, 
                bs=1, 
                fidelity_ratio=1.0
            )
            pred = log["samples"].clamp(0, 1)
            target = log["target"].clamp(0, 1)
            
            if metric_funcs["psnr"] is not None: val_metrics["psnr"].append(metric_funcs["psnr"](pred, target).item())
            if metric_funcs["ms_ssim"] is not None: val_metrics["ms_ssim"].append(metric_funcs["ms_ssim"](pred, target).item())
            if metric_funcs["lpips"] is not None: val_metrics["lpips"].append(metric_funcs["lpips"](pred, target).item())
            if metric_funcs["dists"] is not None: val_metrics["dists"].append(metric_funcs["dists"](pred, target).item())
            val_metrics["bpp"].append(batch_bpp.item() if isinstance(batch_bpp, torch.Tensor) else batch_bpp)
            
    
    avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    print(f"\n[Validation Report - Step {global_step}]")
    print(f"BPP: {avg_metrics['bpp']:.4f} | PSNR: {avg_metrics['psnr']:.2f} dB")
    return avg_metrics



from torch.utils.data import Subset
def main():
    parser = get_parser()
    args = parser.parse_args()
    rank, world_size, gpu = init_dist()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{gpu}")

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #
    save_dir = args.save_dir

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    img_dir = os.path.join(save_dir, "images")
    vis_dir = os.path.join(save_dir, "train_vis")

    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    # ---------------------------------------------------------------------------- #
    # Logs
    # ---------------------------------------------------------------------------- #
    val_log_file = os.path.join(save_dir, "val_metrics.csv")

    if is_main and not os.path.exists(val_log_file):
        with open(val_log_file, "w") as f:
            f.write("Epoch,Step,BPP,PSNR,MS-SSIM,LPIPS,DISTS\n")

    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    model_conf = OmegaConf.load(args.model_config)

    model = load_model_from_config(model_conf)

    model.to(device)

    # ---------------------------------------------------------------------------- #
    # Metrics
    # ---------------------------------------------------------------------------- #
    # Metrics: PSNR/MS-SSIM via torch (no download needed)
    # LPIPS/DISTS via pyiqa (optional, requires network weights)
    import torch.nn.functional as F
    from pytorch_msssim import ms_ssim as msssim_fn

    def compute_psnr(pred, target, data_range=1.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(100.0)
        return 10 * torch.log10(data_range ** 2 / mse)

    def compute_ms_ssim(pred, target, data_range=1.0):
        # pytorch_msssim expects (B, C, H, W), returns scalar
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        return msssim_fn(pred, target, data_range=data_range, size_average=True)

    # LPIPS using standalone lpips package (torchvision weights, no HF needed)
    import lpips as lpips_pkg
    lpips_fn = lpips_pkg.LPIPS(net='vgg', verbose=False).to(device)

    def compute_lpips(pred, target):
        # lpips expects [-1, 1], our images are [0, 1]
        p = pred * 2 - 1
        t = target * 2 - 1
        if p.dim() == 3:
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)
        return lpips_fn(p, t).squeeze()

    # DISTS via pyiqa (may need network for weights)
    try:
        import pyiqa
        dists_fn = pyiqa.create_metric("dists", device=device)
    except Exception:
        dists_fn = None

    metric_funcs = {
        "psnr": compute_psnr,
        "ms_ssim": compute_ms_ssim,
        "lpips": compute_lpips,
        "dists": dists_fn,
    }

    # ---------------------------------------------------------------------------- #
    # Dataset
    # ---------------------------------------------------------------------------- #
    train_dataset = PairKitti(
        path=args.data_root,
        set_type="train",
        resize=(256, 512)
    )

    val_dataset = PairKitti(
        path=args.data_root,
        set_type="val",
        resize=(256, 512)
    )

    val_dataset = Subset(
        val_dataset,
        list(range(len(val_dataset) // 10))
    )

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        # persistent_workers=True,
        collate_fn=collate_fn
    )

    # ---------------------------------------------------------------------------- #
    # DDP
    # ---------------------------------------------------------------------------- #
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[gpu],
            output_device=gpu,
            find_unused_parameters=True,
            broadcast_buffers=False
        )

    raw_model = model.module if hasattr(model, "module") else model



    # if args.train_distortion_decoder_only:
    #     # params = list(model.preprocess_model.stanh_y.parameters())
    #     print("\n" + "!"*80)
    #     print("🚀 MODE: TRAINING QUANTIZER ONLY")
    #     print("!"*80 + "\n")

    #     for param in model.parameters():
    #         param.requires_grad = False
        
    #     for param in model.preprocess_model.stanh_y.parameters():
    #         param.requires_grad = True
            
    #     params = list(model.preprocess_model.stanh_y.parameters())
    # else:
    #     # params = list(model.control_model.parameters()) + list(model.preprocess_model.parameters()) + list(model.fusion_feature.parameters()) + list(model.fusion_latent.parameters()) + list(model.fusion_gate.parameters())+ list(model.lgam.parameters())+ list(model.fusion_feature_multi.parameters())+ list(model.fusion_latent_multi.parameters())
    #     params = list(raw_model.control_model.parameters()) + \
    #              list(raw_model.preprocess_model.parameters()) + \
    #              list(raw_model.fusion_feature.parameters()) + \
    #              list(raw_model.fusion_latent.parameters()) + \
    #              list(raw_model.fusion_gate.parameters()) + \
    #              list(raw_model.lgam.parameters()) + \
    #              list(raw_model.fusion_feature_multi.parameters()) + \
    #              list(raw_model.fusion_latent_multi.parameters())

# Optimizer & Scheduler
    if args.train_distortion_decoder_only:
        print("\n" + "!"*80)
        print("🚀 MODE: TRAINING QUANTIZER ONLY (VBR Refinement)")
        print("!"*80 + "\n")

        # 🟢 修复点：使用 raw_model 访问 preprocess_model
        # 1. 先冻结模型所有参数
        for param in raw_model.parameters():
            param.requires_grad = False
        
        # 2. 只解冻当前要训练的 STanH 量化器参数
        # 确保你的 VAE 分支参数在 derivation 阶段是固定的
        for param in raw_model.preprocess_model.stanh_y.parameters():
            param.requires_grad = True
            
        params = list(raw_model.preprocess_model.stanh_y.parameters())
        print("\n🔍 Trainable parameters:")
        total = 0

        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                print(f"✅ {name} | shape={tuple(param.shape)}")
                total += param.numel()
        print(f"\n📊 Total trainable params: {total}")

    else:
        params = list(raw_model.control_model.parameters()) + \
                 list(raw_model.preprocess_model.parameters()) + \
                 list(raw_model.fusion_feature.parameters()) + \
                 list(raw_model.fusion_latent.parameters()) + \
                 list(raw_model.fusion_gate.parameters()) + \
                 list(raw_model.lgam.parameters())
                #  list(raw_model.fusion_feature_multi.parameters()) + \
                #  list(raw_model.fusion_latent_multi.parameters())

        if not raw_model.sd_locked:
            print("training unet.......................")
            params += list(raw_model.model.diffusion_model.input_blocks.parameters())
            params += list(raw_model.model.diffusion_model.middle_block.parameters())
            params += list(raw_model.model.diffusion_model.output_blocks.parameters())
            params += list(raw_model.model.diffusion_model.out.parameters())
        total = 0

        # for name, param in raw_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"✅ {name} | shape={tuple(param.shape)}")
        #         total += param.numel()
        # print(f"\n📊 Total trainable params: {total}")
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # ---------------------------------------------------------------------------- #
    # Resume
    # ---------------------------------------------------------------------------- #
    start_epoch = 0
    global_step = 0
    best_psnr = 0.0

    if args.resume and os.path.isfile(args.resume):

        if is_main:
            print(f"Loading checkpoint (model weights only): {args.resume}")

        # Load only model weights to CPU to save GPU memory
        checkpoint = torch.load(args.resume, map_location='cpu')

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        raw_model.load_state_dict(state_dict, strict=False)
        # Move model back to device after loading on CPU
        raw_model.to(device)

        # Track progress from checkpoint but start optimizer fresh
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_psnr = checkpoint.get("best_psnr", 0.0)

        if is_main:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
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
                current_vis_path = os.path.join(
                    vis_dir,
                    f"step_{global_step:06d}.png"
                )

            x, c = raw_model.get_input(
                batch,
                k="main",
                vis_path=current_vis_path
            )

            loss, loss_dict = model(x, c)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                1.0
            )

            optimizer.step()

            if is_main:
                pbar.set_description(
                    f"Epoch {epoch+1} | "
                    f"Loss {loss.item():.4f}"
                )

            # ---------------------------------------------------------------------------- #
            # Validation
            # ---------------------------------------------------------------------------- #
            # if global_step % args.val_step_interval == 0:
            #     if dist.is_initialized():
            #         dist.barrier()

            #     model.eval()

            #     if is_main:
            #         best_psnr = run_validation(
            #             raw_model,
            #             val_loader,
            #             metric_funcs,
            #             device,
            #             epoch,
            #             global_step,
            #             img_dir,
            #             val_log_file,
            #             ckpt_dir,
            #             best_psnr,
            #             optimizer,
            #             scheduler,
            #             train_quantizer_only=args.train_distortion_decoder_only
            #         )

            #     # 🌟 修复 2：主进程验证完了，把 best_psnr 广播给所有副进程
            #     if dist.is_initialized():
            #         # 将 best_psnr 打包成 Tensor 进行广播
            #         psnr_tensor = torch.tensor([best_psnr], dtype=torch.float32, device=device)
            #         dist.broadcast(psnr_tensor, src=0)
            #         best_psnr = psnr_tensor.item()
                    
            #         # 再加一个屏障，确保所有进程都拿到最新 PSNR 后，一起进入下一个训练 Step
            #         dist.barrier()
                
            #     # 统一且安全地切回训练模式
            #     model.train()
            if global_step % args.val_step_interval == 0:
                # 1. 所有进程同步
                if dist.is_initialized():
                    dist.barrier()
                
                model.eval()
                
                avg_metrics = None
                if is_main:
                    # 2. 仅主进程执行轻量验证（无文件 I/O）
                    avg_metrics = run_validation_light(
                        raw_model, val_loader, metric_funcs, device, global_step
                    )
                    # 3. 更新最佳 PSNR（仅内存）
                    if avg_metrics['psnr'] > best_psnr:
                        best_psnr = avg_metrics['psnr']
                
                # 4. 广播 best_psnr 给所有进程
                if dist.is_initialized():
                    best_psnr_tensor = torch.tensor([best_psnr], dtype=torch.float32, device=device)
                    dist.broadcast(best_psnr_tensor, src=0)
                    best_psnr = best_psnr_tensor.item()
                    dist.barrier()   # 确保所有进程收到广播
                
                # 5. 所有进程同步完毕后，主进程执行文件保存（日志 + checkpoint）
                if is_main:
                    # 保存验证日志到 CSV
                    log_line = f"{epoch+1},{global_step},{avg_metrics['bpp']:.6f},{avg_metrics['psnr']:.6f},{avg_metrics['ms_ssim']:.6f},{avg_metrics.get('lpips', float('nan')):.6f},{avg_metrics.get('dists', float('nan')):.6f}\n"
                    with open(val_log_file, 'a') as f:
                        f.write(log_line)

                    state_dict_to_save = raw_model.state_dict()
                    
                    if global_step % 4000 == 0:
                        checkpoint_dict = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": state_dict_to_save,
                            "best_psnr": best_psnr,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }
                        torch.save(checkpoint_dict, os.path.join(ckpt_dir, "last.pth"))

                    if avg_metrics['psnr'] > best_psnr:  # 此处逻辑可根据实际需求调整
                        torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "best_psnr.pth"))
                
                model.train()
        scheduler.step()

    cleanup()



if __name__ == "__main__":
    main()
