
# import argparse
# import os
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import pyiqa
# from torch.utils.data import DataLoader, Subset
# from omegaconf import OmegaConf
# if not hasattr(torch, 'compiler'):
#     class DummyCompiler:
#         @staticmethod
#         def disable(func):
#             return func
#     torch.compiler = DummyCompiler()
#     print("⚠️ [Patch Applied] torch.compiler injected manually to bypass e3nn check.")
# from datasets1.kitti_dataset import PairKitti, collate_fn as kitti_collate
# from datasets1.cityscapes_dataset import PairCityscape, collate_fn as city_collate
# from datasets1.instereo2k_dataset import PairInstereo2K, collate_fn as instereo_collate
# from utils1.utils import instantiate_from_config


# def msssim_to_db(msssim_val):
#     if msssim_val >= 1.0 - 1e-9:
#         return 100.0 
#     return -10.0 * np.log10(1.0 - msssim_val)

# def get_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="./configs/model/rdeic.yaml", help="Path to model config")
#     parser.add_argument("--ckpt", type=str, default="logs_pro_MSI/experiment_2_1/checkpoints/best_psnr.pth", help="Path to checkpoint file")
#     parser.add_argument("--kitti_root", type=str, default="dataTVT/KITTI", help="Path to KITTI root")
#     parser.add_argument("--city_root", type=str, default="cityscape_dataset", help="Path to Cityscapes root")
#     parser.add_argument("--instereo_root", type=str, default="dataTVT/InStereo2K", help="Path to InStereo2K root")    
#     parser.add_argument("--output_base_dir", type=str, default="results_pro_MSI", help="Base directory to save results")
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--resize", type=int, nargs=2, default=[256, 512], help="H W") 
#     parser.add_argument("--sample_steps", type=int, default=2) 

#     return parser

# def load_model(config_path, ckpt_path, device):
#     print(f"Loading model from {config_path}")
#     config = OmegaConf.load(config_path)
#     model = instantiate_from_config(config)
    
#     print(f"Loading weights from {ckpt_path}")
#     sd = torch.load(ckpt_path, map_location="cpu")
#     if "state_dict" in sd:
#         sd = sd["state_dict"]
    
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0: print(f"Missing keys: {len(m)}")
#     if len(u) > 0: print(f"Unexpected keys: {len(u)}")
    
#     model.to(device)
#     model.eval()
#     return model, config 

# def tensor2img(tensor, auto_norm=False):
#     img = tensor.permute(1, 2, 0).cpu().numpy()
#     if auto_norm:
#         _min, _max = img.min(), img.max()
#         if _max - _min > 1e-5:
#             img = (img - _min) / (_max - _min)
#         else:
#             img = np.zeros_like(img)
#     img = (img * 255).clip(0, 255).astype(np.uint8)
#     return img

# def evaluate_dataset(model, loader, device, metric_funcs, save_dir, save_dir_gt, save_dir_side_gt, metric_dir,sample_steps):
#     os.makedirs(save_dir, exist_ok=True)
#     # os.makedirs(save_dir_gt, exist_ok=True)
#     # os.makedirs(save_dir_side_gt, exist_ok=True)   
#     os.makedirs(metric_dir, exist_ok=True)   
#     # --- 1. 初始化 CSV 文件 (记录逐图指标) ---
#     csv_file = os.path.join(metric_dir, "detailed_metrics.csv")
#     with open(csv_file, "w") as f:
#         # 增加 MS-SSIM(dB) 列
#         f.write("Index,BPP,PSNR,MS_SSIM,MS_SSIM_dB,LPIPS,DISTS\n")
    
#     # 指标列表
#     metrics = {
#         "psnr": [], 
#         "ms_ssim": [], 
#         "ms_ssim_db": [], # 新增
#         "lpips": [], 
#         "dists": [], 
#         "bpp": []
#     }
    
#     print(f"Start evaluation on {len(loader)} samples. Results -> {save_dir}")
    
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(loader)):
#             # 1. 确保所有 tensor 都在正确的设备上
#             for k in batch.keys():
#                 if isinstance(batch[k], torch.Tensor):
#                     batch[k] = batch[k].to(device)
            
#             # 2. 构造完整的 Input Batch
#             # 必须包含 main_rel_path，否则模型 get_input 会默认当做 KITTI 处理
#             # 导致 InStereo2K 的路径映射（replace）逻辑出错
#             model_input_batch = {
#                 "main": batch["main"],   
#                 "side": batch["side"],  
#                 "side_depth": batch.get("side_depth", None), # 传递原始深度图
#                 "txt": batch.get("txt", [""] * batch["main"].shape[0]),
#                 "main_rel_path": batch.get("main_rel_path", None) # 关键：传递路径用于检索类目判断
#             }
#             try:
#                 log, batch_bpp = model.log_images(model_input_batch, sample_steps=sample_steps, bs=1, fidelity_ratio=0)
                
#                 # 排除 KeyError，仅处理存在的 key
#                 pred = log["samples"].clamp(0, 1)
#                 target = log["target"].clamp(0, 1)
#                 # Inference
#                 log, batch_bpp = model.log_images(model_input_batch, sample_steps=sample_steps, bs=1,fidelity_ratio=0)
                
#                 # pred/target 都在 [0, 1] 范围内
#                 pred = log["samples"].clamp(0, 1)
#                 target = log["target"].clamp(0, 1)
#                 side_image=log["side_image"].clamp(0, 1)
                
#                 # --- 计算各项指标 ---
#                 bpp = batch_bpp.item() if isinstance(batch_bpp, torch.Tensor) else batch_bpp
#                 psnr = metric_funcs["psnr"](pred, target).item()
#                 ms_ssim_val = metric_funcs["ms_ssim"](pred, target).item()
#                 ms_ssim_db = msssim_to_db(ms_ssim_val) # 计算 dB
#                 lpips = metric_funcs["lpips"](pred, target).item()
#                 dists = metric_funcs["dists"](pred, target).item()
                
#                 # 记录到列表
#                 metrics["psnr"].append(psnr)
#                 metrics["ms_ssim"].append(ms_ssim_val)
#                 metrics["ms_ssim_db"].append(ms_ssim_db)
#                 metrics["lpips"].append(lpips)
#                 metrics["dists"].append(dists)
#                 metrics["bpp"].append(bpp)
                
#                 # --- 2. 写入单图指标到 CSV ---
#                 with open(csv_file, "a") as f:
#                     f.write(f"{i},{bpp:.6f},{psnr:.6f},{ms_ssim_val:.6f},{ms_ssim_db:.6f},{lpips:.6f},{dists:.6f}\n")
                
#                 # Save Image (Pred & GT)
#                 # 使用 auto_norm=True 确保可视化清晰 (仅为了看清，不影响指标)
#                 pred_img = tensor2img(log["samples"][0], auto_norm=True)
#                 Image.fromarray(pred_img).save(os.path.join(save_dir, f"{i}.png"))
                
#                 # target_img = tensor2img(log["target"][0], auto_norm=False) 
#                 # Image.fromarray(target_img).save(os.path.join(save_dir_gt, f"{i}.png"))

#                 # side_img = tensor2img(log["side_image"][0], auto_norm=False) 
#                 # Image.fromarray(side_img).save(os.path.join(save_dir_side_gt, f"{i}.png"))
#             except Exception as e:
#                 print(f"❌ Sample {i} failed: {e}")
#                 continue
#     # --- 3. 计算并保存综合平均指标 (TXT) ---
#     avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
#     summary_file = os.path.join(metric_dir, "average_metrics.txt")
#     summary_content = (
#         f"Dataset Evaluation Summary\n"
#         f"--------------------------\n"
#         f"Samples     : {len(loader)}\n"
#         f"BPP         : {avg_metrics['bpp']:.4f}\n"
#         f"PSNR        : {avg_metrics['psnr']:.4f}\n"
#         f"MS-SSIM     : {avg_metrics['ms_ssim']:.4f}\n"
#         f"MS-SSIM(dB) : {avg_metrics['ms_ssim_db']:.4f}\n"
#         f"LPIPS       : {avg_metrics['lpips']:.4f}\n"
#         f"DISTS       : {avg_metrics['dists']:.4f}\n"
#     )
#     with open(summary_file, "w") as f:
#         f.write(summary_content)
        
#     print("-" * 30)
#     print(summary_content)
#     print("-" * 30)
    
#     return avg_metrics

# def main():
#     parser = get_parser()
#     args = parser.parse_args()
    
#     device = torch.device(args.device)
    
#     # 1. Load Model & Config
#     model, config = load_model(args.config, args.ckpt, device)
#     model.fixed_step = args.sample_steps 
    
#     # 从 Config 提取超参数构建文件夹名后缀
#     l_guide = config.params.get("l_guide_weight")
#     l_bpp = config.params.get("l_bpp_weight")
    
#     suffix = f"guideW{l_guide}_bppW{l_bpp}"
#     print(f"Experiment Suffix: {suffix}")
    
#     # 2. Prepare Metrics
#     print("Preparing Metrics...")
#     metric_funcs = {
#         "psnr": pyiqa.create_metric("psnr", device=device),
#         "ms_ssim": pyiqa.create_metric("ms_ssim", device=device),
#         "lpips": pyiqa.create_metric("lpips", device=device, net='vgg'), 
#         "dists": pyiqa.create_metric("dists", device=device)
#     }
    
#     resize_hw = tuple(args.resize)
    
#     # ================= EVALUATION LOOP ================= #

# #     # 1. KITTI Stereo
# #     print("\n>>> Evaluating KITTI Stereo...")
# #     kitti_stereo_ds = PairKitti(
# #         path=args.kitti_root,
# #         set_type="test", 
# #         stereo=True,
# #         resize=resize_hw,
# #         single_view=False
# #     )
# #     # total_len = len(kitti_stereo_ds)
# #     # subset_len = int(total_len * 0.1)  # 或者 total_len // 10
# #     # kitti_stereo_subset = Subset(kitti_stereo_ds, range(subset_len))
# #     # kitti_stereo_loader = DataLoader(kitti_stereo_subset, batch_size=1, shuffle=False, num_workers=4, collate_fn=kitti_collate)
    
# #     kitti_stereo_loader = DataLoader(kitti_stereo_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=kitti_collate)   
# #     evaluate_dataset(
# #         model, kitti_stereo_loader, device, metric_funcs, 
# #         save_dir=os.path.join(args.output_base_dir, f"KITTI_Stereo_{suffix}"),
# #         save_dir_gt=os.path.join(args.output_base_dir, "KITTI_Stereo_gt"),
# #         save_dir_side_gt=os.path.join(args.output_base_dir, "KITTI_Stereo_side_gt"),
# #         metric_dir=os.path.join(args.output_base_dir, f"KITTI_Stereo_{suffix}_metric"),
# #         sample_steps=args.sample_steps
# #     )
# # # '''   
# #     # 2. KITTI General
# #     print("\n>>> Evaluating KITTI General...")
# #     kitti_general_ds = PairKitti(
# #         path=args.kitti_root,
# #         set_type="test", 
# #         stereo=False, 
# #         resize=resize_hw,
# #         single_view=False 
# #     )
# #     # total_len = len(kitti_general_ds)
# #     # subset_len = int(total_len * 0.1)  
# #     # kitti_general_subset = Subset(kitti_general_ds, range(subset_len))
# #     # kitti_general_loader = DataLoader(kitti_general_subset, batch_size=1, shuffle=False, num_workers=4, collate_fn=kitti_collate)
# #     kitti_general_loader = DataLoader(kitti_general_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=kitti_collate)   
# #     evaluate_dataset(
# #         model, kitti_general_loader, device, metric_funcs, 
# #         save_dir=os.path.join(args.output_base_dir, f"KITTI_General_{suffix}"),
# #         save_dir_gt=os.path.join(args.output_base_dir, "KITTI_General_gt"),
# #         save_dir_side_gt=os.path.join(args.output_base_dir, "KITTI_General_side_gt"),
# #         metric_dir=os.path.join(args.output_base_dir, f"KITTI_General_{suffix}_metric"),
# #         sample_steps=args.sample_steps
# #     )
    
#     # 3. Cityscapes
#     print("\n>>> Evaluating Cityscapes...")
#     city_ds = PairCityscape(
#         path=args.city_root,
#         set_type="test", 
#         resize=resize_hw,
#         single_view=False 
#     )

#     # total_len = len(city_ds)
#     # subset_len = int(total_len * 0.1)  
#     # city_subset = Subset(city_ds, range(subset_len))
#     # city_loader = DataLoader(city_subset, batch_size=1, shuffle=False, num_workers=4, collate_fn=city_collate)

#     # city_loader = DataLoader(city_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=city_collate)
    
#     # evaluate_dataset(
#     #     model, city_loader, device, metric_funcs, 
#     #     save_dir=os.path.join(args.output_base_dir, f"Cityscapes_{suffix}"),
#     #     save_dir_gt=os.path.join(args.output_base_dir, "Cityscapes_gt"),
#     #     save_dir_side_gt=os.path.join(args.output_base_dir, "Cityscapes_side_gt"),
#     #     metric_dir=os.path.join(args.output_base_dir, f"Cityscapes_{suffix}_metric"),
#     #     sample_steps=args.sample_steps
#     # )

#     # ================= 4. InStereo2K (新增) ================= #
#     print("\n>>> Evaluating InStereo2K...")
#     try:
#         instereo_ds = PairInstereo2K(
#             path=args.instereo_root,
#             set_type="test", 
#             resize=resize_hw
#         )
        
#         # 检查是否找到数据
#         if len(instereo_ds) > 0:
#             instereo_loader = DataLoader(instereo_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=instereo_collate)
            
#             evaluate_dataset(
#                 model, instereo_loader, device, metric_funcs, 
#                 save_dir=os.path.join(args.output_base_dir, f"InStereo2K_{suffix}"),
#                 save_dir_gt=os.path.join(args.output_base_dir, "InStereo2K_gt"),
#                 save_dir_side_gt=os.path.join(args.output_base_dir, "InStereo2K_side_gt"),
#                 metric_dir=os.path.join(args.output_base_dir, f"InStereo2K_{suffix}_metric"),
#                 sample_steps=args.sample_steps
#             )
#         else:
#             print("Skipping InStereo2K: No images found.")
            
#     except Exception as e:
#         print(f"Skipping InStereo2K: {e}")

# # '''
# if __name__ == "__main__":
#     main()





#-------------------------------------------------------------------------


import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pyiqa
import traceback
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF

if not hasattr(torch, 'compiler'):
    class DummyCompiler:
        @staticmethod
        def disable(func): return func
    torch.compiler = DummyCompiler()
    print("⚠️ [Patch Applied] torch.compiler injected manually.")

from datasets1.kitti_dataset import PairKitti, collate_fn as kitti_collate
from datasets1.cityscapes_dataset import PairCityscape, collate_fn as city_collate
from datasets1.instereo2k_dataset import PairInstereo2K, collate_fn as instereo_collate
from utils1.utils import instantiate_from_config

def msssim_to_db(msssim_val):
    if msssim_val >= 1.0 - 1e-9: return 100.0 
    return -10.0 * np.log10(1.0 - msssim_val)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/model/rdeic.yaml")
    parser.add_argument("--ckpt", type=str, default="logs_pro_MSI/final_2_4/checkpoints/best_psnr.pth")
    parser.add_argument("--kitti_root", type=str, default="dataTVT/KITTI")
    parser.add_argument("--city_root", type=str, default="cityscape_dataset")
    parser.add_argument("--instereo_root", type=str, default="dataTVT/InStereo2K")    
    parser.add_argument("--output_base_dir", type=str, default="results_pro_MSI_4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 512]) 
    parser.add_argument("--sample_steps", type=int, default=2) 
    return parser

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval() 
    return model, config 

def tensor2img(tensor, auto_norm=False):
    # tensor: [1, 3, H, W]
    img = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    if auto_norm:
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def evaluate_dataset(model, loader, device, metric_funcs, save_dir, metric_dir, sample_steps):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)   
    csv_file = os.path.join(metric_dir, "detailed_metrics.csv")
    with open(csv_file, "w") as f:
        f.write("Index,BPP,PSNR,MS_SSIM,MS_SSIM_dB,LPIPS,DISTS\n")
    
    metrics = {"psnr": [], "ms_ssim": [], "ms_ssim_db": [], "lpips": [], "dists": [], "bpp": []}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Eval {os.path.basename(save_dir)}")):
            try:
                for k in ['main', 'side', 'side_depth']:
                    if k in batch and batch[k] is not None:
                        if batch[k].ndim == 4 and batch[k].shape[-1] in [1, 3]:
                            batch[k] = batch[k].permute(0, 3, 1, 2)
                        batch[k] = batch[k].to(device).float()

                model_input = {
                    "main": batch["main"],   
                    "side": batch["side"],  
                    "side_depth": batch.get("side_depth", None),
                    "txt": [""] * batch["main"].shape[0],
                }

                vis_file_path = os.path.join(save_dir, f"{i}_retrieval_vis.png")

                log, batch_bpp = model.log_images(
                    model_input, 
                    sample_steps=sample_steps, 
                    bs=1, 
                    fidelity_ratio=1.0,
                    vis_path=vis_file_path 
                )
                
                pred = log["samples"].clamp(0, 1)
                target = log["target"].clamp(0, 1)
                bpp = batch_bpp.item() if isinstance(batch_bpp, torch.Tensor) else batch_bpp
                psnr = metric_funcs["psnr"](pred, target).item()
                ms_val = metric_funcs["ms_ssim"](pred, target).item()
                ms_db = msssim_to_db(ms_val)
                lpips = metric_funcs["lpips"](pred, target).item()
                dists = metric_funcs["dists"](pred, target).item()
                
                for k, v in zip(["psnr", "ms_ssim", "ms_ssim_db", "lpips", "dists", "bpp"], [psnr, ms_val, ms_db, lpips, dists, bpp]):
                    metrics[k].append(v)
                
                with open(csv_file, "a") as f:
                    f.write(f"{i},{bpp:.6f},{psnr:.6f},{ms_val:.6f},{ms_db:.6f},{lpips:.6f},{dists:.6f}\n")
                

                Image.fromarray(tensor2img(pred[0])).save(os.path.join(save_dir, f"{i}.png"))

            except Exception as e:
                print(f"\n❌ Sample {i} failed: {e}")
                # traceback.print_exc()
                continue


    if len(metrics["psnr"]) > 0:
        avg = {k: np.mean(v) for k, v in metrics.items()}
        with open(os.path.join(metric_dir, "average_metrics.txt"), "w") as f:
            f.write(str(avg))
        print(f"\n>>> Final Results: {avg}")

def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device(args.device)
    
    model, config = load_model(args.config, args.ckpt, device)
    
    metric_funcs = {
        "psnr": pyiqa.create_metric("psnr", device=device),
        "ms_ssim": pyiqa.create_metric("ms_ssim", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device, net='vgg'), 
        "dists": pyiqa.create_metric("dists", device=device)
    }
    
    hw = tuple(args.resize)
    suffix = f"steps{args.sample_steps}"

    
    test_configs = [
        ("KITTI_Stereo", PairKitti(args.kitti_root, "test", stereo=True, resize=hw), kitti_collate),
        # ("KITTI_General", PairKitti(args.kitti_root, "test", stereo=False, resize=hw), kitti_collate),
        # ("Cityscapes", PairCityscape(args.city_root, "test", resize=hw), city_collate),
        # ("InStereo2K", PairInstereo2K(args.instereo_root, "test", resize=hw), instereo_collate),
    ]

    for name, ds, collate in test_configs:
        print(f"\n>>> Running {name} Test...")
        if len(ds) == 0:
            print(f"Empty dataset for {name}, skipping.")
            continue
            
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
        
        evaluate_dataset(
            model, loader, device, metric_funcs, 
            save_dir=os.path.join(args.output_base_dir, f"{name}_{suffix}"),
            metric_dir=os.path.join(args.output_base_dir, f"{name}_{suffix}_metrics"),
            sample_steps=args.sample_steps
        )

if __name__ == "__main__":
    main()