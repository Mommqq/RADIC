import torch
import numpy as np
import faiss
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from utils1.utils import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn.functional as F

def build_gallery(model_config_path, ckpt_path, dataset_roots, save_path):
    config = OmegaConf.load(model_config_path)
    model = instantiate_from_config(config).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)

    all_image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for root in dataset_roots:
        if not os.path.exists(root): continue
        for r, d, files in os.walk(root):
            for f in files:
                if f.lower().endswith(valid_extensions) and "_disp" not in f.lower():
                    full_p = os.path.join(r, f)
                    if "instereo2k" in full_p.lower() and f.lower() != "right.png": continue
                    all_image_paths.append(full_p)

    dim = 512 
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
    metadata = []
    target_size = (256, 512)

    print(f"开始构建固定特征库，总计: {len(all_image_paths)} 张")

    with torch.no_grad():
        for i, path in enumerate(tqdm(all_image_paths)):
            try:
                img = Image.open(path).convert('RGB')
                # 严格匹配数据集几何处理
                if "KITTI" in path.upper():
                    img = TF.center_crop(img, (370, 740))
                img = TF.resize(img, target_size, interpolation=TF.InterpolationMode.BICUBIC)
                
                # RGB [0,1] -> [-1, 1]
                img_t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
                img_t = (img_t * 2 - 1).unsqueeze(0).cuda()
                _, h_fixed = model.encode_first_stage(img_t)
                feat = torch.mean(h_fixed, dim=(2, 3)) # [1, 512]
                feat = F.normalize(feat, p=2, dim=1) 
                
                index.add_with_ids(feat.cpu().numpy().astype('float32'), np.array([i]).astype('int64'))
                metadata.append(os.path.abspath(path))
                
            except Exception as e:
                print(f"Error {path}: {e}")
                continue

    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, os.path.join(save_path, "gallery.index"))
    np.save(os.path.join(save_path, "metadata.npy"), np.array(metadata))
    print(f"✅ 固定特征库已保存至 {save_path}")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

    dataset_list = [
        "dataTVT/KITTI/data_scene_flow_multiview/training/image_3",
        "dataTVT/KITTI/data_scene_flow_multiview/testing/image_3",
        "dataTVT/KITTI/data_stereo_flow_multiview/training/image_3",
        "dataTVT/KITTI/data_stereo_flow_multiview/testing/image_3",
        "cityscape_dataset/rightImg8bit",
        "dataTVT/InStereo2K/test",
        "instereo2k" 
    ]
    build_gallery(
        model_config_path="configs/model/rdeic.yaml", 
        ckpt_path="logs_pro_MSI/experiment_2_0.5_pro/checkpoints/best_psnr.pth", 
        dataset_roots=dataset_list, 
        save_path="retrieval_db"
    )