# import os
# from random import random
# from torch.utils.data import Dataset
# from torchvision import transforms
# import torchvision.transforms.functional as TF
# from PIL import Image
# from torch.utils.data._utils.collate import default_collate

# class PairCityscape(Dataset):
#     def __init__(self, path, set_type, resize=(128, 256), single_view=True, view='left'):
#         super().__init__()
#         self.resize = resize
#         self.single_view = single_view
#         self.view = view

#         # 路径结构适配 Cityscapes 官方结构
#         # path/leftImg8bit/train/city/xxx.png
#         self.dataset = {
#             'left': os.path.join(path, 'leftImg8bit', set_type),
#             'right': os.path.join(path, 'rightImg8bit', set_type)
#         }

#         # 遍历所有城市和图片
#         self.ar = []
#         if os.path.exists(self.dataset['left']):
#             self.cities = sorted([
#                 item for item in os.listdir(self.dataset['left'])
#                 if os.path.isdir(os.path.join(self.dataset['left'], item))
#             ])
            
#             for city in self.cities:
#                 left_dir = os.path.join(self.dataset['left'], city)
#                 right_dir = os.path.join(self.dataset['right'], city)
                
#                 # 获取文件名标识符 (xxx_leftImg8bit.png -> xxx)
#                 pair_names = sorted([
#                     '_'.join(f.split('_')[:-1])
#                     for f in os.listdir(left_dir)
#                     if f.endswith('_leftImg8bit.png')
#                 ])
                
#                 for pair in pair_names:
#                     left_img = os.path.join(left_dir, pair + '_leftImg8bit.png')
#                     right_img = os.path.join(right_dir, pair + '_rightImg8bit.png')
#                     # 确保右图也存在
#                     if os.path.exists(right_img):
#                         self.ar.append((left_img, right_img))
        
#         self.ar = sorted(self.ar)

#         if set_type == 'train':
#             self.transform = self.train_transform
#         else:
#             self.transform = self.eval_transform

#     def train_transform(self, img, side_img):
#         # img = TF.center_crop(img, (370, 740))
#         # side_img = TF.center_crop(side_img, (370, 740))
#         img = TF.resize(img, self.resize)
#         side_img = TF.resize(side_img, self.resize)
#         if random() > 0.5:
#             img = TF.hflip(img)
#             side_img = TF.hflip(side_img)
#         img = transforms.ToTensor()(img)
#         side_img = transforms.ToTensor()(side_img)
#         return img, side_img

#     def eval_transform(self, img, side_img):
#         # img = TF.center_crop(img, (370, 740))
#         # side_img = TF.center_crop(side_img, (370, 740))
#         img = TF.resize(img, self.resize)
#         side_img = TF.resize(side_img, self.resize)
#         img = transforms.ToTensor()(img)
#         side_img = transforms.ToTensor()(side_img)
#         return img, side_img

#     def __getitem__(self, index):
#         left_path, right_path = self.ar[index]
#         left_img = Image.open(left_path).convert("RGB")
#         right_img = Image.open(right_path).convert("RGB")

#         # 同时变换两张图
#         img_l, img_r = self.transform(left_img, right_img)

#         # 逻辑对齐 PairKitti
#         if self.single_view:
#             if self.view == "left":
#                 img = img_l
#                 cond = img_l # single view 时 hint 通常就是 GT 本身
#             elif self.view == "right":
#                 img = img_r
#                 cond = img_r
#             else:
#                 raise ValueError("Invalid view")
            
#             # 归一化到 [-1, 1] 并 HWC
#             img = img * 2 - 1
#             # img = img.permute(1, 2, 0) # 如果你的 train.py 里处理了 permute，这里可以保留；建议在 Dataset 里统一不转，交给 DataLoader 后处理
#             # 既然你的 PairKitti 做了 permute，这里保持一致
#             img = img.permute(1, 2, 0)
            
#             return {"jpg": img, "hint": cond, "txt": "", "index": index}
#         else:
#             # Distributed / Stereo Mode
#             # main: 左图, side: 右图
#             main_img = img_l * 2 - 1
#             # main_img = main_img.permute(1, 2, 0)
            
#             side_img = img_r * 2 - 1
#             # side_img = side_img.permute(1, 2, 0)
            
#             return {"main": main_img, "side": side_img, "txt": "", "index": index}

#     def __len__(self):
#         return len(self.ar)

# def collate_fn(batch):
#     collated = {}
#     for key in batch[0]:
#         try:
#             collated[key] = default_collate([d[key] for d in batch])
#         except Exception:
#             collated[key] = [d[key] for d in batch]
#     return collated




import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data._utils.collate import default_collate
import random

class PairCityscape(Dataset):
    def __init__(self, path, set_type, resize=(256, 512), single_view=False, view='left'):
        super().__init__()
        
        # 统一尺寸格式 (H, W)
        self.resize = (resize[1], resize[0]) if resize[0] > resize[1] else resize
        self.single_view = single_view
        self.view = view
        self.path = path
        self.set_type = set_type

        # 深度图根目录：同级目录下的 _depth 文件夹
        self.depth_root = path.rstrip('/') + "_depth"

        # 路径结构适配 Cityscapes 官方结构
        # path/leftImg8bit/train/city/xxx.png
        self.dataset = {
            'left': os.path.join(path, 'leftImg8bit', set_type),
            'right': os.path.join(path, 'rightImg8bit', set_type)
        }

        self.ar = []
        if os.path.exists(self.dataset['left']):
            self.cities = sorted([
                item for item in os.listdir(self.dataset['left'])
                if os.path.isdir(os.path.join(self.dataset['left'], item))
            ])
            
            for city in self.cities:
                left_dir = os.path.join(self.dataset['left'], city)
                right_dir = os.path.join(self.dataset['right'], city)
                
                # 获取文件名标识符 (xxx_leftImg8bit.png -> xxx)
                pair_names = sorted([
                    '_'.join(f.split('_')[:-1])
                    for f in os.listdir(left_dir)
                    if f.endswith('_leftImg8bit.png')
                ])
                
                for pair in pair_names:
                    left_img = os.path.join(left_dir, pair + '_leftImg8bit.png')
                    right_img = os.path.join(right_dir, pair + '_rightImg8bit.png')
                    # 确保左右图都存在
                    if os.path.exists(right_img) and os.path.exists(left_img):
                        self.ar.append((left_img, right_img))
        
        self.ar = sorted(self.ar)

    def __getitem__(self, index):
        left_path, right_path = self.ar[index]
        
        # 1. 加载图像
        main_img = Image.open(left_path).convert("RGB")
        side_img = Image.open(right_path).convert("RGB")

        # 2. 深度图加载逻辑 (寻找对应的 .npy 文件)
        # 相对路径如: rightImg8bit/val/frankfurt/frankfurt_xxx_rightImg8bit.png
        rel_right_path = os.path.relpath(right_path, self.path)
        depth_rel = os.path.splitext(rel_right_path)[0] + ".npy"
        depth_path = os.path.join(self.depth_root, depth_rel)
        
        if os.path.exists(depth_path):
            side_depth = torch.from_numpy(np.load(depth_path)).unsqueeze(0).float()
            # 归一化到 [0, 1]
            d_min, d_max = side_depth.min(), side_depth.max()
            if d_max - d_min > 1e-6:
                side_depth = (side_depth - d_min) / (d_max - d_min)
        else:
            # Cityscapes 原始分辨率为 1024x2048
            side_depth = torch.zeros((1, 1024, 2048)) 

        # 3. 几何变换 (物理防畸变 CenterCrop + Resize)
        def transform(img, is_depth=False):
            # Cityscapes 原图是 1024x2048 (1:2)
            # 为了防止某些图尺寸不合规，先做严谨的 CenterCrop 保证 1:2
            if not is_depth:
                w, h = img.size
            else:
                _, h, w = img.shape

            target_w = min(w, h * 2)
            target_h = target_w // 2
            
            img = TF.center_crop(img, (target_h, target_w))
            
            # 再 Resize 到要求的输入尺寸 (例如 256x512)
            interp = transforms.InterpolationMode.BILINEAR if is_depth else transforms.InterpolationMode.BICUBIC
            img = TF.resize(img, self.resize, interpolation=interp)
            
            if not is_depth:
                # RGB: 转 Tensor 并除以 255
                return torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
            else:
                return img

        # RGB 映射到 [-1, 1]，Depth 保持 [0, 1]
        main_tensor = transform(main_img) * 2.0 - 1.0
        side_tensor = transform(side_img) * 2.0 - 1.0
        depth_tensor = transform(side_depth, is_depth=True)

        # 获取相对路径，方便测试端检索和记录
        main_rel_path = os.path.relpath(left_path, self.path)

        # 4. 统一返回值格式 (严格对齐 PairKitti)
        return {
            "main": main_tensor, 
            "side": side_tensor, 
            "side_depth": depth_tensor,
            "txt": "", 
            "index": index,
            "main_rel_path": main_rel_path
        }

    def __len__(self):
        return len(self.ar)

def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        try:
            collated[key] = default_collate([d[key] for d in batch])
        except Exception:
            collated[key] = [d[key] for d in batch]
    return collated