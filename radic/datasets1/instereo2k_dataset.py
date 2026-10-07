# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import torchvision.transforms.functional as TF
# from PIL import Image
# from torch.utils.data._utils.collate import default_collate

# class PairInstereo2K(Dataset):
#     def __init__(self, path, set_type="test", resize=(256, 512)):
#         super().__init__()
#         self.resize = resize
#         self.samples = []
        
#         # 你的路径逻辑：InStereo2K/test/Scene1/left.png
#         search_path = os.path.join(path, set_type)
            
#         if os.path.exists(search_path):
#             # 遍历所有子文件夹寻找成对图片
#             for root, dirs, files in os.walk(search_path):
#                 # InStereo2K 的标准命名通常是 left.png 和 right.png
#                 if 'left.png' in files and 'right.png' in files:
#                     self.samples.append((
#                         os.path.join(root, 'left.png'), 
#                         os.path.join(root, 'right.png')
#                     ))
#         else:
#             print(f"Warning: Instereo2K path {search_path} not found.")

#         # 如果需要排序以保证测试顺序一致
#         self.samples.sort()

#     def transform(self, img):
#         # Resize 并转 Tensor [0, 1]
#         img = TF.resize(img, self.resize)
#         return transforms.ToTensor()(img)

#     def __getitem__(self, index):
#         l_path, r_path = self.samples[index]
        
#         left_img = Image.open(l_path).convert('RGB')
#         right_img = Image.open(r_path).convert('RGB')

#         # [0, 1] Tensor
#         img_l = self.transform(left_img)
#         img_r = self.transform(right_img)
        
#         # 归一化到 [-1, 1] 以适配你的模型输入
#         img_l = img_l * 2.0 - 1.0
#         img_r = img_r * 2.0 - 1.0
        
#         # 这里的 HWC/CHW 取决于你的 collate_fn 和 test.py 处理
#         # 根据你之前的 test.py 逻辑 (batch[k].permute)，这里应该返回 HWC 吗？
#         # 不，你的 kitti/cityscapes 在 test.py 里是 HWC -> CHW，说明 Dataset 返回的是 HWC
#         # 但 transforms.ToTensor() 返回的是 CHW
#         # 让我们保持一致：Dataset 返回 HWC，test.py 负责转 CHW
        
#         img_l = img_l.permute(1, 2, 0)
#         img_r = img_r.permute(1, 2, 0)

#         # 返回与 Kitti/Cityscapes 一致的字典格式
#         return {
#             "main": img_l,  # 左图 (Target)
#             "side": img_r,  # 右图 (Reference)
#             "txt": "",      # 空文本
#             "index": index
#         }

#     def __len__(self):
#         return len(self.samples)

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

class PairInstereo2K(Dataset):
    def __init__(self, path, set_type="test", resize=(256, 512)):
        super().__init__()
        self.resize = (resize[1], resize[0]) if resize[0] > resize[1] else resize
        self.path = path
        self.set_type = set_type
        
        self.depth_root = path.rstrip('/') + "_depth" 

        self.samples = []
        search_path = os.path.join(path, set_type)
            
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                if 'left.png' in files and 'right.png' in files:
                    rel_dir = os.path.relpath(root, path)
                    self.samples.append({
                        'left_rel': os.path.join(rel_dir, 'left.png'),
                        'right_rel': os.path.join(rel_dir, 'right.png')
                    })
        else:
            print(f"Warning: InStereo2K path {search_path} not found.")

        self.samples.sort(key=lambda x: x['left_rel'])

    def transform(self, img, is_depth=False):
        img = TF.resize(img, self.resize, 
                        interpolation=transforms.InterpolationMode.BILINEAR if is_depth else transforms.InterpolationMode.BICUBIC)
        
        if is_depth:
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            return img
        else:
            return transforms.ToTensor()(img) * 2.0 - 1.0

    def __getitem__(self, index):
        sample = self.samples[index]
        
        main_img = Image.open(os.path.join(self.path, sample['left_rel'])).convert("RGB")
        side_img = Image.open(os.path.join(self.path, sample['right_rel'])).convert("RGB")
        #  dataTVT/InStereo2K_depth/test/Scene/left.npy
        depth_rel = os.path.splitext(sample['right_rel'])[0] + ".npy"
        depth_path = os.path.join(self.depth_root, depth_rel)
        
        if os.path.exists(depth_path):
            side_depth = torch.from_numpy(np.load(depth_path)).unsqueeze(0).float()
            d_min, d_max = side_depth.min(), side_depth.max()
            if d_max - d_min > 1e-6:
                side_depth = (side_depth - d_min) / (d_max - d_min)
        else:
            side_depth = torch.zeros((1, 375, 1242))
            print("no instereo2k-depth")
        main_tensor = self.transform(main_img, is_depth=False)
        side_tensor = self.transform(side_img, is_depth=False)
        depth_tensor = self.transform(side_depth, is_depth=True)

        return {
            "main": main_tensor,        # [3, H, W]
            "side": side_tensor,        # [3, H, W]
            "side_depth": depth_tensor,  # [1, H, W]
            "txt": "", 
            "index": index,
            "main_rel_path": sample['left_rel'] 
        }

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        try:
            collated[key] = default_collate([d[key] for d in batch])
        except Exception:
            collated[key] = [d[key] for d in batch]
    return collated