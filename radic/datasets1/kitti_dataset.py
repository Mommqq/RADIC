# from random import random
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
# import torchvision.transforms.functional as TF
# from PIL import Image
# from torch.utils.data._utils.collate import default_collate
# import numpy as np

# class PairKitti(Dataset):
#     def __init__(self, path, set_type, stereo=True, resize=(128, 256), single_view=True, view="left"):
#         super().__init__()
#         # ✅ 自动保证 resize = (H, W)
#         if resize[0] > resize[1]:
#             resize = (resize[1], resize[0])
#         self.resize = resize

#         self.single_view = single_view
#         self.view = view
#         self.ar = []

#         idx_path = path + '/data_paths/' + ('KITTI_stereo_' if stereo else 'KITTI_general_') + set_type + '.txt'
#         with open(idx_path) as f:
#             content = f.readlines()

#         for i in range(0, len(content), 2):
#             left_id = content[i].strip()
#             right_id = content[i + 1].strip()
#             self.ar.append((path + '/' + left_id, path + '/' + right_id))

#         if set_type == 'train':
#             self.transform = self.train_transform
#         else:
#             self.transform = self.val_transform

#     def train_transform(self, img, side_img):
#         img = TF.center_crop(img, (370, 740))
#         side_img = TF.center_crop(side_img, (370, 740))
#         img = TF.resize(img, self.resize)
#         side_img = TF.resize(side_img, self.resize)
#         if random() > 0.5:
#             img = TF.hflip(img)
#             side_img = TF.hflip(side_img)
#         img = transforms.ToTensor()(img)
#         side_img = transforms.ToTensor()(side_img)
#         return img, side_img

#     def val_transform(self, img, side_img):
#         img = TF.center_crop(img, (370, 740))
#         side_img = TF.center_crop(side_img, (370, 740))
#         img = TF.resize(img, self.resize)
#         side_img = TF.resize(side_img, self.resize)
#         img = transforms.ToTensor()(img)
#         side_img = transforms.ToTensor()(side_img)
#         return img, side_img

#     def __getitem__(self, index):
#         # ✅ 如果是 both 模式，ar 中是单路径
#         # if self.single_view and self.view == "both":
#         #     img_path = self.ar[index]
#         #     img = Image.open(img_path).convert("RGB")
#         #     img = self.transform(img)
#         #     cond = img
#         #     img = img * 2 - 1
#         #     img = img.permute(1, 2, 0)
#         #     return {"main": img, "side": cond, "txt": "", "index": index}

#         # ✅ 否则仍用左右对结构
#         left_path, right_path = self.ar[index]
#         left_img = Image.open(left_path).convert("RGB")
#         right_img = Image.open(right_path).convert("RGB")

#         img_l,img_r = self.transform(left_img,right_img)


#         if self.single_view:
#             if self.view == "left":
#                 img = img_l
#             elif self.view == "right":
#                 img = img_r
#             else:
#                 raise ValueError("Invalid view, choose from ['left', 'right', 'both']")
#             cond = img
#             img = img * 2 - 1
#             img = img.permute(1, 2, 0)
#             return {"main": img, "side": cond, "txt": "", "index": index}
#         else:
#             # print("distributed image compression dataset processing ......")
#             img = img_l * 2 - 1
#             # print("img:",img.shape) # [3, 256, 512]
#             # img = img_l.permute(1, 2, 0)           
#             cond = img_r * 2 - 1
#             # cond = img_r.permute(1, 2, 0)             
#             return {"main": img, "side": cond, "txt": "", "index": index}

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


# # if __name__ == "__main__":
# #     ds = PairKitti(path="./", set_type="train", resize=(128, 256))
# #     dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
# #     for b in dl:
# #         print("jpg:", b["jpg"].shape, "hint:", b["hint"].shape)
# #         break


#-------------------------------------------original------------------------------------------------------
# import os
# import numpy as np
# from random import random
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
# import torchvision.transforms.functional as TF
# from PIL import Image
# from torch.utils.data._utils.collate import default_collate
# import torch

# class PairKitti(Dataset):
#     def __init__(self, path, set_type, stereo=False, resize=(128, 256), single_view=False, view="left"):
#         super().__init__()
#         # ✅ 自动保证 resize = (H, W)
#         if resize[0] > resize[1]:
#             resize = (resize[1], resize[0])
#         self.resize = resize

#         self.path = path  # 记录根路径，例如 ./data/KITTI
#         self.set_type = set_type # 记录是 train 还是 val
#         self.single_view = single_view
#         self.view = view
        
#         # 假设深度图根目录为原目录名 + "_depth"，例如 ./data/KITTI_depth
#         # 你可以根据实际存放位置修改这里
#         self.depth_root = path.rstrip('/') + "_depth" 

#         self.ar = []

#         # 读取 TXT 列表
#         # 注意：这里 stereo=False 时读取 KITTI_general_xxx.txt
#         idx_path = os.path.join(path, 'data_paths', ('KITTI_stereo_' if stereo else 'KITTI_general_') + set_type + '.txt')
        
#         with open(idx_path) as f:
#             content = f.readlines()

#         for i in range(0, len(content), 2):
#             left_rel = content[i].strip()   # 相对路径，例如 data_stereo/training/image_2/000.png
#             right_rel = content[i + 1].strip()
#             # 保存相对路径，以便灵活构造深度图路径
#             self.ar.append((left_rel, right_rel))

#     def apply_transforms(self, img_l, img_r, depth_r):
#         """
#         统一处理几何变换，确保 RGB 和 Depth 对齐
#         """
#         # 1. Random Crop (Center Crop 也可以，这里保持你原本的逻辑)
#         # 注意：Training 时通常用 RandomCrop，这里你原本用的是 CenterCrop
#         # 为了严谨，这里对所有输入做相同的 Crop
#         crop_h, crop_w = 370, 740
        
#         # 定义裁剪参数 (如果需要随机裁剪，这里要生成一次参数然后应用到三个图)
#         # 这里沿用你之前的 CenterCrop
#         img_l = TF.center_crop(img_l, (crop_h, crop_w))
#         img_r = TF.center_crop(img_r, (crop_h, crop_w))
#         if depth_r is not None:
#             depth_r = TF.center_crop(depth_r, (crop_h, crop_w))

#         # 2. Resize
#         img_l = TF.resize(img_l, self.resize)
#         img_r = TF.resize(img_r, self.resize)
#         if depth_r is not None:
#             # 深度图使用双线性插值 (Bilinear) 或 最近邻 (Nearest)
#             # 这里的 resize 必须和 RGB 一致
#             depth_r = TF.resize(depth_r, self.resize, interpolation=transforms.InterpolationMode.BILINEAR)

#         # 3. Random Horizontal Flip (仅训练时)
#         if self.set_type == 'train' and random() > 0.5:
#             img_l = TF.hflip(img_l)
#             img_r = TF.hflip(img_r)
#             if depth_r is not None:
#                 depth_r = TF.hflip(depth_r)

#         # 4. ToTensor & Normalize (RGB)
#         # 图片转为 Tensor [0, 1]
#         img_l = transforms.ToTensor()(img_l)
#         img_r = transforms.ToTensor()(img_r)
        
#         return img_l, img_r, depth_r

#     def __getitem__(self, index):
#         # 获取相对路径
#         left_rel, right_rel = self.ar[index]
        
#         # 1. 读取 RGB 图像
#         left_full = os.path.join(self.path, left_rel)
#         right_full = os.path.join(self.path, right_rel)
        
#         left_img = Image.open(left_full).convert("RGB")
#         right_img = Image.open(right_full).convert("RGB")

#         # 2. 读取深度图 (仅在训练时读取 Side View/Right 的深度)
#         side_depth_tensor = None
#         has_depth = False
        
#         if self.set_type == 'train' or self.set_type == 'val':
#             # 构造深度图路径：替换根目录，替换后缀
#             # 例如: ./data/KITTI/data_stereo/... -> ./data/KITTI_depth/data_stereo/...
#             # 并把 .png 换成 .npy
#             depth_rel = os.path.splitext(right_rel)[0] + ".npy"
#             depth_full = os.path.join(self.depth_root, depth_rel)
            
#             try:
#                 # 加载 npy [H, W] float32
#                 depth_np = np.load(depth_full)
#                 # 转为 Tensor 并增加 Channel 维 [1, H, W]
#                 side_depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
#                 has_depth = True
#             except Exception:
#                 # 如果没找到文件，就保持 None，后续处理
#                 pass

#         # 3. 执行同步变换
#         img_l, img_r, side_depth_tensor = self.apply_transforms(left_img, right_img, side_depth_tensor)

#         # 4. 深度图归一化 (到 0-1)
#         if side_depth_tensor is not None:
#             d_min = side_depth_tensor.min()
#             d_max = side_depth_tensor.max()
#             if d_max - d_min > 1e-6:
#                 side_depth_tensor = (side_depth_tensor - d_min) / (d_max - d_min)
#         else:
#             # 如果是验证集或没读到深度，给一个全 0 的占位符
#             # 形状要匹配 (1, H, W)
#             side_depth_tensor = torch.zeros((1, self.resize[0], self.resize[1]))
#             print("no side_depth")
#         # 5. RGB 归一化到 [-1, 1]
#         if self.single_view:
#             pass
#             # if self.view == "left":
#             #     img = img_l
#             #     cond = img_l
#             # elif self.view == "right":
#             #     img = img_r
#             #     cond = img_r
#             # else:
#             #     raise ValueError("Invalid view")
            
#             # img = img * 2 - 1
#             # # img = img.permute(1, 2, 0) 
#             # return {
#             #     "jpg": img, 
#             #     "hint": cond * 2 - 1, # 统一格式
#             #     "side_depth": side_depth_tensor,
#             #     "has_depth": has_depth,
#             #     "txt": "", 
#             #     "index": index
#             # }
#         else:
#             # Distributed / Stereo Mode
#             # main: 左图, side: 右图
#             main_img = img_l * 2 - 1
#             # main_img = main_img.permute(1, 2, 0) 
            
#             side_img = img_r * 2 - 1
#             # side_img = side_img.permute(1, 2, 0)
            
#             return {
#                 "main": main_img, 
#                 "side": side_img, 
#                 "side_depth": side_depth_tensor,
#                 "has_depth": has_depth, 
#                 "txt": "", 
#                 "index": index
#             }

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




# if __name__ == "__main__":
#     # 测试代码
#     # 假设你的目录结构:
#     # ./data/KITTI/... (images)
#     # ./data/KITTI_depth/... (npy files)
#     ds = PairKitti(path="./data/KITTI", set_type="train", stereo=False, resize=(256, 512))
#     dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    
#     for i, b in enumerate(dl):
#         print(f"Batch {i}:")
#         print("Main shape:", b["main"].shape)       # Should be [4, 3, 256, 512]
#         print("Side shape:", b["side"].shape)       # Should be [4, 3, 256, 512]
#         print("Depth shape:", b["side_depth"].shape)# Should be [4, 1, 256, 512]
#         print("Has Depth:", b["has_depth"])
#         break
    
#------------------------------------------------------original-----------------------------------------------------------



import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data._utils.collate import default_collate
import random

class PairKitti(Dataset):
    def __init__(self, path, set_type, stereo=False, resize=(256, 512), single_view=False, view="left"):
        super().__init__()

        self.resize = (resize[1], resize[0]) if resize[0] > resize[1] else resize
        self.path = path
        self.set_type = set_type
        self.depth_root = path.rstrip('/') + "_depth" 

        self.ar = []
        idx_path = os.path.join(path, 'data_paths', ('KITTI_stereo_' if stereo else 'KITTI_general_') + set_type + '.txt')
        with open(idx_path) as f:
            content = f.readlines()

        for i in range(0, len(content), 2):
            self.ar.append((content[i].strip(), content[i + 1].strip()))

    def __getitem__(self, index):
        left_rel, right_rel = self.ar[index]
        
        # 1. 加载图像
        main_img = Image.open(os.path.join(self.path, left_rel)).convert("RGB")
        side_img = Image.open(os.path.join(self.path, right_rel)).convert("RGB")

        # 2. 深度图加载逻辑 (KITTI -> KITTI_depth, .png -> .npy)
        depth_rel = os.path.splitext(right_rel)[0] + ".npy"
        depth_path = os.path.join(self.depth_root, depth_rel)
        if os.path.exists(depth_path):
            side_depth = torch.from_numpy(np.load(depth_path)).unsqueeze(0).float()
            # 归一化
            side_depth = (side_depth - side_depth.min()) / (side_depth.max() - side_depth.min() + 1e-6)
        else:
            side_depth = torch.zeros((1, 375, 1242)) # 占位
            print("no side_depth")

        # 3. 几何变换 (统一 Crop 和 Resize)
        w, h = main_img.size
        th, tw = 370, 740
        i, j = (h - th) // 2, (w - tw) // 2
        # flip = (self.set_type == 'train' and random.random() > 0.5)

        def transform(img, is_depth=False):
            img = TF.crop(img, i, j, th, tw)
            img = TF.resize(img, self.resize, interpolation=transforms.InterpolationMode.BILINEAR if is_depth else transforms.InterpolationMode.BICUBIC)
            # if flip: img = TF.hflip(img)
            return torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0 if not is_depth else img

        main_tensor = transform(main_img) * 2 - 1
        side_tensor = transform(side_img) * 2 - 1
        depth_tensor = transform(side_depth, is_depth=True)

        return {
            "main": main_tensor, 
            "side": side_tensor, 
            "side_depth": depth_tensor,
            "txt": "", 
            "index": index,
            "main_rel_path": left_rel 
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
