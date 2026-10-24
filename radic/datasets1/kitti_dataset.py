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

    def __init__(self, path, set_type, stereo=False, resize=(256, 512), single_view=False, view='left'):
        super().__init__()
        self.resize = (resize[1], resize[0]) if resize[0] > resize[1] else resize
        self.path = path
        self.set_type = set_type
        self.depth_root = path.rstrip('/') + '_depth'
        self.ar = []
        idx_path = os.path.join(path, 'data_paths', ('KITTI_stereo_' if stereo else 'KITTI_general_') + set_type + '.txt')
        with open(idx_path) as f:
            content = f.readlines()
        for i in range(0, len(content), 2):
            self.ar.append((content[i].strip(), content[i + 1].strip()))

    def __getitem__(self, index):
        (left_rel, right_rel) = self.ar[index]
        main_img = Image.open(os.path.join(self.path, left_rel)).convert('RGB')
        side_img = Image.open(os.path.join(self.path, right_rel)).convert('RGB')
        depth_rel = os.path.splitext(right_rel)[0] + '.npy'
        depth_path = os.path.join(self.depth_root, depth_rel)
        if os.path.exists(depth_path):
            side_depth = torch.from_numpy(np.load(depth_path)).unsqueeze(0).float()
            side_depth = (side_depth - side_depth.min()) / (side_depth.max() - side_depth.min() + 1e-06)
        else:
            side_depth = torch.zeros((1, 375, 1242))
            print('no side_depth')
        (w, h) = main_img.size
        (th, tw) = (370, 740)
        (i, j) = ((h - th) // 2, (w - tw) // 2)

        def transform(img, is_depth=False):
            img = TF.crop(img, i, j, th, tw)
            img = TF.resize(img, self.resize, interpolation=transforms.InterpolationMode.BILINEAR if is_depth else transforms.InterpolationMode.BICUBIC)
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 if not is_depth else img
        main_tensor = transform(main_img) * 2 - 1
        side_tensor = transform(side_img) * 2 - 1
        depth_tensor = transform(side_depth, is_depth=True)
        return {'main': main_tensor, 'side': side_tensor, 'side_depth': depth_tensor, 'txt': '', 'index': index, 'main_rel_path': left_rel}

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
