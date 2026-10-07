import os
from random import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class PairCityscape(Dataset):
    def __init__(self, path, set_type, resize=(128, 256), single_view=True, view='both'):
        super().__init__()
        self.resize = resize
        self.single_view = single_view
        self.view = view  # 'left', 'right', 或 'both'

        self.dataset = {
            'left': os.path.join(path, 'leftImg8bit', set_type),
            'right': os.path.join(path, 'rightImg8bit', set_type)
        }

        self.cities = sorted([
            item for item in os.listdir(self.dataset['left'])
            if os.path.isdir(os.path.join(self.dataset['left'], item))
        ])

        self.ar = []
        for city in self.cities:
            left_dir = os.path.join(self.dataset['left'], city)
            right_dir = os.path.join(self.dataset['right'], city)
            pair_names = sorted([
                '_'.join(f.split('_')[:-1])
                for f in os.listdir(left_dir)
                if os.path.splitext(f)[-1].lower() == '.png'
            ])
            for pair in pair_names:
                left_img = os.path.join(left_dir, pair + '_leftImg8bit.png')
                right_img = os.path.join(right_dir, pair + '_rightImg8bit.png')
                self.ar.append((left_img, right_img))

        # ✅ 如果是单图像模式且要使用左右视角，则展开成所有图像
        if self.single_view and self.view == 'both':
            expanded = []
            for left, right in self.ar:
                expanded.append(left)
                expanded.append(right)
            self.ar = expanded  # 每个元素都是单张图像路径
        else:
            # 保持原始结构 (left, right)
            self.ar = sorted(self.ar)

        if set_type == 'train':
            self.transform = self.train_transform
        else:
            self.transform = self.eval_transform

    def train_transform(self, img):
        img = TF.resize(img, self.resize)
        if random() > 0.5:
            img = TF.hflip(img)
        return transforms.ToTensor()(img)

    def eval_transform(self, img):
        img = TF.resize(img, self.resize)
        return transforms.ToTensor()(img)

    def __getitem__(self, index):
        if self.single_view:
            # ✅ 如果是 view='both' 模式，ar 里存的是单张图像路径
            if self.view == 'both':
                img_path = self.ar[index]
            else:
                left_path, right_path = self.ar[index]
                img_path = left_path if self.view == 'left' else right_path

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            cond = img  # 0~1
            img = img * 2 - 1  # [-1,1]
            img = img.permute(1, 2, 0)
            return {"jpg": img, "hint": cond, "txt": "", "index": index}

        else:
            print("Not single-view mode! Double-view mode not implemented here.")

    def __len__(self):
        return len(self.ar)

    def __str__(self):
        return f"Cityscape(single_view={self.single_view}, view={self.view})"
