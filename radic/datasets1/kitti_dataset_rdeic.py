from random import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data._utils.collate import default_collate
import numpy as np

class PairKitti(Dataset):
    def __init__(self, path, set_type, stereo=True, resize=(128, 256), single_view=True, view="both"):
        super().__init__()
        # ✅ 自动保证 resize = (H, W)
        if resize[0] > resize[1]:
            resize = (resize[1], resize[0])
        self.resize = resize

        self.single_view = single_view
        self.view = view  # 'left', 'right', or 'both'
        self.ar = []

        idx_path = path + '/data_paths/' + ('KITTI_stereo_' if stereo else 'KITTI_general_') + set_type + '.txt'
        with open(idx_path) as f:
            content = f.readlines()

        for i in range(0, len(content), 2):
            left_id = content[i].strip()
            right_id = content[i + 1].strip()
            self.ar.append((path + '/' + left_id, path + '/' + right_id))

        # ✅ 如果要单图像训练且希望用左右视角全部数据
        if self.single_view and self.view == "both":
            expanded = []
            for left, right in self.ar:
                expanded.append(left)
                expanded.append(right)
            self.ar = expanded  # 变成 [img_path1, img_path2, ...]
        else:
            # 保留原结构
            self.ar = self.ar

        if set_type == 'train':
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform

    def train_transform(self, img):
        img = TF.center_crop(img, (370, 740))
        img = TF.resize(img, self.resize)
        if random() > 0.5:
            img = TF.hflip(img)
        return transforms.ToTensor()(img)

    def val_transform(self, img):
        img = TF.center_crop(img, (370, 740))
        img = TF.resize(img, self.resize)
        return transforms.ToTensor()(img)

    def __getitem__(self, index):
        # ✅ 如果是 both 模式，ar 中是单路径
        # if self.single_view and self.view == "both":
        #     img_path = self.ar[index]
        #     img = Image.open(img_path).convert("RGB")
        #     img = self.transform(img)
        #     cond = img
        #     img = img * 2 - 1
        #     img = img.permute(1, 2, 0)
        #     return {"main": img, "side": cond, "txt": "", "index": index}

        # ✅ 否则仍用左右对结构
        left_path, right_path = self.ar[index]
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")

        img_l = self.transform(left_img)
        img_r = self.transform(right_img)

        if self.single_view:
            if self.view == "left":
                img = img_l
            elif self.view == "right":
                img = img_r
            else:
                raise ValueError("Invalid view, choose from ['left', 'right', 'both']")
            cond = img
            img = img * 2 - 1
            img = img.permute(1, 2, 0)
            return {"main": img, "side": cond, "txt": "", "index": index}
        else:
            print("distributed image compression dataset processing ......")
            img = img_l * 2 - 1
            img = img_l.permute(1, 2, 0)           
            cond = img_r * 2 - 1
            cond = img_r.permute(1, 2, 0)             
            return {"main": img, "side": cond, "txt": "", "index": index}

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


if __name__ == "__main__":
    ds = PairKitti(path="./", set_type="train", resize=(128, 256), single_view=True, view="both")
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    for b in dl:
        print("jpg:", b["jpg"].shape, "hint:", b["hint"].shape)
        break
    print("Total samples:", len(ds))
