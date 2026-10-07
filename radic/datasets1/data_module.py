import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import OmegaConf

from datasets1.kitti_dataset import PairKitti
from datasets1.cityscapes_dataset import PairCityscape


class DataModule(pl.LightningDataModule):
    """
    通用数据加载模块：
    支持 KITTI / Cityscape 双目或单视角训练。
    """

    def __init__(self, train_config, val_config):
        super().__init__()
        self.train_config = OmegaConf.load(train_config)
        self.val_config = OmegaConf.load(val_config)

        # 数据集通用参数
        self.dataset_name = self.train_config.get("dataset_name", "KITTI")
        self.dataset_path = self.train_config.get("dataset_path", "./datasets/KITTI")
        self.batch_size = self.train_config.get("batch_size", 8)
        self.num_workers = self.train_config.get("num_workers", 8)
        self.resize = tuple(self.train_config.get("resize", [128, 256]))
        self.single_view = self.train_config.get("single_view", False)
        self.view = self.train_config.get("view", "left")

    def setup(self, stage=None):
        # 根据配置创建不同数据集
        if self.dataset_name.lower().startswith("kitti"):
            self.train_dataset = PairKitti(
                path=self.dataset_path,
                set_type="train",
                stereo=True,
                resize=self.resize,
                single_view=self.single_view,
                view=self.view,
            )
            self.val_dataset = PairKitti(
                path=self.dataset_path,
                set_type="val",
                stereo=True,
                resize=self.resize,
                single_view=self.single_view,
                view=self.view,
            )
        elif self.dataset_name.lower().startswith("city"):
            self.train_dataset = PairCityscape(
                path=self.dataset_path,
                set_type="train",
                resize=self.resize,
                single_view=self.single_view,
                view=self.view,
            )
            self.val_dataset = PairCityscape(
                path=self.dataset_path,
                set_type="val",
                resize=self.resize,
                single_view=self.single_view,
                view=self.view,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
