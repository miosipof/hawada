"""Vision datasets and transforms for demos/recipes.

These utilities are lightweight wrappers around `torchvision` to avoid
introducing data logic into the core library. Nothing here is required for
users who bring their own dataloaders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_train_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_val_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

@dataclass
class VisionDataConfig:
    train_root: str
    val_root: Optional[str] = None
    img_size: int = 224
    batch_size: int = 32
    val_batch_size: Optional[int] = None
    workers: int = 8
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None

class SafeImageFolder(datasets.ImageFolder):
    """SafeImageFolder that ignores hidden classes like `.ipynb_checkpoints`."""
    def find_classes(self, directory: str):
        classes, _ = super().find_classes(directory)
        # drop any class directory that starts with a dot
        classes = [c for c in classes if not c.startswith('.')]
        classes = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_imagenet_like_loaders(cfg):
    # allow dict or object with attributes; provide defaults
    def get(k, default=None):
        return cfg.get(k, default) if isinstance(cfg, dict) else getattr(cfg, k, default)

    train_root = get("train_root")
    val_root   = get("val_root")
    if not train_root or not val_root:
        raise ValueError("data.train_root and data.val_root must be provided")

    img_size    = int(get("img_size", 224))
    batch_size  = int(get("batch_size", 64))
    num_workers = int(get("num_workers", 8))
    limit_train = get("limit_train", None)
    limit_val   = get("limit_val", None)

    train_tf = make_train_transform(img_size)
    val_tf   = make_val_transform(img_size)

    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset

    train_ds = SafeImageFolder(train_root, transform=train_tf)
    val_ds   = SafeImageFolder(val_root,   transform=val_tf)

    if limit_train is not None:
        n = min(int(limit_train), len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
    if limit_val is not None:
        n = min(int(limit_val), len(val_ds))
        val_ds = Subset(val_ds, list(range(n)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1, batch_size // 2), shuffle=False,
                              num_workers=max(0, num_workers // 2), pin_memory=True, drop_last=False)
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Synthetic dataset (for CI / smoke tests)
# -----------------------------------------------------------------------------

class RandomImageDataset(Dataset):
    def __init__(self, n: int = 1024, size: int = 224, num_classes: int = 1000):
        self.n = n
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(3, self.size, self.size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def build_random_loaders(n_train: int = 1024, n_val: int = 128, size: int = 224, batch_size: int = 32):
    train_ds = RandomImageDataset(n_train, size)
    val_ds = RandomImageDataset(n_val, size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size // 2), shuffle=False)
    return train_loader, val_loader
