"""
The standard deepdetect.py applies the same transform to all images. This means
degradation augmentations (JPEG, blur, noise) are applied before patch selection,
which destroys the frequency artifacts the frequency branch is trying to detect.

This module implements the Chen et al. (2024) approach: select the frequency patch
from the raw unaugmented image, while the spatial branch sees the full augmented image.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image

from data.transforms import get_transforms


class DeepDetectDualTransform(Dataset):
    """
    DeepDetect dataset that returns (aug_image, clean_image, label) per sample.

    aug_image:   full degradation pipeline — JPEG, blur, noise, recompression, mixed
    clean_image: spatial augmentation only (RandomResizedCrop, RandomHorizontalFlip)
                 NO degradation augmentations
    """

    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, root, split="train",
                 aug_transform=None, clean_transform=None):
        assert split in ("train", "test"), f"Unknown split: {split}"
        self.root            = Path(root)
        self.split           = split
        self.aug_transform   = aug_transform
        self.clean_transform = clean_transform
        self.samples         = []
        self._load_samples()

    def _load_samples(self):
        for class_name, label in self.LABEL_MAP.items():
            class_dir = self.root / self.split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected directory not found: {class_dir}\n"
                    f"Check that root points to: data/raw/deep_detect/data"
                )
            with os.scandir(class_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ):
                        self.samples.append((entry.path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        aug_image   = self.aug_transform(image)   if self.aug_transform   else image
        clean_image = self.clean_transform(image) if self.clean_transform else image
        return aug_image, clean_image, label


def get_deepdetect_dual_loaders(cfg):
    """
    Build train, val, test DataLoaders with dual transforms.

    Spatial branch receives fully augmented images.
    Frequency branch receives spatially augmented but spectrally clean images.

    Usage:
        from data.deepdetect_dual import get_deepdetect_dual_loaders
        train_loader, val_loader, test_loader = get_deepdetect_dual_loaders(cfg)

    DataLoader yields: (aug_image, clean_image, label)
    """
    # Full degradation pipeline for spatial branch
    aug_tf = get_transforms(
        "train",
        image_size         = cfg.data.image_size,
        jpeg_aug           = cfg.data.jpeg_aug,
        jpeg_quality_range = cfg.data.jpeg_aug_quality_range,
        blur_aug           = cfg.data.blur_aug,
        noise_aug          = cfg.data.noise_aug,
        recompression_aug  = cfg.data.recompression_aug,
        mixed_aug          = cfg.data.mixed_aug,
        mixed_aug_prob     = cfg.data.mixed_aug_prob,
    )

    # Spatial-only transform for frequency branch — no degradation
    clean_tf = get_transforms(
        "train",
        image_size         = cfg.data.image_size,
        jpeg_aug           = False,
        blur_aug           = False,
        noise_aug          = False,
        recompression_aug  = False,
        mixed_aug          = False,
    )

    # Test/val: both clean — no augmentation at all
    test_tf = get_transforms("test", image_size=cfg.data.image_size)

    full_train_ds = DeepDetectDualTransform(
        cfg.data.deepdetect_root, split="train",
        aug_transform=aug_tf, clean_transform=clean_tf,
    )
    test_ds = DeepDetectDualTransform(
        cfg.data.deepdetect_root, split="test",
        aug_transform=test_tf, clean_transform=test_tf,
    )

    # 85/15 train/val split
    n_total = len(full_train_ds)
    n_val   = int(n_total * cfg.data.val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(cfg.train.seed)
    train_ds, val_indices = random_split(
        full_train_ds, [n_train, n_val], generator=generator
    )

    # Val uses test transforms for both — clean, deterministic
    clean_val_ds = DeepDetectDualTransform(
        cfg.data.deepdetect_root, split="train",
        aug_transform=test_tf, clean_transform=test_tf,
    )
    val_ds = Subset(clean_val_ds, val_indices.indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
