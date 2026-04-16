"""
Dataset structure:
    data/raw/deep_detect/data/
        train/
            real/  *.jpg
            fake/  *.jpg
        test/
            real/  *.jpg
            fake/  *.jpg

Labels: 0 = real, 1 = fake

Download from:
    https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025
"""

import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from PIL import Image
from data.transforms import get_transforms


class DeepDetectDataset(Dataset):

    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test"), f"Unknown split: {split}"
        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.samples   = []
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
        if self.transform:
            image = self.transform(image)
        return image, label


def get_deepdetect_loaders(cfg):
    """
    Build train, val, and test DataLoaders from a Config instance.

    Splits the training set into train (85%) and val (15%).
    All augmentations read from cfg.data.

    Usage:
        from data.deepdetect import get_deepdetect_loaders
        train_loader, val_loader, test_loader = get_deepdetect_loaders(cfg)
    """
    train_tf = get_transforms(
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
    test_tf = get_transforms("test", image_size=cfg.data.image_size)

    full_train_ds = DeepDetectDataset(
        cfg.data.deepdetect_root, split="train", transform=train_tf
    )
    test_ds = DeepDetectDataset(
        cfg.data.deepdetect_root, split="test", transform=test_tf
    )

    # 85/15 train/val split — seeded for reproducibility
    n_total = len(full_train_ds)
    n_val   = int(n_total * cfg.data.val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(cfg.train.seed)
    train_ds, val_ds = random_split(
        full_train_ds, [n_train, n_val], generator=generator
    )

    # Val set uses clean test transforms — no augmentation
    val_ds.dataset = DeepDetectDataset(
        cfg.data.deepdetect_root, split="train", transform=test_tf
    )

    pin_memory = torch.cuda.is_available() # pin memory can be false on MPS

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=pin_memory,
    )

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader