from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data.transforms import get_transforms


class CIFAKEDataset(Dataset):

    LABEL_MAP = {"REAL": 0, "FAKE": 1}

    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test"), f"Unknown split: {split}"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        import os
        for class_name, label in self.LABEL_MAP.items():
            class_dir = self.root / self.split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected directory not found: {class_dir}")
            # os.scandir is significantly faster than Path.iterdir() on Windows
            # and we skip sorting since order does not matter (DataLoader shuffles)
            with os.scandir(class_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((entry.path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")  # path is str from os.scandir
        if self.transform:
            image = self.transform(image)
        return image, label


def get_cifake_loaders(cfg):
    """
    Build train and test DataLoaders from a Config instance.

    Usage in notebook:
        from data.cifake import get_cifake_loaders
        train_loader, test_loader = get_cifake_loaders(cfg)
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

    train_ds = CIFAKEDataset(cfg.data.cifake_root, split="train", transform=train_tf)
    test_ds  = CIFAKEDataset(cfg.data.cifake_root, split="test",  transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    return train_loader, test_loader