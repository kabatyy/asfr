"""
A CLI entry point for quick training runs before notebook experiments.
Wraps around experiments/train.py
Usage:
    python train.py
    python train.py --backbone vit_b_16 --fusion gating
    python train.py --backbone swin_v2_s --fusion scalar --frozen
    python train.py --dataset deepdetect --image_size 224 --epochs 50

"""

import argparse
import torch

from config import Config
from data.cifake import get_cifake_loaders
from experiments.train import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnext_base",
                   choices=["convnext_base", "dino_vits8", "swin_v2_s", "vit_b_16", "vit_b_32"])
    p.add_argument("--fusion", default="gating",
                   choices=["joint_only", "scalar", "gating"])
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--dataset", default="cifake", choices=["cifake", "deepdetect"])
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.backbone.name    = args.backbone
    cfg.backbone.frozen  = args.frozen
    cfg.fusion.mode      = args.fusion
    cfg.data.dataset     = args.dataset
    cfg.data.image_size  = args.image_size
    cfg.train.epochs     = args.epochs
    cfg.train.seed       = args.seed
    cfg.notes            = args.notes
    cfg.experiment_name  = (f"{args.backbone}_{args.fusion}"
                            f"{'_frozen' if args.frozen else ''}")

    if args.dataset == "cifake":
        train_loader, test_loader = get_cifake_loaders(
            root=cfg.data.cifake_root,
            image_size=cfg.data.image_size,
            batch_size=args.batch_size,
        )
    else:
        from data.deepdetect import get_deepdetect_loaders
        train_loader, test_loader = get_deepdetect_loaders(
            root=cfg.data.deepdetect_root,
            batch_size=args.batch_size,
        )

    train(cfg, train_loader, test_loader)


if __name__ == "__main__":
    main()