"""
CLI entry point for quick evaluation of a saved checkpoint.

Usage:
    python evaluate.py --checkpoint checkpoints/best_swin_v2_s_gating.pt
    python evaluate.py --checkpoint checkpoints/best_vit_b_16_gating.pt \
                       --backbone vit_b_16 --fusion gating
    python evaluate.py --checkpoint checkpoints/best_xyz.pt \
                       --dataset deepdetect --image_size 224

A thin wrapper around experiments/evaluate.py.
"""

import argparse
from config import Config
from experiments.evaluate import full_evaluation

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="convnext_base",
                   choices=["convnext_base", "dino_vits8", "swin_v2_s", "vit_b_16", "vit_b_32"])
    p.add_argument("--fusion", default="gating",
                   choices=["joint_only", "scalar", "gating"])
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--dataset", default="cifake", choices=["cifake", "deepdetect"])
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.backbone.name   = args.backbone
    cfg.backbone.frozen = args.frozen
    cfg.fusion.mode     = args.fusion
    cfg.data.dataset    = args.dataset
    cfg.data.image_size = args.image_size
    cfg.experiment_name = (f"{args.backbone}_{args.fusion}"
                           f"{'_frozen' if args.frozen else ''}")

    cfg.data.batch_size = args.batch_size
    if args.dataset == "cifake":
        from data.cifake import get_cifake_loaders
        _, test_loader = get_cifake_loaders(cfg)
    else:
        from data.deepdetect import get_deepdetect_loaders
        _, test_loader = get_deepdetect_loaders(cfg)

    full_evaluation(cfg, args.checkpoint, test_loader, dataset_type=args.dataset)


if __name__ == "__main__":
    main()