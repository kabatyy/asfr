"""
Uses synthetic random data to test and validate the full pipeline. 

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from experiments.train import train
from pathlib import Path
from experiments.evaluate import full_evaluation

def make_fake_loader(n=128, image_size=32, batch_size=16):
    """Synthetic dataset — random images with random binary labels."""
    images = torch.rand(n, 3, image_size, image_size)
    labels = torch.randint(0, 2, (n,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def main():
    print("=" * 50)
    print("ASFR Pipeline Smoke Test")
    print("=" * 50)

    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Minimal config — smallest backbone, no pretrained weights
    cfg = Config()
    cfg.backbone.name      = "swin_v2_s"
    cfg.backbone.pretrained = False  
    cfg.fusion.mode        = "gating"
    cfg.train.epochs       = 2      
    cfg.data.image_size    = 32
    cfg.experiment_name    = "smoke_test"
    cfg.notes              = "synthetic data smoke test"

    train_loader = make_fake_loader(n=64,  image_size=32, batch_size=16)
    test_loader  = make_fake_loader(n=32,  image_size=32, batch_size=16)

    print(f"\nRunning {cfg.train.epochs} epochs on synthetic data...")
    try:
        train(cfg, train_loader, test_loader)

        # Test full_evaluation() with the saved checkpoint
        print("\nTesting full evaluation pipeline...")
        checkpoint = f"{cfg.train.checkpoint_dir}/best_{cfg.experiment_name}.pt"
        if Path(checkpoint).exists():
            results = full_evaluation(cfg, checkpoint, test_loader, save_to_csv=True)
            assert "accuracy"  in results
            assert "auc_roc"   in results
            assert "f1"        in results
            assert "gate_stats" in results  # gating mode
            assert "warnings"  in results
            print("full_evaluation(): OK")
        else:
            print(f"No checkpoint found at {checkpoint} — skipping eval test")
    except Exception as e:
        print(f"\nSmoke test FAILED: {e}")
        raise


if __name__ == "__main__":
    main()