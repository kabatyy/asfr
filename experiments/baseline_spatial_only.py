"""
Trains only the SpatialBranch as a standalone binary classifier.
No frequency branch, no fusion.

WHY THIS EXISTS
---------------
full_evaluation() reports spatial_only_accuracy from the spatial aux head
of the full model. But that aux head trains jointly alongside the frequency
branch. Its gradients are influenced by the shared loss. A truly isolated
spatial-only model gives a cleaner floor.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import Config
from models.spatial_branch import SpatialBranch
from utils.metrics import binary_accuracy, binary_auc_roc, binary_f1
from utils.results_logger import save_results


def run_spatial_only_baseline(cfg: Config, train_loader, test_loader) -> float:
    """
    Train and evaluate the spatial branch as a standalone binary classifier.
    Number of epochs read from cfg.train.epochs.

    Args:
        cfg:          Config instance — set cfg.backbone.name before calling
        train_loader: Training DataLoader
        test_loader:  Test DataLoader

    Returns:
        Test accuracy (float).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = cfg.train.epochs
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # SpatialBranch outputs 512-dim features — add a classification head on top
    backbone = SpatialBranch(cfg.backbone, feature_dim=512).to(device)
    classifier = nn.Linear(512, 2).to(device)

    params = list(backbone.parameters()) + list(classifier.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"Training spatial-only baseline ({cfg.backbone.name}) for {epochs} epochs...")

    for epoch in range(epochs):
        backbone.train()
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features = backbone(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")

    # Save checkpoint
    ckpt_path = f"{cfg.train.checkpoint_dir}/best_{cfg.experiment_name}.pt"
    torch.save({"backbone": backbone.state_dict(),
                "classifier": classifier.state_dict()}, ckpt_path)

    # Evaluation
    backbone.eval()
    classifier.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            features = backbone(images.to(device))
            logits = classifier(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    acc = binary_accuracy(logits, labels)
    auc = binary_auc_roc(logits, labels)
    f1  = binary_f1(logits, labels)

    print(f"\nSpatial-only results ({cfg.backbone.name}):")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  AUC-ROC:  {auc:.3f}")
    print(f"  F1:       {f1:.3f}")

    metrics = {"accuracy": acc, "auc_roc": auc, "f1": f1}
    save_results(cfg, metrics,
                 notes=f"spatial-only baseline, no freq branch, {cfg.backbone.name}")
    print(f"Results saved to {cfg.train.results_dir}/results.csv")

    return acc