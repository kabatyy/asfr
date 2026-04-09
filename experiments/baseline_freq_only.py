"""
Thresholds:
    >= 70%  — healthy, proceed to fusion experiments
    60-70%  — weak but not broken, investigate before proceeding
    < 60%   — hard stop, fix the FFT representation first

WHAT THIS DOES
--------------
Trains only the FrequencyBranch as a standalone binary classifier using
its aux_head directly. No spatial branch, no fusion. This isolates whether
the frequency signal alone is useful.

The branch is saved as a checkpoint after training. Full evaluation is then
run.

HOW TO USE (in your notebook)
------------------------------
    from experiments.baseline_freq_only import run_freq_only_baseline
    freq_acc = run_freq_only_baseline(cfg, train_loader, test_loader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from config import Config
from models.frequency_branch import FrequencyBranch
from utils.diagnostics import check_warning_signs
from utils.metrics import binary_accuracy, binary_auc_roc, binary_f1
from utils.results_logger import save_results


def run_freq_only_baseline(cfg: Config, train_loader, test_loader) -> float:
    """
    Train and evaluate the frequency branch as a standalone binary classifier.
    Number of epochs is read from cfg.train.epochs.

    Args:
        cfg:          Config instance — set cfg.train.epochs before calling
        train_loader: Training DataLoader
        test_loader:  Test DataLoader

    Returns:
        Test accuracy (float).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = cfg.train.epochs
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = FrequencyBranch(cfg.frequency, feature_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training frequency-only baseline for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, aux_logits, _ = model(images)
            loss = criterion(aux_logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")

    # Save checkpoint
    ckpt_path = f"{cfg.train.checkpoint_dir}/best_{cfg.experiment_name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Full evaluation
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            _, aux_logits, _ = model(images.to(device))
            all_logits.append(aux_logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    acc = binary_accuracy(logits, labels)
    auc = binary_auc_roc(logits, labels)
    f1  = binary_f1(logits, labels)

    print(f"\nFrequency-only results:")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  AUC-ROC:  {auc:.3f}")
    print(f"  F1:       {f1:.3f}")

    # Warning sign check
    warnings = check_warning_signs(freq_only_acc=acc)
    for w in warnings:
        print(f"\n{w}")

    if acc >= 0.70:
        print("\nResult: PASS — frequency branch is capturing real signal (>= 70%).")
        print("Safe to proceed to fusion experiments.")
    elif acc >= 0.60:
        print("\nResult: WEAK — frequency branch is below the 70% target (60-70%).")
        print("Consider investigating before fusion. You are not blocked.")
    else:
        print("\nResult: FAIL — frequency branch is below 60%.")
        print("Do not proceed to fusion. Fix the FFT representation first.")

    # Log to results.csv — same format as all other experiments
    metrics = {
        "accuracy": acc,
        "auc_roc":  auc,
        "f1":       f1,
    }
    save_results(cfg, metrics, notes="freq-only baseline, no fusion, no spatial branch")
    print(f"Results saved to {cfg.train.results_dir}/results.csv")

    return acc