"""
Thresholds:
    >= 70%  — healthy, proceed to fusion experiments
    60-70%  — weak but not broken, investigate before proceeding
    < 60%   — hard stop, fix the FFT representation first

WHAT THIS DOES
--------------
Trains only the FrequencyBranch as a standalone binary classifier using
its aux_head directly. No spatial branch, no fusion. Reports every epoch
with train_loss and val_acc exactly like the full experiment loop.

HOW TO USE
----------
    from experiments.baseline_freq_only import run_freq_only_baseline
    freq_acc = run_freq_only_baseline(cfg, train_loader, val_loader, test_loader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
from config import Config
from models.frequency_branch import FrequencyBranch
from utils.diagnostics import check_warning_signs
from utils.metrics import binary_accuracy, binary_auc_roc, binary_f1
from utils.results_logger import save_results


def run_freq_only_baseline(cfg: Config, train_loader, val_loader,
                            test_loader=None) -> float:
    """
    Train and evaluate the frequency branch as a standalone binary classifier.
    Reports every epoch with train_loss and val_acc.
    Final evaluation runs on test_loader if provided, otherwise val_loader.

    Args:
        cfg:          Config instance — epochs read from cfg.train.epochs
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader — evaluated every epoch
        test_loader:  Optional. Final evaluation set.

    Returns:
        Final accuracy (float).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    epochs = cfg.train.epochs
    eval_loader = test_loader if test_loader is not None else val_loader

    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model     = FrequencyBranch(cfg.frequency, feature_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=device.type == "cuda")

    print(f"\n{'='*70}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Frequency-only baseline | Epochs: {epochs}")
    print(f"Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}")
    print(f"{'='*70}\n")

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>3}/{epochs}",
                    leave=False, unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                _, aux_logits, _ = model(images)
                loss = criterion(aux_logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
        scheduler.step()

        # Val evaluation every epoch
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                _, aux_logits, _ = model(images.to(device))
                val_logits.append(aux_logits.cpu())
                val_labels.append(labels)
        vl = torch.cat(val_logits)
        yl = torch.cat(val_labels)
        val_acc = binary_accuracy(vl, yl)

        print(f"Epoch {epoch+1:>3}/{epochs} | "
              f"train_loss={total_loss/len(train_loader):.4f} | "
              f"val_acc={val_acc:.1%}")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f"{cfg.train.checkpoint_dir}/best_{cfg.experiment_name}.pt")
            print(f"  -> Saved best val_acc={best_val_acc:.1%}")

    # Final evaluation
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in eval_loader:
            _, aux_logits, _ = model(images.to(device))
            all_logits.append(aux_logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    acc = binary_accuracy(logits, labels)
    auc = binary_auc_roc(logits, labels)
    f1  = binary_f1(logits, labels)

    print(f"\nFrequency-only final results:")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  AUC-ROC:  {auc:.3f}")
    print(f"  F1:       {f1:.3f}")

    warnings = check_warning_signs(freq_only_acc=acc)
    for w in warnings:
        print(f"\n{w}")

    if acc >= 0.70:
        print("\nResult: PASS — frequency branch is capturing real signal (>= 70%).")
    elif acc >= 0.60:
        print("\nResult: WEAK — frequency branch is below the 70% target (60-70%).")
    else:
        print("\nResult: FAIL — frequency branch is below 60%.")
    metrics = {"accuracy": acc, "auc_roc": auc, "f1": f1}
    save_results(cfg, metrics, notes="freq-only baseline, no fusion, no spatial branch")
    return acc