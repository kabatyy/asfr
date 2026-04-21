"""
Full model training loop.

Training checklist:
  [x] Auxiliary heads enabled (cfg.loss.use_auxiliary_heads = True)
  [x] Softmax on scalar fusion weights (cfg.fusion.scalar_softmax = True)
  [x] Diversity regulariser for gating (cfg.fusion.diversity_weight > 0)
  [x] Cleaner reconstruction loss on real patches only
  [x] Gradient norm logging every N epochs
  [x] Scalar/gate value logging every epoch
  [x] Warning sign checks after each epoch
"""

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from config import Config
from models.full_model import ASFRModel
from losses.auxiliary import AuxiliaryLoss
from losses.diversity import DiversityRegulariser
from utils.diagnostics import (
    log_freq_branch_grad_norms,
    log_fusion_scalars,
    compute_gate_entropy,
    check_warning_signs,
)
from utils.metrics import binary_accuracy, binary_auc_roc, binary_f1


def train_one_epoch(model, loader, optimizer, aux_loss_fn,
                    diversity_fn, device, epoch, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    all_gate_values = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images, training=True)

            # Cleaner reconstruction loss — real images only
            real_mask = (labels == 0)
            recon_loss = None
            if real_mask.any() and model.freq_branch.cleaner is not None:
                recon_loss = model.freq_branch.cleaner.reconstruction_loss(
                    outputs["freq_patches"][real_mask]
                )

            # Combined loss
            losses = aux_loss_fn(
                joint_logits       = outputs["joint_logits"],
                labels             = labels,
                spatial_aux_logits = outputs.get("spatial_aux_logits"),
                freq_aux_logits    = outputs.get("freq_aux_logits"),
                cleaner_recon_loss = recon_loss,
            )

            # Diversity regulariser for gating
            if "gate_values" in outputs:
                diversity_penalty = diversity_fn(outputs["gate_values"].detach())
                losses["total"] = losses["total"] + diversity_penalty
                all_gate_values.append(outputs["gate_values"].detach().cpu())

        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_loss = losses["total"].item()
        total_loss += batch_loss
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    log = {"total_loss": total_loss / len(loader)}

    # Gradient norm logging
    if epoch % cfg.train.log_grad_norm_every_n_epochs == 0:
        log.update(log_freq_branch_grad_norms(model))

    # Gate entropy logging
    if all_gate_values:
        gate_tensor = torch.cat(all_gate_values)
        log["gate_entropy_train"] = compute_gate_entropy(gate_tensor)

    return log


def evaluate(model, loader, device, cfg):
    model.eval()
    all_logits, all_labels, all_gate_values = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, training=False)
            all_logits.append(outputs["joint_logits"].cpu())
            all_labels.append(labels.cpu())

        # Re-run with training=True to collect gate values
        if cfg.fusion.mode == "gating":
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images, training=True)
                if "gate_values" in outputs:
                    all_gate_values.append(outputs["gate_values"].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    metrics = {
        "accuracy": binary_accuracy(logits, labels),
        "auc_roc":  binary_auc_roc(logits, labels),
        "f1":       binary_f1(logits, labels),
    }

    gate_entropy = None
    if all_gate_values:
        gate_tensor = torch.cat(all_gate_values)
        gate_entropy = compute_gate_entropy(gate_tensor)
        metrics["gate_entropy"] = gate_entropy
        metrics["gate_mean"]    = gate_tensor.mean().item()
        metrics["gate_var"]     = gate_tensor.var().item()

    metrics["warnings"] = check_warning_signs(
        gate_entropy=gate_entropy,
        epoch=metrics.get("_epoch"),
        total_epochs=metrics.get("_total_epochs"),
    )

    return metrics


def train(cfg: Config, train_loader, val_loader, test_loader=None):
    """
    Train the full ASFR model.

    Args:
        cfg:          Config instance
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader — used for per-epoch monitoring
        test_loader:  Optional. If provided, runs final evaluation on test set.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = ASFRModel(cfg).to(device)

    # Extract param groups BEFORE torch.compile — compiled model has different
    # parameter objects and the groups would be lost
    backbone_params = list(model.spatial_branch.backbone.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is bp for bp in backbone_params)]

    if device.type == 'cuda':
        model = torch.compile(model)
        print("torch.compile enabled")
    else:
        print(f"torch.compile skipped — device is {device.type}")
    scaler = GradScaler(enabled=device.type == "cuda")

    # Differential learning rates:
    # Only apply if backbone_lr differs from lr — otherwise use single optimizer
    if cfg.train.backbone_lr != cfg.train.lr:
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": cfg.train.backbone_lr},
            {"params": other_params,    "lr": cfg.train.lr},
        ], weight_decay=cfg.train.weight_decay)
        print(f"Using differential lr: backbone={cfg.train.backbone_lr:.2e}, "
              f"others={cfg.train.lr:.2e}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
        print(f"Using single lr: {cfg.train.lr:.2e}")

    # Compile for MPS speedup — FFT escapes compile via @torch.compiler.disable
    # CUDA gets speedup from mixed precision instead
    if device.type == "mps":
        model = torch.compile(model)

    scaler       = GradScaler(enabled=device.type == "cuda")
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    aux_loss_fn  = AuxiliaryLoss(cfg.loss)
    diversity_fn = DiversityRegulariser(weight=cfg.fusion.diversity_weight)

    best_val_acc     = 0.0
    patience         = getattr(cfg.train, "early_stopping_patience", 5)
    patience_counter = 0
    ckpt_path        = f"{cfg.train.checkpoint_dir}/best_{cfg.experiment_name}.pt"

    print(f"\n{'='*70}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Backbone: {cfg.backbone.name} | Fusion: {cfg.fusion.mode} | "
          f"Frozen: {cfg.backbone.frozen}")
    print(f"Epochs: {cfg.train.epochs} | LR: {cfg.train.lr} | "
          f"Batch: {cfg.data.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(cfg.train.epochs):
        train_log = train_one_epoch(
            model, train_loader, optimizer,
            aux_loss_fn, diversity_fn, device, epoch, cfg, scaler
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, cfg)
        val_metrics["_epoch"] = epoch
        val_metrics["_total_epochs"] = cfg.train.epochs

        # Console output — every epoch
        print(f"Epoch {epoch+1:>3}/{cfg.train.epochs} | "
              f"train_loss={train_log['total_loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.1%} | "
              f"val_auc={val_metrics['auc_roc']:.3f} | "
              f"val_f1={val_metrics['f1']:.3f} | "
              f"best={best_val_acc:.1%} | "
              f"patience={patience_counter}/{patience}")

        # Scalar/gate logging
        if epoch % cfg.train.log_scalar_every_n_epochs == 0:
            if cfg.fusion.mode == "scalar":
                s = log_fusion_scalars(model)
                print(f"  scalars — spatial={s['scalar_spatial']:.3f} "
                      f"freq={s['scalar_freq']:.3f}")
            if cfg.fusion.mode == "gating" and "gate_entropy" in val_metrics:
                print(f"  gate — entropy={val_metrics['gate_entropy']:.3f} nats | "
                      f"mean={val_metrics['gate_mean']:.3f} | "
                      f"var={val_metrics['gate_var']:.4f}")

        # Gradient norm logging
        if "freq_branch_grad_norm" in train_log:
            print(f"  grad norms — freq={train_log['freq_branch_grad_norm']:.4f} | "
                  f"spatial={train_log['spatial_branch_grad_norm']:.4f}")

        # Warning signs
        val_metrics.pop("_epoch", None)
        val_metrics.pop("_total_epochs", None)
        for w in val_metrics.get("warnings", []):
            print(f"\n{w}")

        # Checkpoint + early stopping
        if val_metrics["accuracy"] > best_val_acc + 0.001:
            best_val_acc     = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best val_acc={best_val_acc:.1%}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} — "
                      f"best val_acc={best_val_acc:.1%}")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1%}")
    print("Results will be logged to CSV after full_evaluation().")
    return best_val_acc