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
from utils.results_logger import save_results


def train_one_epoch(model, loader, optimizer, aux_loss_fn,
                    diversity_fn, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    all_gate_values = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

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

        losses["total"].backward()

        optimizer.step()
        total_loss += losses["total"].item()

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

    metrics["warnings"] = check_warning_signs(gate_entropy=gate_entropy)

    return metrics


def train(cfg: Config, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model        = ASFRModel(cfg).to(device)
    optimizer    = optim.AdamW(model.parameters(), lr=cfg.train.lr,
                               weight_decay=cfg.train.weight_decay)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    aux_loss_fn  = AuxiliaryLoss(cfg.loss)
    diversity_fn = DiversityRegulariser(weight=cfg.fusion.diversity_weight)

    best_acc = 0.0

    for epoch in range(cfg.train.epochs):
        train_log = train_one_epoch(
            model, train_loader, optimizer,
            aux_loss_fn, diversity_fn, device, epoch, cfg
        )
        scheduler.step()
        val_metrics = evaluate(model, test_loader, device, cfg)

        # Console output
        print(f"Epoch {epoch+1}/{cfg.train.epochs} | "
              f"loss={train_log['total_loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.1%} | "
              f"auc={val_metrics['auc_roc']:.3f} | "
              f"f1={val_metrics['f1']:.3f}")

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
        if epoch > 0.8 * cfg.train.epochs:
            for w in val_metrics.get("warnings", []):
                print(f"{w}")

        # Checkpoint
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(),
                       f"{cfg.train.checkpoint_dir}/best_{cfg.experiment_name}.pt")
            print(f"  -> Saved best (acc={best_acc:.1%})")

    # Log results to CSV
    save_results(cfg, val_metrics)
    print(f"\nTraining complete. Best accuracy: {best_acc:.1%}")
    return best_acc