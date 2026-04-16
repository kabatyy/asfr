"""
This is the main evaluation function called from notebooks after training.
It runs all required metrics and prints a structured report.

HOW TO USE
----------
    from experiments.evaluate import full_evaluation

    results = full_evaluation(
        cfg,
        checkpoint_path="checkpoints/best_swin_v2_s_gating.pt",
        test_loader=test_loader,
    )
    # results dict is also passed to save_results() automatically

WHAT IT REPORTS
---------------
    - Joint accuracy, AUC-ROC, F1
    - Spatial-only accuracy  (from spatial aux head)
    - Frequency-only accuracy (from freq aux head)
    - Delta = joint - spatial_only  (how much freq branch added)
    - Gate distribution: mean, variance, entropy (gating mode only)
    - Warning sign checks
    - Per-generator accuracy (DeepDetect only)
    - Per-JPEG-quality accuracy (DeepDetect only)
"""

import torch
from config import Config
from models.full_model import ASFRModel
from utils.diagnostics import check_warning_signs
from utils.metrics import (
    binary_accuracy, binary_auc_roc, binary_f1,
    gate_distribution_stats,
    per_generator_accuracy, per_jpeg_quality_accuracy,
)
from utils.results_logger import save_results


def full_evaluation(cfg: Config, checkpoint_path: str, test_loader,
                    dataset_type: str = "cifake",
                    save_to_csv: bool = True) -> dict:
    """
    Load a checkpoint and run evaluation.

    Args:
        cfg:              Config used for training this checkpoint.
        checkpoint_path:  Path to saved model weights (.pt file).
        test_loader:      Test DataLoader.
        dataset_type:     "cifake" or "deepdetect".
                          "deepdetect" enables per-generator and per-JPEG metrics.
        save_to_csv:      If True, appends results to results/results.csv.

    Returns:
        Dict of all computed metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASFRModel(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    (all_joint_logits, all_spatial_logits, all_freq_logits,
     all_labels, all_gate_values, all_gen_ids, all_jpeg_q) = (
        [], [], [], [], [], [], []
    )

    with torch.no_grad():
        for batch in test_loader:
            images  = batch[0].to(device)
            labels  = batch[1]

            # Inference pass (joint logits only)
            out_inf = model(images, training=False)
            all_joint_logits.append(out_inf["joint_logits"].cpu())
            all_labels.append(labels)

            # Training pass to get auxiliary logits and gate values
            out_train = model(images, training=True)
            if out_train.get("spatial_aux_logits") is not None:
                all_spatial_logits.append(out_train["spatial_aux_logits"].cpu())
            if out_train.get("freq_aux_logits") is not None:
                all_freq_logits.append(out_train["freq_aux_logits"].cpu())
            if out_train.get("gate_values") is not None:
                all_gate_values.append(out_train["gate_values"].cpu())

            # DeepDetect metadata
            if dataset_type == "deepdetect" and len(batch) >= 4:
                all_gen_ids.append(batch[2])
                all_jpeg_q.append(batch[3])

    # Aggregate
    joint_logits = torch.cat(all_joint_logits)
    labels       = torch.cat(all_labels)

    results = {
        "accuracy": binary_accuracy(joint_logits, labels),
        "auc_roc":  binary_auc_roc(joint_logits, labels),
        "f1":       binary_f1(joint_logits, labels),
    }

    if all_spatial_logits:
        results["spatial_only_accuracy"] = binary_accuracy(
            torch.cat(all_spatial_logits), labels)

    if all_freq_logits:
        results["freq_only_accuracy"] = binary_accuracy(
            torch.cat(all_freq_logits), labels)

    if "spatial_only_accuracy" in results:
        results["delta"] = results["accuracy"] - results["spatial_only_accuracy"]

    # Gate stats (gating mode only)
    gate_entropy = None
    if all_gate_values:
        gate_tensor = torch.cat(all_gate_values)
        results["gate_stats"] = gate_distribution_stats(gate_tensor)
        gate_entropy = results["gate_stats"]["entropy"]

    # DeepDetect-only metrics
    if dataset_type == "deepdetect" and all_gen_ids:
        from data.deepdetect import GENERATOR_NAMES
        results["per_generator_accuracy"] = per_generator_accuracy(
            joint_logits, labels,
            torch.cat(all_gen_ids), GENERATOR_NAMES
        )
    if dataset_type == "deepdetect" and all_jpeg_q:
        results["per_jpeg_accuracy"] = per_jpeg_quality_accuracy(
            joint_logits, labels, torch.cat(all_jpeg_q)
        )

    # Warning signs
    results["warnings"] = check_warning_signs(
        freq_only_acc=results.get("freq_only_accuracy"),
        fused_acc=results.get("accuracy"),
        spatial_only_acc=results.get("spatial_only_accuracy"),
        gate_entropy=gate_entropy,
    )

    _print_report(results, cfg)

    if save_to_csv:
        save_results(cfg, results)

    return results


def _print_report(results, cfg):
    print("\n" + "="*60)
    print(f"EVALUATION — {cfg.experiment_name}")
    print(f"Backbone: {cfg.backbone.name} | Fusion: {cfg.fusion.mode} | "
          f"Frozen: {cfg.backbone.frozen}")
    print("="*60)
    print(f"  Joint accuracy:     {results['accuracy']:.1%}")
    print(f"  AUC-ROC:            {results['auc_roc']:.3f}")
    print(f"  F1:                 {results['f1']:.3f}")

    if "spatial_only_accuracy" in results:
        print(f"  Spatial-only:       {results['spatial_only_accuracy']:.1%}")
    if "freq_only_accuracy" in results:
        print(f"  Freq-only:          {results['freq_only_accuracy']:.1%}")
    if "delta" in results:
        print(f"  Delta (Δ):          {results['delta']:+.1%}  "
              f"(freq branch contribution)")

    if "gate_stats" in results:
        g = results["gate_stats"]
        print(f"\n  Gate distribution:")
        print(f"    entropy:  {g['entropy']:.3f} nats  "
              f"({'OK' if g['entropy'] >= 0.3 else 'COLLAPSED < 0.3'})")
        print(f"    mean:     {g['mean']:.3f}")
        print(f"    variance: {g['variance']:.4f}")

    if "per_generator_accuracy" in results:
        print(f"\n  Per-generator accuracy:")
        for gen, acc in results["per_generator_accuracy"].items():
            print(f"    {gen:<25} {acc:.1%}")

    if "per_jpeg_accuracy" in results:
        print(f"\n  Per-JPEG-quality accuracy:")
        for bucket, acc in results["per_jpeg_accuracy"].items():
            print(f"    quality {bucket}:   {acc:.1%}")

    if results.get("warnings"):
        for w in results["warnings"]:
            print(f"{w}")
    else:
        print("\n  No warning signs triggered.")
    print("="*60)