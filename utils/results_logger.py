"""
Automatic results logging to CSV.

Usage:
    from utils.results_logger import save_results
    save_results(cfg, metrics)

Each run adds one row. If the file doesn't exist it is created with a header.

Columns logged:
    - Run identity:  timestamp, experiment_name, fusion, backbone, frozen, dataset
    - Key hparams:   freq_aux_weight, spatial_aux_weight, diversity_weight, epochs
    - Core metrics:  accuracy, auc_roc, f1
    - Gate metrics:  gate_entropy, gate_mean, gate_var  (0.0 if not gating mode)
    - Notes:         free-text from cfg.notes (optional)
"""

import csv
import os
from datetime import datetime
from config import Config


COLUMNS = [
    "timestamp",
    "experiment_name",
    "fusion",
    "backbone",
    "frozen",
    "dataset",
    "image_size",
    "epochs",
    "freq_aux_weight",
    "spatial_aux_weight",
    "diversity_weight",
    "accuracy",
    "auc_roc",
    "f1",
    "gate_entropy",
    "gate_mean",
    "gate_var",
    "notes",
]


def save_results(cfg: Config, metrics: dict, notes: str = "") -> None:
    """
    Append one row to results/results.csv.

    Args:
        cfg:     Config instance for the completed run.
        metrics: Dict from evaluate() or full_evaluation().
                 Gate stats can be flat keys (gate_entropy, gate_mean, gate_var)
                 or nested under 'gate_stats' — both formats are handled.
        notes:   Optional note. Falls back to cfg.notes if not provided.
    """
    results_path = os.path.join(cfg.train.results_dir, "results.csv")
    os.makedirs(cfg.train.results_dir, exist_ok=True)
    file_exists = os.path.exists(results_path)

    # Gate stats may be flat or nested under 'gate_stats'
    gate_stats = metrics.get("gate_stats", {})

    def _gate(flat_key, nested_key):
        # flat_key:   e.g. "gate_entropy" (from train.py evaluate())
        # nested_key: e.g. "entropy"      (from gate_distribution_stats / evaluate.py)
        return metrics.get(flat_key, gate_stats.get(nested_key, 0.0))

    row = {
        "timestamp":          datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_name":    cfg.experiment_name,
        "fusion":             cfg.fusion.mode,
        "backbone":           cfg.backbone.name,
        "frozen":             cfg.backbone.frozen,
        "dataset":            cfg.data.dataset,
        "image_size":         cfg.data.image_size,
        "epochs":             cfg.train.epochs,
        "freq_aux_weight":    cfg.loss.freq_aux_weight,
        "spatial_aux_weight": cfg.loss.spatial_aux_weight,
        "diversity_weight":   cfg.fusion.diversity_weight,
        "accuracy":           round(metrics.get("accuracy", 0.0), 4),
        "auc_roc":            round(metrics.get("auc_roc",   0.0), 4),
        "f1":                 round(metrics.get("f1",        0.0), 4),
        "gate_entropy":       round(_gate("gate_entropy", "entropy"),  4),
        "gate_mean":          round(_gate("gate_mean",    "mean"),      4),
        "gate_var":           round(_gate("gate_var",     "variance"),  4),
        "notes":              notes or getattr(cfg, "notes", ""),
    }

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results saved → {results_path}  "
          f"({row['experiment_name']}, acc={row['accuracy']})")