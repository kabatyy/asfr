
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils.diagnostics import compute_gate_entropy


def binary_accuracy(logits, labels):
    preds = logits.argmax(dim=1).cpu().numpy()
    return accuracy_score(labels.cpu().numpy(), preds)


def binary_auc_roc(logits, labels):
    scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return roc_auc_score(labels.cpu().numpy(), scores)


def binary_f1(logits, labels):
    preds = logits.argmax(dim=1).cpu().numpy()
    return f1_score(labels.cpu().numpy(), preds, zero_division=0)


def gate_distribution_stats(gate_values):
    """
    Summary statistics of gate outputs over the test set.
    Report these alongside accuracy for ALL gating experiments.
    Variance < 0.02 or entropy < 0.3 nats = gate collapse.
    """
    g = gate_values.cpu()
    return {
        "mean":     g.mean().item(),
        "variance": g.var().item(),
        "entropy":  compute_gate_entropy(g),
        "min":      g.min().item(),
        "max":      g.max().item(),
    }


def per_generator_accuracy(logits, labels, generator_ids, generator_names):
    """
    Accuracy broken down by generator type.
    Used for cross-generator generalisation experiment.
    """
    results = {}
    for gen_id, gen_name in generator_names.items():
        mask = (generator_ids == gen_id)
        if mask.sum() == 0:
            continue
        results[gen_name] = accuracy_score(
            labels[mask].cpu().numpy(),
            logits[mask].argmax(dim=1).cpu().numpy()
        )
    return results


def per_jpeg_quality_accuracy(logits, labels, jpeg_qualities):
    """
    Accuracy broken down by JPEG compression quality bucket.
    Verifies the gate responds to image degradation level.
    """
    buckets = {"70-79": (70, 79), "80-89": (80, 89), "90-100": (90, 100)}
    results = {}
    for name, (lo, hi) in buckets.items():
        mask = (jpeg_qualities >= lo) & (jpeg_qualities <= hi)
        if mask.sum() == 0:
            continue
        results[name] = accuracy_score(
            labels[mask].cpu().numpy(),
            logits[mask].argmax(dim=1).cpu().numpy()
        )
    return results