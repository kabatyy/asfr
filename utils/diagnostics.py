"""
Training diagnostics to detect silent failure modes.
Things to monitor during training:

  1. Frequency branch gradient norms
     Near zero = gradient starvation. Increase freq_aux_weight.

  2. Fusion scalar / gate values
     Scalar: if b < 0.1 by epoch 20, freq branch is being ignored.
     Gating: if gate entropy < 0.3 nats on test set, gate has collapsed.

  3. Frequency-only standalone accuracy
     Must be >= 70% before building fusion on top.
"""

import numpy as np

def log_freq_branch_grad_norms(model):
    """
    Compute gradient norms for frequency and spatial branch parameters.
    Call after loss.backward(), before optimizer.step().
    Returns dict with freq_branch_grad_norm and spatial_branch_grad_norm.
    """
    def _norm(module):
        total, count = 0.0, 0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
                count += 1
        return total ** 0.5 if count > 0 else 0.0

    return {
        "freq_branch_grad_norm":    _norm(model.freq_branch),
        "spatial_branch_grad_norm": _norm(model.spatial_branch),
    }


def log_fusion_scalars(model):
    """
    For scalar fusion mode: return current (a, b) after softmax.
    Returns dict with scalar_spatial and scalar_freq.
    """
    a, b = model.fusion.get_scalars()
    return {"scalar_spatial": a, "scalar_freq": b}


def compute_gate_entropy(gate_values, n_bins=20):
    """
    Compute entropy (nats) of the gate value distribution.
    Below 0.3 nats indicates the gate has collapsed to near-constant output.
    """
    gate_np = gate_values.detach().cpu().numpy()
    counts, _ = np.histogram(gate_np, bins=n_bins, range=(0.0, 1.0))
    probs = counts / (counts.sum() + 1e-8)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    return float(entropy)


def check_warning_signs(
    freq_only_acc=None,
    fused_acc=None,
    spatial_only_acc=None,
    gate_entropy=None,
    epoch=None,
    total_epochs=None,
):
    """
    Check for known failure modes. Returns a list of warning strings.

    Warnings are only raised in the final 20% of training (when epoch and
    total_epochs are provided) to give the model time to settle first.
    Pass epoch=None to always check regardless of training stage.
    """
    # Only check in the final 20% of training
    if epoch is not None and total_epochs is not None:
        if epoch < 0.8 * total_epochs:
            return []

    warnings = []

    if freq_only_acc is not None and freq_only_acc < 0.60:
        warnings.append(
            f"UserWarning: Frequency-only accuracy is {freq_only_acc:.1%}, which is below "
            f"the 60% hard stop. The FFT representation is not capturing useful signal. "
            f"Check that fftshift=True, log-scaling is applied, and normalisation is "
            f"correct. Do not proceed to fusion experiments until this is resolved."
        )

    if fused_acc is not None and spatial_only_acc is not None:
        if fused_acc <= spatial_only_acc:
            warnings.append(
                f"UserWarning: Fused accuracy ({fused_acc:.1%}) is not higher than "
                f"spatial-only accuracy ({spatial_only_acc:.1%}). The frequency branch "
                f"is not contributing. Check gradient norms and gate/scalar values."
            )

    if gate_entropy is not None and gate_entropy < 0.3:
        warnings.append(
            f"UserWarning: Gate entropy is {gate_entropy:.3f} nats, below the 0.3 "
            f"threshold. The gate is outputting near-constant values and is not adapting "
            f"per sample. Try increasing diversity_weight in config and retraining."
        )

    return warnings