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
    Near-zero freq_branch_grad_norm = gradient starvation.

    Returns dict with 'freq_branch_grad_norm', 'spatial_branch_grad_norm'.
    """
    def _branch_norm(module):
        total = 0.0
        count = 0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
                count += 1
        return (total ** 0.5) if count > 0 else 0.0

    return {
        "freq_branch_grad_norm":    _branch_norm(model.freq_branch),
        "spatial_branch_grad_norm": _branch_norm(model.spatial_branch),
    }


def log_fusion_scalars(model):
    """
    For scalar fusion mode: return current (a, b) after softmax.
    Warn if b < 0.1.
    """
    a, b = model.fusion.get_scalars()
    if b < 0.1:
        print(f"WARNING: freq scalar b={b:.3f} < 0.1 — "
              "freq branch being ignored. Increase freq_aux_weight.")
    return {"scalar_spatial": a, "scalar_freq": b}


def compute_gate_entropy(gate_values, n_bins=20):
    """
    Compute entropy (nats) of the gate value distribution.
    Below 0.3 nats = gate has collapsed to near-constant output.

    Args:
        gate_values: 1D tensor of gate outputs in [0, 1]

    Returns:
        Entropy in nats (float).
    """
    gate_np = gate_values.detach().cpu().numpy()
    counts, _ = np.histogram(gate_np, bins=n_bins, range=(0.0, 1.0))
    probs = counts / (counts.sum() + 1e-8)
    # -sum(p * log(p)), skip zero bins
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    return float(entropy)


def check_warning_signs(
    freq_only_acc=None,
    fused_acc=None,
    spatial_only_acc=None,
    gate_entropy=None,
):
    """
    Check the three warning signs.
    Returns a list of warning strings (empty if all clear).
    """
    warnings = []

    if freq_only_acc is not None and freq_only_acc < 0.60:
        warnings.append(
            f"WARNING: Freq-only accuracy {freq_only_acc:.1%} < 60%. "
            "Check fftshift, log-scaling, and normalisation before building fusion."
        )
    if fused_acc is not None and spatial_only_acc is not None:
        if fused_acc <= spatial_only_acc:
            warnings.append(
                f"WARNING: Fused accuracy {fused_acc:.1%} <= spatial-only "
                f"{spatial_only_acc:.1%}. Freq branch is adding noise. "
                "Check gradient norms and scalar/gate values for collapse."
            )
    if gate_entropy is not None and gate_entropy < 0.3:
        warnings.append(
            f"WARNING: Gate entropy {gate_entropy:.3f} nats < 0.3. "
            "Gate collapsed to near-constant output. Add entropy regulariser and retrain."
        )

    return warnings