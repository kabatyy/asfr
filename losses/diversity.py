"""
losses/diversity.py — Entropy diversity regulariser for the gating mechanism.

WHAT THE GATE IS
----------------
In gating fusion mode, a small MLP (defined in models/fusion.py::GatingFusion)
takes the spatial and frequency feature vectors for a single image and outputs
one number between 0 and 1. That number is the gate value. It controls how much
the frequency branch contributes to the final prediction for that specific image:

    fused = gate * freq_features + (1 - gate) * spatial_features

The scientific claim of this project is that the gate learns to adapt per image:
  - High gate (~1.0): trust the frequency branch more.
    Example: a clean GAN image with strong spectral artifacts.
  - Low gate (~0.0): trust the spatial branch more.
    Example: a heavily JPEG-compressed image where compression has destroyed
    the frequency artifacts the freq branch relies on.

THE FAILURE MODE THIS PREVENTS
--------------------------------
The gate MLP might learn to output roughly the same value for every image —
say 0.3 regardless of what the image looks like. If that happens, the model
is just doing a fixed weighted average, not per-sample adaptation. It would
still achieve decent accuracy, but the core scientific contribution (adaptive
fusion) would be fake. You would not detect this from accuracy alone.

HOW THIS REGULARISER WORKS
----------------------------
We compute the entropy of the gate value distribution across the batch.
Entropy is high when gate values are spread out (the gate is genuinely varying).
Entropy is low when gate values are all clustered together (the gate is constant).

    Diversity Loss = -Entropy(gate values across the batch)

This term is added to the total training loss. Because we are minimising loss,
the model is pushed to increase entropy — i.e. to spread out its gate values
rather than collapsing to a constant. The weight (default 0.1) controls how
strongly this is enforced relative to the classification loss.

WHEN TO WORRY
-------------
If gate entropy on the test set is < 0.3 nats, the gate has collapsed.
Increase diversity_weight in config.py and retrain from scratch.
"""

import torch
import torch.nn as nn


class DiversityRegulariser(nn.Module):
    """
    Penalises the gating network for outputting near-constant gate values.

    Used only in gating fusion mode (models/fusion.py::GatingFusion).
    Not needed for scalar or joint_only fusion.
    """

    def __init__(self, weight: float = 0.1, n_bins: int = 20):
        """
        Args:
            weight: How strongly to penalise constant gate outputs relative
                    to the classification loss. Default 0.1 from advisor notes.
                    Increase if gate entropy stays below 0.3 nats during training.
            n_bins: Number of histogram buckets used to estimate the gate value
                    distribution. 20 bins over [0, 1] gives 0.05-wide buckets,
                    which is fine-grained enough to detect collapse.
        """
        super().__init__()
        self.weight = weight
        self.n_bins = n_bins

    def forward(self, gate_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_values: (B,) gate outputs for the current batch, values in [0, 1].
                         These come from GatingFusion.forward() during training.

        Returns:
            Scalar penalty tensor. High when all gate values are similar (gate is
            constant). Low when gate values are spread across [0, 1] (gate is
            genuinely adapting per image). Add this to your total training loss.
        """
        # bin gate values into a soft histogram over [0, 1].
        # We use differentiable soft binning rather than torch.histc because
        # torch.histc produces integer counts with no gradient. We need gradients
        # to flow back through gate_values into the gating MLP — otherwise this
        # penalty has no effect on the gating network's weights.
        bin_edges = torch.linspace(0.0, 1.0, self.n_bins + 1,
                                   device=gate_values.device)
        bin_width = 1.0 / self.n_bins
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2  # (n_bins,)

        # For each of the B gate values, compute its distance to each of the
        # n_bins bin centres. Shape: (B, n_bins).
        distances = torch.abs(gate_values.unsqueeze(1) - bin_centres.unsqueeze(0))

        # Triangular kernel: a gate value contributes 1.0 to its nearest bin
        # centre, dropping linearly to 0.0 at distance >= bin_width. Values
        # beyond one bin width away contribute nothing. This is differentiable
        # everywhere except exactly at the bin edges.
        soft_counts = torch.clamp(1.0 - distances / bin_width, min=0.0)  # (B, n_bins)

        # sum contributions across the batch to get a count per bin,
        # then normalise to a probability distribution.
        bin_probs = soft_counts.sum(dim=0)      # (n_bins,)
        bin_probs = bin_probs / bin_probs.sum() # sums to 1.0

        # compute entropy H = -sum(p * log(p)) in nats.
        # Clamp before log to avoid log(0) = -inf for empty bins.
        # Empty bins contribute 0 to entropy.
        entropy = -(bin_probs * torch.log(bin_probs.clamp(min=1e-8))).sum()

        # return the penalty. Because we minimise total loss, the model
        # is pushed to maximise entropy (spread out gate values), which is what
        # we want. The penalty is high when entropy is low (gate is constant).
        return self.weight * (-entropy)