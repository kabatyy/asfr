"""
THREE FUSION MODES 
-------------------
  1. joint_only  — concatenate features, single shared classifier. BASELINE.
                   HIGH collapse risk: backbone dominates, freq branch starves.

  2. scalar      — learnable (a, b) scalars weight each branch globally.
                   CRITICAL: softmax applied so a + b = 1 always. Without this,
                   b can collapse to zero (freq branch silently ignored).
                   Monitor b every epoch — if < 0.1 by ep20, increase freq_aux_weight.

  3. gating      — small MLP outputs a per-sample gate value in [0, 1].
                   MAIN SCIENTIFIC CONTRIBUTION of the project.
                   Gate should be HIGH for clean GAN images (strong spectral artifacts).
                   Gate should be LOW for JPEG-compressed images (freq signal destroyed).
                   RISK: gate collapses to near-constant output (not actually adaptive).
                   FIX: diversity regulariser in losses/diversity.py
                   REQUIRED METRIC: report gate entropy on test set (must be > 0.3 nats).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FusionConfig


class ScalarFusion(nn.Module):
    """
    Weighted combination using two learnable scalars (a, b).
    Softmax ensures a + b = 1 at all times, preventing freq branch collapse.
    Output is the concatenation of weighted features for the joint head.
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(0.0))  # spatial weight (raw, before softmax)
        self.raw_b = nn.Parameter(torch.tensor(0.0))  # freq weight    (raw, before softmax)

    def forward(self, spatial_feat, freq_feat):
        weights = F.softmax(torch.stack([self.raw_a, self.raw_b]), dim=0)
        a, b = weights[0], weights[1]
        fused = torch.cat([a * spatial_feat, b * freq_feat], dim=1)
        return fused, {"scalar_spatial": a.item(), "scalar_freq": b.item()}

    def get_scalars(self):
        """Return current (a, b) after softmax. Use for logging."""
        weights = F.softmax(torch.stack([self.raw_a, self.raw_b]), dim=0)
        return weights[0].item(), weights[1].item()


class GatingFusion(nn.Module):
    """
    Per-sample adaptive gating via a small MLP.

    gate = sigmoid(MLP(concat(spatial_feat, freq_feat)))  — scalar in [0, 1]
    fused = gate * freq_feat + (1 - gate) * spatial_feat

    The gate adapts per image: high for clean GAN images with strong spectral
    artifacts, low for compressed images where the freq signal is degraded.

    gate_init_bias > 0 gives the freq branch a slight head-start before the
    pretrained backbone's stronger gradients pull the gate toward 0.
    """

    def __init__(self, cfg: FusionConfig, spatial_dim: int, freq_dim: int):
        super().__init__()
        # Project both branches to a common dim before gating
        # so the weighted sum is well-defined
        self.common_dim = min(spatial_dim, freq_dim)
        self.spatial_proj = nn.Linear(spatial_dim, self.common_dim)
        self.freq_proj    = nn.Linear(freq_dim,    self.common_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, cfg.gate_init_bias)

    def forward(self, spatial_feat, freq_feat):
        # Compute gate from original (unrojected) features for full information
        combined = torch.cat([spatial_feat, freq_feat], dim=1)  # (B, D_s + D_f)
        gate = torch.sigmoid(self.gate_mlp(combined)).squeeze(1)  # (B,)
        # Project to common dim, then gate-blend
        s = self.spatial_proj(spatial_feat)   # (B, common_dim)
        f = self.freq_proj(freq_feat)         # (B, common_dim)
        fused = gate.unsqueeze(1) * f + (1 - gate).unsqueeze(1) * s
        return fused, gate  # gate returned for diversity loss + logging


class JointOnlyFusion(nn.Module):
    """
    Simple concatenation — baseline only.
    Both branches feed a single shared classifier with no weighting.
    High risk of freq branch collapse due to backbone gradient dominance.
    """

    def forward(self, spatial_feat, freq_feat):
        return torch.cat([spatial_feat, freq_feat], dim=1), None


def build_fusion(cfg: FusionConfig, spatial_dim: int, freq_dim: int) -> nn.Module:
    if cfg.mode == "scalar":
        return ScalarFusion(cfg)
    elif cfg.mode == "gating":
        return GatingFusion(cfg, spatial_dim, freq_dim)
    elif cfg.mode == "joint_only":
        return JointOnlyFusion()
    else:
        raise ValueError(f"Unknown fusion mode: {cfg.mode}")