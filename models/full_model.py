"""
At TRAINING time, forward() returns:
    joint_logits, spatial_aux_logits, freq_aux_logits, gate_values, freq_patches

    freq_patches is returned so the training loop can compute the cleaner
    reconstruction loss on real-image patches only:
        real_mask = (labels == 0)
        recon_loss = model.freq_branch.cleaner.reconstruction_loss(
                         freq_patches[real_mask])

At INFERENCE time (training=False), forward() returns:
    joint_logits only — auxiliary outputs and gate values are not returned.
"""

import torch
import torch.nn as nn
from config import Config
from models.spatial_branch import SpatialBranch
from models.frequency_branch import FrequencyBranch, FrequencyBranchV2
from models.fusion import build_fusion


SPATIAL_FEAT_DIM = 512
FREQ_FEAT_DIM    = 256


class ASFRModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.spatial_branch = SpatialBranch(cfg.backbone, feature_dim=SPATIAL_FEAT_DIM)
        self.freq_branch    = FrequencyBranch(cfg.frequency, feature_dim=FREQ_FEAT_DIM)
        self.fusion         = build_fusion(cfg.fusion, SPATIAL_FEAT_DIM, FREQ_FEAT_DIM)

        # Joint head input dim depends on fusion mode
        joint_in_dim = self._joint_in_dim()
        self.joint_head = nn.Linear(joint_in_dim, 2)

        # Spatial auxiliary head (training only)
        if cfg.loss.use_auxiliary_heads:
            self.spatial_aux_head = nn.Linear(SPATIAL_FEAT_DIM, 2)
        else:
            self.spatial_aux_head = None

    def _joint_in_dim(self):
        mode = self.cfg.fusion.mode
        if mode == "joint_only":
            return SPATIAL_FEAT_DIM + FREQ_FEAT_DIM
        elif mode == "scalar":
            return SPATIAL_FEAT_DIM + FREQ_FEAT_DIM
        elif mode == "gating":
            return min(SPATIAL_FEAT_DIM, FREQ_FEAT_DIM)  # common dim from GatingFusion
        raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, x, training=False):
        """
        Args:
            x:        (B, C, H, W) image batch
            training: If True, return auxiliary outputs for loss computation.

        Returns (training=True):
            dict with joint_logits, spatial_aux_logits, freq_aux_logits,
                       gate_values (or scalars), freq_patches

        Returns (training=False):
            dict with joint_logits only
        """
        # Spatial branch
        spatial_feat = self.spatial_branch(x)

        # Frequency branch — returns (features, aux_logits, patch)
        freq_feat, freq_aux_logits, freq_patches = self.freq_branch(x)

        # Fusion
        fused, gate_info = self.fusion(spatial_feat, freq_feat)

        # Joint prediction
        joint_logits = self.joint_head(fused)

        if not training:
            return {"joint_logits": joint_logits}

        # Auxiliary outputs for loss computation
        spatial_aux_logits = (self.spatial_aux_head(spatial_feat)
                              if self.spatial_aux_head is not None else None)

        out = {
            "joint_logits":       joint_logits,
            "spatial_aux_logits": spatial_aux_logits,
            "freq_aux_logits":    freq_aux_logits,
            "freq_patches":       freq_patches,  # for cleaner recon loss on real images
        }

        # gate_info is gate tensor for gating mode, scalar dict for scalar mode, None for joint_only
        if isinstance(gate_info, torch.Tensor):
            out["gate_values"] = gate_info
        elif isinstance(gate_info, dict):
            out["scalars"] = gate_info

        return out



# ASFRModelV2 — full model with FrequencyBranchV2 + dual-transform support
class ASFRModelV2(nn.Module):
    """
    Full ASFR model using the v2 frequency pipeline.

    Differences from ASFRModel:
      - Uses FrequencyBranchV2 instead of FrequencyBranch
      - forward() accepts (aug_x, clean_x) — spatial branch uses aug_x,
        frequency branch uses clean_x for patch selection
      - Designed for use with get_deepdetect_dual_loaders

    All existing experiments use ASFRModel (unchanged).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.spatial_branch = SpatialBranch(cfg.backbone, feature_dim=SPATIAL_FEAT_DIM)
        self.freq_branch    = FrequencyBranchV2(cfg.frequency, feature_dim=FREQ_FEAT_DIM)
        self.fusion         = build_fusion(cfg.fusion, SPATIAL_FEAT_DIM, FREQ_FEAT_DIM)

        joint_in_dim = self._joint_in_dim()
        self.joint_head = nn.Linear(joint_in_dim, 2)

        if cfg.loss.use_auxiliary_heads:
            self.spatial_aux_head = nn.Linear(SPATIAL_FEAT_DIM, 2)
        else:
            self.spatial_aux_head = None

    def _joint_in_dim(self):
        mode = self.cfg.fusion.mode
        if mode in ("joint_only", "scalar"):
            return SPATIAL_FEAT_DIM + FREQ_FEAT_DIM
        elif mode == "gating":
            return min(SPATIAL_FEAT_DIM, FREQ_FEAT_DIM)
        raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, aug_x: torch.Tensor, clean_x: torch.Tensor,
                training: bool = False) -> dict:
        """
        Args:
            aug_x:    (B, C, H, W) augmented image — for spatial branch
            clean_x:  (B, C, H, W) clean image — for frequency branch patch selection
            training: if True, return auxiliary outputs

        Returns (training=True):
            dict with joint_logits, spatial_aux_logits, freq_aux_logits,
                       gate_values (or scalars), freq_patches

        Returns (training=False):
            dict with joint_logits only
        """
        spatial_feat = self.spatial_branch(aug_x)
        freq_feat, freq_aux_logits, freq_patches = self.freq_branch(clean_x)

        fused, gate_info = self.fusion(spatial_feat, freq_feat)
        joint_logits     = self.joint_head(fused)

        if not training:
            return {"joint_logits": joint_logits}

        spatial_aux_logits = (self.spatial_aux_head(spatial_feat)
                              if self.spatial_aux_head is not None else None)

        out = {
            "joint_logits":       joint_logits,
            "spatial_aux_logits": spatial_aux_logits,
            "freq_aux_logits":    freq_aux_logits,
            "freq_patches":       freq_patches,
        }

        if isinstance(gate_info, torch.Tensor):
            out["gate_values"] = gate_info
        elif isinstance(gate_info, dict):
            out["scalars"] = gate_info

        return out