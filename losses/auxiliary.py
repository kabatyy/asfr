"""
Total loss during training:

    total = joint_loss
          + 0.3 x spatial_aux_loss
          + 0.5 x freq_aux_loss
          + 0.1 x cleaner_recon_loss  (real images only)

The frequency branch weight (0.5) is intentionally higher than spatial (0.3)
to compensate for the weaker initial gradient from the randomly initialised branch.

The cleaner reconstruction loss is computed on real images only. Fake images
must not contribute. If they do, the cleaner learns to repair generator
artifacts, erasing the signal the SRM filters are meant to detect.

At inference: auxiliary heads are discarded. Only joint_head is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LossConfig


class AuxiliaryLoss(nn.Module):
    """Combined loss with optional auxiliary heads."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, joint_logits, labels,
                spatial_aux_logits=None, freq_aux_logits=None,
                cleaner_recon_loss=None):
        """Compute joint loss + optional auxiliary losses."""
        joint_loss = F.cross_entropy(joint_logits, labels)
        total = joint_loss
        zero = torch.zeros(1, device=joint_logits.device)
        spatial_aux_loss = zero
        freq_aux_loss    = zero
        recon_loss       = zero

        if self.cfg.use_auxiliary_heads:
            if spatial_aux_logits is not None:
                spatial_aux_loss = F.cross_entropy(spatial_aux_logits, labels)
                total = total + self.cfg.spatial_aux_weight * spatial_aux_loss
            if freq_aux_logits is not None:
                freq_aux_loss = F.cross_entropy(freq_aux_logits, labels)
                total = total + self.cfg.freq_aux_weight * freq_aux_loss

        if cleaner_recon_loss is not None:
            recon_loss = cleaner_recon_loss
            total = total + self.cfg.cleaner_recon_weight * recon_loss

        return {
            "total":         total,
            "joint":         joint_loss,
            "spatial_aux":   spatial_aux_loss,
            "freq_aux":      freq_aux_loss,
            "cleaner_recon": recon_loss,
        }