import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LossConfig


class AuxiliaryLoss(nn.Module):  # fixed: Module not module
    """Combined loss with optional auxiliary heads."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, joint_logits, labels,
                spatial_aux_logits=None, freq_aux_logits=None):
        """Compute joint loss + optional auxiliary losses."""
        joint_loss = F.cross_entropy(joint_logits, labels)
        total = joint_loss
        zero = torch.zeros(1, device=joint_logits.device)
        spatial_aux_loss = zero 
        freq_aux_loss = zero

        if self.cfg.use_auxiliary_heads: 
            if spatial_aux_logits is not None:
                spatial_aux_loss = F.cross_entropy(spatial_aux_logits, labels)
                total = total + self.cfg.spatial_aux_weight * spatial_aux_loss
            if freq_aux_logits is not None:
                freq_aux_loss = F.cross_entropy(freq_aux_logits, labels)
                total = total + self.cfg.freq_aux_weight * freq_aux_loss

        return {  
            "total":       total,
            "joint":       joint_loss,
            "spatial_aux": spatial_aux_loss,
            "freq_aux":    freq_aux_loss,
        }