"""
TIMM MODEL NAMES
----------------
  convnext_base  -> convnext_base
  swin_v2_s      -> swinv2_base_window8_256
  vit_b_16       -> vit_base_patch16_224
  vit_b_32       -> vit_base_patch32_224
  dino_vits8     -> vit_small_patch8_224.dino
"""

import torch
import torch.nn as nn
import timm
from config import BackboneConfig


# Maps config names to timm model strings and their feature dimensions
TIMM_MODEL_MAP = {
    "convnext_base": ("convnext_base",           1024),
    "swin_v2_s":     ("swinv2_base_window8_256", 1024),
    "vit_b_16":      ("vit_base_patch16_224",     768),
    "vit_b_32":      ("vit_base_patch32_224",     768),
    "dino_vits8":    ("vit_small_patch8_224.dino", 384),
}


class SpatialBranch(nn.Module):
    """
    Pretrained backbone loaded via timm with a linear projection head.
    Outputs a 512-dim feature vector for the fusion module.
    """

    def __init__(self, cfg: BackboneConfig, feature_dim: int = 512):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim

        if cfg.name not in TIMM_MODEL_MAP:
            raise ValueError(
                f"Unknown backbone '{cfg.name}'. "
                f"Choose from: {list(TIMM_MODEL_MAP.keys())}"
            )

        timm_name, backbone_dim = TIMM_MODEL_MAP[cfg.name]

        # num_classes=0 
        self.backbone = timm.create_model(
            timm_name,
            pretrained=cfg.pretrained,
            num_classes=0,        
        )

        self.projection = nn.Linear(backbone_dim, feature_dim)

        if cfg.frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, normalised to ImageNet stats

        Returns:
            (B, feature_dim) feature vector
        """
        features = self.backbone(x)     # (B, backbone_dim) 
        if features.dim() == 4:
            features = features.flatten(start_dim=1)

        return self.projection(features)  # (B, feature_dim)