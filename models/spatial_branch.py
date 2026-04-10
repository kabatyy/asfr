"""
TIMM MODEL NAMES
----------------
  convnext_base  -> convnext_base
  swin_v2_s      -> swinv2_base_window8_256  (auto-resized to 256x256)
  vit_b_16       -> vit_base_patch16_224
  vit_b_32       -> vit_base_patch32_224
  dino_vits8     -> vit_small_patch8_224.dino
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import BackboneConfig


# (timm_name, feature_dim, needs_img_size, forced_res)
# forced_res: resize inputs to this resolution in forward() — None means no resize
TIMM_MODEL_MAP = {
    "convnext_base": ("convnext_base",            1024, False, None),
    "swin_v2_s":     ("swinv2_base_window8_256",  1024, True,  256),
    "vit_b_16":      ("vit_base_patch16_224",      768, True,  None),
    "vit_b_32":      ("vit_base_patch32_224",      768, True,  None),
    "dino_vits8":    ("vit_small_patch8_224.dino", 384, True,  None),
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

        timm_name, backbone_dim, needs_img_size, forced_res = TIMM_MODEL_MAP[cfg.name]

        # Store forced resolution — inputs will be resized in forward() if set
        # swin_v2_s requires exactly 256x256 regardless of dataset image size
        self.forced_res = forced_res

        # num_classes=0 strips the classification head — backbone returns features directly
        kwargs = {"pretrained": cfg.pretrained, "num_classes": 0}
        if needs_img_size:
            kwargs["img_size"] = forced_res if forced_res else cfg.img_size

        self.backbone = timm.create_model(timm_name, **kwargs)
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
        # Resize if backbone requires a specific resolution
        # swin_v2_s: 32x32 or 224x224 inputs are upsampled to 256x256 here
        # All other backbones: no resize, input passes through as-is
        if self.forced_res is not None:
            x = F.interpolate(
                x, size=(self.forced_res, self.forced_res),
                mode="bilinear", align_corners=False
            )

        features = self.backbone(x)

        # Some timm models return (B, dim, 1, 1) — flatten spatial dims
        if features.dim() == 4:
            features = features.flatten(start_dim=1)

        return self.projection(features)