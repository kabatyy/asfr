"""
BACKBONE BENCHMARK 
------------------
We benchmark five backbones. The key scientific question is how much the
frequency branch ADDS (delta) to each backbone. A backbone that already
captures frequency information might benefit less from the frequency branch.

  convnext_base  — Large 7x7 kernels; captures more high-freq detail than
                   ResNet but still trained to suppress noise. Moderate delta.

  dino_vits8     — Self-supervised, patch=8; preserves fine-grained local
                   detail that supervised models discard. Low delta expected

  swin_v2_s      — Local window attention; cannot learn global frequency
                   patterns (never processes full image at once). Moderate-high delta.

  vit_b_16       — patch=16; any artifact < 16px is invisible. High delta.

  vit_b_32       — patch=32; even larger blind spot. Very high delta.

CONTROLLED EXPERIMENT
---------------------
We run vit_b_16 AND vit_b_32 together. These are identical except patch size.
Any difference in delta is caused by patch size alone.

FROZEN vs FINE-TUNED
--------------------
Run each backbone frozen (cfg.frozen=True) and fine-tuned. If the frequency
branch helps more when frozen, the backbone was learning to capture spectral
info during fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from config import BackboneConfig


# Output feature dimensions for each backbone (before classification head)
_BACKBONE_DIMS = {
    "convnext_base": 1024,
    "dino_vits8":    384,
    "swin_v2_s":     768,
    "vit_b_16":      768,
    "vit_b_32":      768,
}


class SpatialBranch(nn.Module):
    """
    Wraps a pretrained vision backbone with a linear projection head.
    Strips the backbone's classification head and outputs a feature vector
    of size feature_dim for the fusion module.
    """

    def __init__(self, cfg: BackboneConfig, feature_dim: int = 512):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim
        self.backbone = self._load_backbone()
        self.projection = nn.Linear(_BACKBONE_DIMS[cfg.name], feature_dim)

        if cfg.frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _load_backbone(self) -> nn.Module:
        """
        Load pretrained backbone and strip its classification head.
        Returns the backbone with its final classifier removed so it
        outputs a feature vector rather than class logits.
        """
        name = self.cfg.name
        weights = "DEFAULT" if self.cfg.pretrained else None

        if name == "convnext_base":
            model = tvm.convnext_base(weights=weights)
            # ConvNeXt classifier: model.classifier = [LayerNorm, Flatten, Linear]
            # Remove the final Linear to get 1024-dim features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            return model

        elif name == "swin_v2_s":
            model = tvm.swin_v2_s(weights=weights)
            # Swin head: model.head = Linear(768, n_classes)
            model.head = nn.Identity()
            return model

        elif name == "vit_b_16":
            model = tvm.vit_b_16(weights=weights)
            # ViT head: model.heads = Sequential(Linear(768, n_classes))
            model.heads = nn.Identity()
            return model

        elif name == "vit_b_32":
            model = tvm.vit_b_32(weights=weights)
            model.heads = nn.Identity()
            return model

        elif name == "dino_vits8":
            # DINO is not in torchvision — load from torch.hub
            model = torch.hub.load(
                "facebookresearch/dino:main",
                "dino_vits8",
                pretrained=self.cfg.pretrained,
            )
            # DINO ViT has no classification head by default — outputs 384-dim CLS token
            return model

        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, normalised to ImageNet stats

        Returns:
            (B, feature_dim) feature vector
        """
        features = self.backbone(x)  # (B, backbone_dim)

        # DINO returns a tuple (cls_token, patch_tokens) — take cls token only
        if isinstance(features, tuple):
            features = features[0]

        # Some backbones return (B, dim, 1, 1) — flatten spatial dims
        if features.dim() == 4:
            features = features.flatten(start_dim=1)

        return self.projection(features)  # (B, feature_dim)