"""
DESIGN 
------
The cleaner has two components:

  1. PerceptionModule — a small classifier that looks at the input patch and
     outputs soft probabilities over three degradation types:
       - clean:      no significant degradation
       - blurry:     Gaussian blur has softened high-frequency content
       - compressed: JPEG compression has introduced blocking artifacts

  2. Three learnable 3x3 conv filters, one specialising in each degradation type.
     The soft probabilities from the perception module weight how much each
     filter contributes to the final cleaned output. A heavily compressed patch
     will up-weight the compression filter; a clean patch will up-weight the
     clean filter.

This makes the cleaner degradation-aware rather than a plain conv stack as
it adapts its cleaning strategy per patch based on what kind of degradation
it detects.

TRAINING (real images only)
-----------------------------
The cleaner is trained separately using MAE reconstruction loss on real images
only. Fake images do not contribute to the reconstruction loss. This encourages
the cleaner to specialise in modelling high-frequency statistics of real camera
images, without learning to repair generator-specific artifacts.

When fake images pass through the trained cleaner, frequency inconsistencies
introduced by the generator are not repaired they survive and are picked up
as stronger anomaly signals by the SRM filters downstream.

Monitor during cleaner training:
  - Reconstruction loss on REAL images: should decrease (cleaner learning)
  - Reconstruction loss on FAKE images: should stay higher (artifacts not repaired)
  - If fake loss matches real: cleaner is too powerful, reduce capacity or steps

CRITICAL SIZE CONSTRAINT
-------------------------
Keep n_filters=3 (one per degradation type). Do not increase this.

PAPER SOURCES
-------------
Patch + perception + enhancement module concept:
  Chen, Yao & Niu (2024). "A Single Simple Patch is All You Need for
  AI-generated Image Detection." arXiv:2402.01123v2.

Real-only training / one-class specialisation idea:
  Cai, Ren, Chen & Lian (2025). "AI-Generated Image Detection in Degraded
  Scenarios." In Advanced Intelligent Computing Technology and Applications,
  pp. 521-532. Springer Nature Singapore.

----------
This is an original team design adapting ideas from both papers above.
The perception-guided soft weighting and the specific filter/training setup
are not direct reimplementations of either paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# Perception module
class PerceptionModule(nn.Module):
    """
    Small classifier that predicts the degradation type of an input patch.
    Outputs soft probabilities over [clean, blurry, compressed].

    These probabilities are used to weight the three specialised cleaning
    filters in DegradationAwareCleaner. The cleaner adapts per patch.

    Deliberately lightweight: 2 conv layers + global avg pool + linear.
    It only needs to distinguish three coarse degradation types, not classify
    image content.
    """

    DEGRADATION_TYPES = ["clean", "blurry", "compressed"]  # index 0, 1, 2

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 32, 1, 1)
        )
        self.classifier = nn.Linear(32, 3)  # 3 degradation types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) patch tensor

        Returns:
            (B, 3) soft probabilities over [clean, blurry, compressed]
        """
        feat = self.features(x).flatten(start_dim=1)  # (B, 32)
        return F.softmax(self.classifier(feat), dim=1)  # (B, 3)


# Degradation-aware cleaner
class DegradationAwareCleaner(nn.Module):
    """
    Perception-guided cleaner: three specialised filters weighted by degradation type.

    Forward pass:
      1. Perception module predicts degradation probabilities (B, 3)
      2. Each of the 3 filters is applied to the input patch independently
      3. Filter outputs are weighted by the perception probabilities and summed
      4. Residual connection adds back the original input
    """

    def __init__(self, n_filters: int = 3):
        super().__init__()
        assert n_filters == 3, (
            "n_filters must be 3 — one per degradation type (clean, blurry, compressed)"
        )

        self.perception = PerceptionModule(in_channels=3)

        # Three specialised 3x3 conv filters, one per degradation type.
        # Each is a single Conv2d: 3 input channels -> 3 output channels.
        # They are kept separate so each can specialise independently during training.
        self.filter_clean      = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.filter_blurry     = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.filter_compressed = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image or patch tensor, values in [0, 1]

        Returns:
            (B, 3, H, W) cleaned patch.
            Each image in the batch gets a different filter blend based on
            what the perception module detected about its degradation type.
        """
        # Perception module predicts degradation type probabilities
        # Weights shape: (B, 3) — [p_clean, p_blurry, p_compressed] per image
        weights = self.perception(x)  # (B, 3)

        out_clean      = self.filter_clean(x)       # (B, 3, H, W)
        out_blurry     = self.filter_blurry(x)      # (B, 3, H, W)
        out_compressed = self.filter_compressed(x)  # (B, 3, H, W)

        # Weight each filter output by its perception probability and sum.
        # weights[:, i] is (B,) — reshape to (B, 1, 1, 1) to broadcast over C, H, W
        w_clean      = weights[:, 0].view(-1, 1, 1, 1)
        w_blurry     = weights[:, 1].view(-1, 1, 1, 1)
        w_compressed = weights[:, 2].view(-1, 1, 1, 1)

        blended = (w_clean      * out_clean +
                   w_blurry     * out_blurry +
                   w_compressed * out_compressed)  # (B, 3, H, W)

        # Residual connection — if all filters output zero, input passes through
        return x + blended

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        MAE reconstruction loss for cleaner training on real images only.

        The goal: the cleaned output should be close to the original input
        (the cleaner learns to model natural statistics, not transform them).
        Fake images must NOT be passed to this method during training.

        Args:
            x: (B, 3, H, W) batch of REAL images only

        Returns:
            Scalar MAE loss
        """
        cleaned = self.forward(x)
        return F.l1_loss(cleaned, x)