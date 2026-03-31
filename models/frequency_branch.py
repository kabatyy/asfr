"""
PIPELINE PER IMAGE
------------------
  raw image
    → select_flat_patch_batch   find lowest-variance region (Chen et al. 2024)
    → DegradationAwareCleaner   clean the patch only — 3 filters weighted by perception module
    → SRMFilterLayer            fixed kernels: suppress content, extract noise residual
    → fft_spectrum_tensor       log-magnitude FFT with fftshift
    → FrequencyCNN              small CNN learns to detect spectral artifacts
    → aux_head                  auxiliary classifier (training only — prevents gradient starvation)

GRADIENT STARVATION PREVENTION
--------------------------------
This branch has its own auxiliary classification head (aux_head) with its own
direct BCE loss (weight 0.5, defined in losses/auxiliary.py). This ensures the
frequency branch receives a strong gradient signal regardless of what the
backbone is doing. At inference, aux_head output is ignored and only the feature
vector is passed to fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FrequencyConfig
from utils.fft_utils import fft_spectrum_tensor
from utils.patch_select import select_flat_patch_batch
from models.cleaner import DegradationAwareCleaner


class FrequencyBranch(nn.Module):
    """
    Full frequency processing pipeline: cleaner → SRM → patch → FFT → CNN → features.
    """

    def __init__(self, cfg: FrequencyConfig, feature_dim: int = 256):
        super().__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim

        # Optional degradation-aware cleaner (3 filters max — see DegradationAwareCleaner)
        self.cleaner = DegradationAwareCleaner(n_filters=cfg.cleaner_filters) \
                       if cfg.cleaner_filters > 0 else None

        # Fixed SRM noise-residual filters (non-trainable buffers)
        self.srm = SRMFilterLayer() if cfg.srm_filters else nn.Identity()

        # Determine CNN input channels:
        # SRM produces 3 output channels (one per SRM kernel) per input channel.
        # RGB input (3ch) → SRM → 9 channels. Grayscale (1ch) → SRM → 3 channels.
        # If SRM is disabled, input channels pass through unchanged.
        srm_out_channels = 9 if cfg.srm_filters else 3

        # CNN that processes the log-FFT spectrum
        self.cnn = FrequencyCNN(in_channels=srm_out_channels, feature_dim=feature_dim)

        # Auxiliary classification head — training only, discarded at inference
        self.aux_head = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) raw image tensor, values in [0, 1]

        Returns:
            features:   (B, feature_dim) — passed to fusion module
            aux_logits: (B, 2)           — used for auxiliary loss during training
        """
        # Select flattest patch per image (Chen et al. 2024)
        # Cleaner and SRM operate on the patch only, not the full image
        x = select_flat_patch_batch(x, patch_size=self.cfg.patch_size)

        # Optional degradation-aware cleaner — operates on patch only
        if self.cleaner is not None:
            x = self.cleaner(x)

        # SRM filters — suppress content, extract noise residual
        x = self.srm(x)

        # log-magnitude FFT with fftshift
        x = fft_spectrum_tensor(x, fftshift=self.cfg.use_fftshift)

        # CNN forward → feature vector
        features = self.cnn(x)

        # Auxiliary head for training loss
        aux_logits = self.aux_head(features)

        return features, aux_logits




# SRM filter layer
class SRMFilterLayer(nn.Module):
    """
    Fixed Spatial Rich Model (SRM) filters.

    SRM filters suppress visible image content (objects, colours, textures)
    and extract the high-frequency noise residual. Real camera photographs
    have a characteristic noise pattern from the sensor. AI-generated images
    have a different noise pattern from the generation process. SRM makes
    this difference visible to the CNN downstream.

    Because these filters are fixed, they cannot overfit to specific generators. 
    The same mathematical operation is applied to every image.

    We use 3 standard SRM kernels applied to each of the 3 RGB channels,
    producing 9 output channels total.
    """

    # Three standard SRM high-pass kernels (3x3)
    # They approximate image derivatives
    # and suppress smooth/low-frequency content
    SRM_KERNELS = torch.tensor([
        # Kernel 1: horizontal gradient
        [[ 0,  0,  0],
         [ 0, -1,  1],
         [ 0,  0,  0]],
        # Kernel 2: Laplacian (all-direction high-pass)
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]],
        # Kernel 3: diagonal gradient
        [[ 0,  0,  0],
         [ 0, -1,  0],
         [ 0,  1,  0]],
    ], dtype=torch.float32)  # shape: (3, 3, 3)

    def __init__(self):
        super().__init__()

        # Build conv weight: (out_channels, in_channels, kH, kW)
        # Apply each of the 3 SRM kernels to each of the 3 RGB channels independently.
        # Result: 9 output channels (3 kernels × 3 input channels).
        # Each output channel isolates a specific noise direction in a specific colour channel.
        kernels = self.SRM_KERNELS  # (3, 3, 3) = (n_kernels, kH, kW)

        # Expand to (9, 1, 3, 3) then use groups=3 to apply per-channel
        # Alternatively: build a (9, 3, 3, 3) weight with zeros off-diagonal
        weight = torch.zeros(9, 3, 3, 3)
        for i, k in enumerate(kernels):     # i = kernel index (0-2)
            for c in range(3):              # c = RGB channel index (0-2)
                weight[i * 3 + c, c] = k   # kernel i applied to channel c -> output i*3+c

        # Register as buffer 
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor

        Returns:
            (B, 9, H, W) noise residual tensor — same spatial size, padding=1
        """
        return F.conv2d(x, self.weight, padding=1)

# Frequency CNN
class FrequencyCNN(nn.Module):
    """
    Lightweight CNN that processes the log-magnitude FFT spectrum after SRM filtering.

    Kept deliberately small — the input is already a highly processed noise residual
    spectrum, not a raw image. Heavy capacity is not needed and risks overfitting to
    the specific generators seen during training.

    Architecture: 4 conv blocks (conv → BN → ReLU → MaxPool), then global average
    pooling, then a linear projection to feature_dim.
    """

    def __init__(self, in_channels: int = 9, feature_dim: int = 256):
        super().__init__()

        self.blocks = nn.Sequential(
            # Block 1: in_channels → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8, W/8

            # Block 4: 128 → 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/16, W/16
        )

        # Global average pooling collapses spatial dims to a single vector
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) log-FFT spectrum

        Returns:
            (B, feature_dim) feature vector
        """
        x = self.blocks(x)
        x = self.gap(x)           # (B, 128, 1, 1)
        x = x.flatten(start_dim=1)  # (B, 128)
        x = self.proj(x)          # (B, feature_dim)
        return x