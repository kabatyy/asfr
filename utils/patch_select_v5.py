"""
utils/patch_select_v5.py — Skin-tone guided patch selection (v5).

WHY THIS EXISTS
---------------
The original variance-only selector (patch_select.py) finds the lowest-variance
region in the image. On general object images (CIFAKE) this works well — flat
regions like skies and walls are common and informative.

On face-dominated datasets (DeepDetect), the flattest region is almost always
a dark background corner — uninformative and spectrally near-empty. 38.9% of
patches selected by v1 on DeepDetect have mean brightness < 0.2 vs 1.0% on CIFAKE.

v5 fixes this by preferring patches in skin-tone regions (HSV-based detection),
then selecting the flattest among those. Falls back to v1 if no skin region found.

PERFORMANCE
-----------
v1 on DeepDetect: ~58% freq-only accuracy
v5 on DeepDetect: ~75% freq-only accuracy (with correct pipeline)

IMPLEMENTATION
--------------
Fully vectorised — all operations run on GPU via batched conv2d.
No Python loops over spatial positions. Only a small per-image loop for
argmax and slice (B iterations, not H*W iterations).

PAPER REFERENCE
---------------
Extends Chen, Yao & Niu (2024). "A Single Simple Patch is All You Need for
AI-generated Image Detection." arXiv:2402.01123.
We adapt their variance-only selector to face-dominated datasets.

VERSION HISTORY
---------------
v1 (patch_select.py):   variance-only — original, all CIFAKE experiments
v5 (this file):         skin-tone guided — DeepDetect and face-dominated datasets
"""

import torch
import torch.nn.functional as F

# ImageNet normalisation constants — needed for denormalisation before HSV
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _rgb_to_hsv_batch(rgb: torch.Tensor):
    """
    Batched RGB → HSV conversion. All operations on the input device.

    Args:
        rgb: (B, 3, H, W) tensor with values in [0, 1]

    Returns:
        h, s, v: each (B, H, W) tensor
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = rgb.max(1).values   # (B, H, W)
    minc = rgb.min(1).values
    v    = maxc
    s    = torch.where(maxc != 0,
                       (maxc - minc) / (maxc + 1e-8),
                       torch.zeros_like(maxc))
    rc   = (maxc - r) / (maxc - minc + 1e-8)
    gc   = (maxc - g) / (maxc - minc + 1e-8)
    bc   = (maxc - b) / (maxc - minc + 1e-8)
    h    = torch.where(r == maxc, bc - gc,
           torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h    = (h / 6.0) % 1.0
    return h, s, v


def select_flat_patch_v5(
    image: torch.Tensor,
    patch_size: int = 56,
    min_skin_density: float = 0.3,
) -> torch.Tensor:
    """
    Single-image v5 patch selector. Convenience wrapper around the batch version.

    Args:
        image:            (C, H, W) tensor, ImageNet-normalised
        patch_size:       size of square patch to extract
        min_skin_density: minimum fraction of skin pixels in a candidate patch

    Returns:
        (C, patch_size, patch_size) tensor
    """
    return select_flat_patch_v5_batch(
        image.unsqueeze(0), patch_size, min_skin_density
    ).squeeze(0)


def select_flat_patch_v5_batch(
    images: torch.Tensor,
    patch_size: int = 56,
    min_skin_density: float = 0.3,
) -> torch.Tensor:
    """
    Batched v5 patch selector. Fully vectorised — runs on GPU.

    Selects the flattest (lowest-variance) patch that contains sufficient
    skin-tone content. Falls back to v1 variance-only if no skin region found
    (handles non-face images gracefully).

    At 32x32 (CIFAKE), patch_size is clamped to image size and the full image
    is returned — identical behaviour to v1.

    Args:
        images:           (B, C, H, W) tensor, ImageNet-normalised
        patch_size:       size of square patch to extract
        min_skin_density: minimum fraction of skin pixels required in a patch

    Returns:
        (B, C, patch_size, patch_size) tensor
    """
    B, C, H, W = images.shape
    patch_size  = min(patch_size, H, W)

    # At small resolutions, return full image — same as v1
    if patch_size == H and patch_size == W:
        return images

    device = images.device

    # ── Step 1: variance map via batched conv2d ───────────────────────────────
    gray   = images.mean(dim=1, keepdim=True)           # (B, 1, H, W)
    kernel = torch.ones(1, 1, patch_size, patch_size,
                        device=device) / (patch_size ** 2)
    local_mean    = F.conv2d(gray,      kernel, padding=0)  # (B, 1, H', W')
    local_mean_sq = F.conv2d(gray ** 2, kernel, padding=0)
    local_var     = (local_mean_sq - local_mean ** 2).squeeze(1)  # (B, H', W')

    # ── Step 2: skin density map via batched conv2d ───────────────────────────
    mean_d     = _MEAN.to(device)
    std_d      = _STD.to(device)
    img_denorm = (images * std_d + mean_d).clamp(0, 1)   # (B, 3, H, W)

    h, s, v    = _rgb_to_hsv_batch(img_denorm)
    skin_mask  = (
        (h >= 0.0) & (h <= 0.1) &
        (s >= 0.1) & (s <= 0.7) &
        (v >= 0.2)
    ).float().unsqueeze(1)                                # (B, 1, H, W)

    skin_kernel = torch.ones(1, 1, patch_size, patch_size,
                             device=device) / (patch_size ** 2)
    local_skin  = F.conv2d(skin_mask, skin_kernel, padding=0).squeeze(1)  # (B, H', W')

    # ── Step 3: per-image argmax and slice ────────────────────────────────────
    patches = []
    for i in range(B):
        skin_present = (local_skin[i] >= min_skin_density)
        if skin_present.any():
            # Prefer flattest patch in skin region
            var_masked = local_var[i].clone()
            var_masked[~skin_present] = float("inf")
            flat_idx = var_masked.argmin()
        else:
            # Fall back to v1 — no skin region found
            flat_idx = local_var[i].argmin()

        top  = (flat_idx // local_var.shape[2]).item()
        left = (flat_idx  % local_var.shape[2]).item()
        patches.append(images[i, :, top:top + patch_size, left:left + patch_size])

    return torch.stack(patches)
