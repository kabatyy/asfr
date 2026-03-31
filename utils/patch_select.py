
"""
PAPER SOURCE
------------
The core idea comes from:
 
  Chen, Yao & Niu (2024). "A Single Simple Patch is All You Need for
  AI-generated Image Detection." arXiv:2402.01123.
 
  Key finding: generative models focus on rendering high-texture regions
  (faces, objects) convincingly, but neglect the hidden camera noise present
  in flat/simple regions. A single flat patch therefore carries a cleaner
  noise signal than the full image or a randomly chosen patch.
 
WHY WE SELECT A PATCH
----------------------
We do not run the FFT on the whole image. Instead we find the flattest
(lowest-variance) region and analyse that patch only.
 
High-texture regions (fur, foliage, fabric) contain so much natural visual
detail that any generation artifacts are buried in it. A flat region (sky,
wall, smooth skin, plain background) has almost no natural texture, so any
noise patterns found there are much more likely to be generation artifacts
from the AI model rather than natural image content.
 
HOW IT WORKS
------------
We slide a window across the image and compute the local pixel variance at
each position. The window with the lowest variance is the flattest region.
That window's position is used to extract the patch.
 
This is done on a single-channel (grayscale) version of the image for speed —
we only need variance to find the flat region, not colour information.
 
NOTE ON PATCH SIZE
------------------
At 224x224 (DeepDetect phase) we extract a 56x56 patch.
At 32x32 (CIFAKE prototype phase) a 56x56 patch does not fit — the entire
image is used instead (patch_size is clamped to image size).
We do not upsample a small patch to a larger size because this introduces interpolation
artifacts that contaminate the frequency signal you are trying to detect.
"""
import torch
import torch.nn.functional as F


def select_flat_patch(image: torch.Tensor, patch_size: int = 56) -> torch.Tensor:
    """
    Extract the lowest-variance (flattest) patch from a single image.

    Args:
        image:      (C, H, W) tensor, values in [0, 1]
        patch_size: Size of square patch to extract. Clamped to min(H, W)
                    if the image is smaller than patch_size.

    Returns:
        (C, patch_size, patch_size) tensor — the flattest patch found.
        If image is smaller than patch_size, returns the full image.
    """
    C, H, W = image.shape
    patch_size = min(patch_size, H, W)

    if patch_size == H and patch_size == W:
        return image  

    # Convert to grayscale for variance computation: (1, H, W)
    # Simple average across channels is sufficient
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

    # Local variance = E[x^2] - E[x]^2 computed with avg_pool2d
    kernel = torch.ones(1, 1, patch_size, patch_size,
                        device=image.device) / (patch_size * patch_size)
    local_mean_sq = F.conv2d(gray ** 2, kernel, padding=0)   # (1, 1, H', W')
    local_mean    = F.conv2d(gray,      kernel, padding=0)   # (1, 1, H', W')
    local_var     = (local_mean_sq - local_mean ** 2).squeeze()  # (H', W')

    # Find the position of minimum variance
    flat_idx = local_var.argmin()
    top  = flat_idx // local_var.shape[1]
    left = flat_idx  % local_var.shape[1]

    return image[:, top:top + patch_size, left:left + patch_size]


def select_flat_patch_batch(images: torch.Tensor, patch_size: int = 56) -> torch.Tensor:
    """
    Extract the flattest patch from each image in a batch.

    Args:
        images:     (B, C, H, W) tensor
        patch_size: Size of square patch. Clamped to image size if needed.

    Returns:
        (B, C, patch_size, patch_size) tensor.
    """
    B, C, H, W = images.shape
    patch_size = min(patch_size, H, W)
    patches = torch.stack([
        select_flat_patch(images[i], patch_size) for i in range(B)
    ])
    return patches