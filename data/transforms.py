"""
TRAINING AUGMENTATIONS 
----------------------------------------------
Inspired by the Random Degradation Simulator in:
  Cai, Ren, Chen & Lian (2025). "AI-Generated Image Detection in Degraded
  Scenarios." Advanced Intelligent Computing Technology and Applications,
  pp. 521-532. Springer Nature Singapore.

We apply mixed degradation augmentations at training time to improve
robustness under real-world distribution shifts. The augmentations are:

  - JPEG compression      (quality 70-90) — simulates social media recompression
  - Gaussian blur         (sigma 0.5-1.5) — simulates camera blur / resizing
  - Additive Gaussian noise (std 0.01-0.03) — simulates sensor noise
  - Recompression         (resize down + back up) — simulates repeated uploads
  - Mixed combinations    (randomly apply 2+ degradations together)

These are applied randomly at training time only. Test transforms are clean.

The gate in the fusion module should learn to DOWN-weight the frequency branch
for heavily degraded images (JPEG destroys spectral artifacts). The per-JPEG-
quality accuracy metric in utils/metrics.py verifies this is happening.

"""

import io
import random
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T



# Individual degradation transforms
class RandomJPEGCompression:
    """
    Random JPEG compression at quality in [min_q, max_q].
    Simulates social media recompression and real-world image sharing.
    The frequency branch gate should learn to contribute less for
    heavily compressed images where spectral artifacts are destroyed.
    """

    def __init__(self, min_quality: int = 70, max_quality: int = 90):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomGaussianBlur:
    """
    Random Gaussian blur with sigma in [min_sigma, max_sigma].
    Simulates camera blur, out-of-focus images, and lossy resizing.
    """

    def __init__(self, min_sigma: float = 0.5, max_sigma: float = 1.5):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class RandomAdditiveNoise:
    """
    Random additive Gaussian noise with std in [min_std, max_std].
    Simulates camera sensor noise and low-light conditions.
    """

    def __init__(self, min_std: float = 0.01, max_std: float = 0.03):
        self.min_std = min_std
        self.max_std = max_std

    def __call__(self, img: Image.Image) -> Image.Image:
        std = random.uniform(self.min_std, self.max_std)
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomRecompression:
    """
    Resize down then back up — simulates repeated upload/download cycles
    where images are rescaled by social media platforms.
    Scale factor in [min_scale, max_scale] — e.g. 0.5 means halved then doubled.
    """

    def __init__(self, min_scale: float = 0.5, max_scale: float = 0.75):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((small_w, small_h), Image.BILINEAR)
        return img.resize((w, h), Image.BILINEAR)


class RandomMixedDegradation:
    """
    Apply a random combination of 2 or more degradation types.
    Inspired by Cai et al. (2025)'s Random Degradation Simulator,
    which tests combined degradations rather than one at a time.
    Probability p controls how often mixed degradation is applied.
    """

    def __init__(self, p: float = 0.3):
        self.p = p
        self.degradations = [
            RandomJPEGCompression(70, 90),
            RandomGaussianBlur(0.5, 1.5),
            RandomAdditiveNoise(0.01, 0.03),
            RandomRecompression(0.5, 0.75),
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        # Pick 2-3 degradations at random and apply in sequence
        n = random.randint(2, 3)
        chosen = random.sample(self.degradations, n)
        for deg in chosen:
            img = deg(img)
        return img



# Transform builders

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(
    split: str,
    image_size: int = 32,
    jpeg_aug: bool = False,
    jpeg_quality_range: tuple = (70, 90),
    blur_aug: bool = False,
    noise_aug: bool = False,
    recompression_aug: bool = False,
    mixed_aug: bool = False,
    mixed_aug_prob: float = 0.3,
):
    """
    Return a transform pipeline for the given split and augmentation config.

    For the CIFAKE prototype:
        get_transforms("train", image_size=32, jpeg_aug=True)

    For the DeepDetect full-scale run with all augmentations:
        get_transforms("train", image_size=224, jpeg_aug=True, blur_aug=True,
                       noise_aug=True, recompression_aug=True, mixed_aug=True)

    Args:
        split:              "train" or "test"
        image_size:         32 for CIFAKE, 224 for DeepDetect
        jpeg_aug:           Random JPEG compression
        jpeg_quality_range: (min_q, max_q) for JPEG augmentation
        blur_aug:           Random Gaussian blur
        noise_aug:          Random additive Gaussian noise
        recompression_aug:  Random resize-down + resize-up
        mixed_aug:          Random combinations of 2-3 degradations (Cai et al. 2025)
        mixed_aug_prob:     Probability of applying mixed degradation per image
    """
    if split == "test":
        transforms = []
        if image_size != 32:
            transforms.append(T.Resize(image_size))
            transforms.append(T.CenterCrop(image_size))
        transforms += [
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        return T.Compose(transforms)

    # Training transforms
    # For small images (32x32), pad by 4px then crop — standard CIFAR practice.
    # For large images (224x224), use RandomResizedCrop which randomly crops and
    # resizes to the target size, providing scale/aspect-ratio jitter.
    if image_size <= 64:
        spatial_aug = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(image_size, padding=4),
        ]
    else:
        spatial_aug = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
        ]
    aug_list = spatial_aug

    # Individual degradation augmentations (applied before ToTensor, on PIL images)
    if jpeg_aug:
        aug_list.append(RandomJPEGCompression(*jpeg_quality_range))
    if blur_aug:
        aug_list.append(RandomGaussianBlur())
    if noise_aug:
        aug_list.append(RandomAdditiveNoise())
    if recompression_aug:
        aug_list.append(RandomRecompression())

    # Mixed degradation — applies on top of individual ones if both are enabled
    if mixed_aug:
        aug_list.append(RandomMixedDegradation(p=mixed_aug_prob))

    aug_list += [
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    return T.Compose(aug_list)