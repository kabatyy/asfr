"""
All hyperparameters live here. Pass a Config instance to train/evaluate.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    dataset: Literal["cifake", "deepdetect"] = "cifake"
    cifake_root: str = "./data/raw/cifake"
    deepdetect_root: str = "./data/raw/deepdetect"
    image_size: int = 32
    batch_size: int = 64
    num_workers: int = 4
    # Training augmentations — all False by default, enable per experiment
    jpeg_aug: bool = False               # JPEG compression (quality 70-90)
    blur_aug: bool = False              # Gaussian blur (sigma 0.5-1.5)
    noise_aug: bool = False             # additive Gaussian noise (std 0.01-0.03)
    recompression_aug: bool = False     # resize down and back up (scale 0.5-0.75)
    mixed_aug: bool = False             # random combinations of 2-3 degradations
    mixed_aug_prob: float = 0.3         # probability of applying mixed degradation
    jpeg_aug_quality_range: tuple = (70, 90)

    @property
    def data_root(self) -> str:
        """Return the correct root for the active dataset."""
        return self.cifake_root if self.dataset == "cifake" else self.deepdetect_root


@dataclass
class FrequencyConfig:
    patch_size: int = 56           # patch extracted for freq analysis (at 224px; scale if smaller)
    use_fftshift: bool = True      # CRITICAL: always True — moves DC to centre
    log_scale: bool = True         # log-magnitude of FFT spectrum
    srm_filters: bool = True       # prepend fixed SRM noise-residual filters
    cleaner_filters: int = 3       # number of conv filters in degradation-aware cleaner


@dataclass
class FusionConfig:
    mode: Literal["scalar", "gating", "joint_only"] = "gating"
    # scalar mode
    scalar_softmax: bool = True    # CRITICAL: softmax(a,b) so they sum to 1, prevents b->0 collapse
    # gating mode
    gate_hidden_dim: int = 64
    gate_init_bias: float = 0.1    # slight freq-branch head-start before backbone dominates
    # diversity regulariser for gating (entropy penalty)
    diversity_weight: float = 0.1  # weight on -Entropy(gate values) term


@dataclass
class LossConfig:
    # Auxiliary heads (CRITICAL — prevents gradient starvation of freq branch)
    use_auxiliary_heads: bool = True
    spatial_aux_weight: float = 0.3
    freq_aux_weight: float = 0.5       # intentionally higher — weaker initial gradient
    # Cleaner reconstruction loss (real images only, joint training)
    cleaner_recon_weight: float = 0.1  # weight on MAE reconstruction loss for cleaner


@dataclass
class BackboneConfig:
    name: Literal[
        "convnext_base",
        "dino_vits8",
        "swin_v2_s",
        "vit_b_16",
        "vit_b_32",
    ] = "convnext_base"
    pretrained: bool = True
    frozen: bool = False
    img_size: int = 32 # To match timm            


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    log_scalar_every_n_epochs: int = 1
    log_grad_norm_every_n_epochs: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: str = "asfr_default"
    notes: str = ""