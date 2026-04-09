# Adaptive Spatial-Frequency Reasoning (ASFR)

## What this project does

This repository implements a dual-branch neural network for detecting AI-generated images. Given a photo, the model decides whether it was taken by a real camera or produced by a generative AI system such as Midjourney, Stable Diffusion, or DALL-E.

The core idea is that AI-generated images leave two kinds of evidence: visible evidence in the pixels (objects that look slightly off, unnatural textures) and invisible evidence in the image's frequency spectrum (mathematical fingerprints left by the generation process). This project combines both signals through a learnable fusion mechanism that adapts per image. 

The three fusion strategies we benchmark, in order of complexity:

- **Joint-only** — both branches feed a single shared classifier (baseline)
- **Scalar fusion** — two learned numbers `a` and `b` weight the branches globally
- **Gating fusion** — a small MLP outputs a per-image weight, enabling sample-level adaptation (main contribution)

---

## New to this project?

Read the [Guide](docs/Guide.docx). It explains the repository structure, the experiment sequence, what to measure, and the failure modes to watch for.

---

## Requirements

- Python 3.10+
- PyTorch 2.x

```bash
pip install -r requirements.txt
```
---

## Datasets

Download both datasets from Kaggle before running any experiments.

| Dataset | Purpose | Resolution | Link |
|---------|---------|-----------|------|
| CIFAKE | Prototyping — fast iteration | 32×32 | [kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| DeepDetect-2025 | Final training and evaluation | 224×224 | [kaggle.com/datasets/ayushmandatta1/deepdetect-2025](https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025/data) |

Place downloaded data in `data/raw/cifake/` and `data/raw/deepdetect/` respectively.

> **Note:** Do not resize CIFAKE images to 224×224 during prototyping. The point of CIFAKE is that it is small and fast and resizing loses that advantage.

## Repository structure

```
asfr/
├── config.py                   # All hyperparameters 
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
│
├── data/
│   ├── cifake.py               # CIFAKE dataset loader (32×32)
│   ├── deepdetect.py           # DeepDetect-2025 loader (224×224)
│   └── transforms.py           # Augmentations 
│
├── models/
│   ├── spatial_branch.py       # Pretrained backbone + projection head
│   ├── frequency_branch.py     # Cleaner → SRM filters → FFT → CNN
│   ├── cleaner.py              # Degradation-aware noise cleaner (3 filters max)
│   ├── fusion.py               # Fusion modes: scalar / gating / joint_only
│   └── full_model.py           # Assembles both branches into one model
│
├── losses/
│   ├── auxiliary.py            # Joint loss + spatial aux (×0.3) + freq aux (×0.5)
│   └── diversity.py            # Gate entropy regulariser
│
├── utils/
│   ├── fft_utils.py            # Log-magnitude FFT with fftshift
│   ├── patch_select.py         # Lowest-variance patch extraction
│   ├── diagnostics.py          # Gradient norms, scalar logging, warning sign checks
│   └── metrics.py              # Accuracy, gate stats, per-generator, per-JPEG
│
├── experiments/
│   ├── baseline_freq_only.py   # Frequency-only standalone (run FIRST)
│   ├── train.py                # Training loop
│   └── evaluate.py             # Full evaluation suite
│
└── docs/
    └── Guide.pdf
```
---