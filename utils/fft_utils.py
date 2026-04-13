import torch
import numpy as np
 
 
def compute_log_fft(image: np.ndarray, fftshift: bool = True) -> np.ndarray:
    """
    Compute the log-magnitude 2D FFT spectrum of a single-channel image patch.
    For inspection and debugging. For training, use fft_spectrum_tensor.
 
    Args:
        image:    2D numpy array (H, W), single channel, float in [0, 1]
        fftshift: Always True - moves DC component to centre.
 
    Returns:
        2D log-magnitude spectrum, same shape as input, values in [0, ~log(max)].
    """
    spectrum = np.fft.fft2(image)
    if fftshift:
        spectrum = np.fft.fftshift(spectrum)
    return np.log1p(np.abs(spectrum))
 
 
def fft_spectrum_tensor(image_tensor: torch.Tensor, fftshift: bool = True) -> torch.Tensor:
    """
    Compute log-magnitude FFT spectrum for a batch of images.
    This is the main function called by the frequency branch during training.
 
    Args:
        image_tensor: (B, C, H, W) float tensor, values in [0, 1]
        fftshift:     Always True — moves DC component to centre of spectrum.
 
    Returns:
        (B, C, H, W) log-magnitude spectrum tensor, per-sample normalised
        to zero mean and unit std so the CNN receives consistent input scale.
    """
    # Cast to float32 before FFT — cuFFT in half precision (float16) only supports
    # power-of-two dimensions. Casting ensures compatibility with mixed precision training
    # regardless of patch size. The cast is cheap relative to the FFT itself.
    spectrum = torch.fft.fft2(image_tensor.float())

    if fftshift:
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

    # log1p(|spectrum|): magnitude first, then log-compress
    log_magnitude = torch.log1p(torch.abs(spectrum))

    return normalise_spectrum(log_magnitude)


def normalise_spectrum(spectrum: torch.Tensor) -> torch.Tensor:
    """
    Normalise a spectrum tensor to zero mean and unit std per sample.
    Operates over the H and W dimensions so each image in the batch is
    normalised independently. This prevents batch statistics from leaking
    between samples during training.
 
    Args:
        spectrum: (B, C, H, W) log-magnitude spectrum
 
    Returns:
        (B, C, H, W) normalised spectrum
    """
    # Compute mean and std over H and W for each (B, C) slice
    mean = spectrum.mean(dim=(-2, -1), keepdim=True)
    std  = spectrum.std(dim=(-2, -1), keepdim=True)
    return (spectrum - mean) / (std + 1e-8)  