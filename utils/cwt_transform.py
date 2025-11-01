import numpy as np
import pywt
from scipy.ndimage import zoom


def apply_cwt(signal, fs=250, wavelet='morl', freq_range=(8, 30), n_scales=64):
    """
    Continuous Wavelet Transform using Morlet wavelet
    Returns time-frequency representation (scalogram)
    """
    frequencies = np.linspace(freq_range[0], freq_range[1], n_scales)
    scales = pywt.frequency2scale(wavelet, frequencies / fs)
    coefficients, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    return np.abs(coefficients)


def scalogram_to_rgb(scalogram, target_size=(224, 224)):
    """Convert scalogram to RGB image for CNN"""
    # Normalize to [0, 1]
    scalogram_norm = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8)
    
    # Resize
    zoom_factors = (target_size[0] / scalogram_norm.shape[0],
                   target_size[1] / scalogram_norm.shape[1])
    resized = zoom(scalogram_norm, zoom_factors, order=1)
    
    # Convert to RGB (C, H, W) for PyTorch
    rgb_image = np.stack([resized, resized, resized], axis=0)
    return rgb_image.astype(np.float32)
