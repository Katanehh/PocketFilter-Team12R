"""Signal preprocessing: sliding median filter and z-score normalization."""

import numpy as np


def median_filter(signal: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove isolated noise spikes with an odd-length sliding median filter."""
    arr = np.asarray(signal, dtype=np.float64)
    if kernel_size <= 1:
        return arr.copy()
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer.")
    pad = kernel_size // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=kernel_size)
    return np.median(windows, axis=-1)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to zero mean and unit variance."""
    arr = np.asarray(signal, dtype=np.float64)
    std = float(np.std(arr))
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def preprocess_signal(signal: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filtering followed by z-score normalization."""
    cleaned = median_filter(signal, kernel_size=kernel_size)
    return zscore_normalize(cleaned)
