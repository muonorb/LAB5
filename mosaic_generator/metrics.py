import numpy as np
from skimage.metrics import structural_similarity as _ssim


def compute_mse(original: np.ndarray, mosaic: np.ndarray) -> float:
    """Returns the average pixel difference between two images. Lower = more similar."""
    orig = original.astype(np.float64)
    mos = mosaic.astype(np.float64)
    return float(np.mean((orig - mos) ** 2))


def compute_ssim(original: np.ndarray, mosaic: np.ndarray) -> float:
    """Returns how visually similar two images look. 1.0 = identical."""
    score = _ssim(
        original,
        mosaic,
        channel_axis=2,
        data_range=255,
    )
    return float(score)


def compute_all_metrics(original: np.ndarray, mosaic: np.ndarray) -> dict:
    """Returns both MSE and SSIM scores in one dict."""
    return {
        "mse": compute_mse(original, mosaic),
        "ssim": compute_ssim(original, mosaic),
    }
