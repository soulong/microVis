from __future__ import annotations

import numpy as np
from skimage import exposure


def apply_contrast(image: np.ndarray, method: str, **params) -> np.ndarray:
    """Apply contrast enhancement to a normalized [0,1] float image.

    Args:
        image: (H, W) float array in [0, 1].
        method: "none", "gamma", "histogram_equalization", "clahe".
        **params: Method-specific parameters.

    Returns:
        (H, W) float array in [0, 1].
    """
    if method == "none":
        return image

    if method == "gamma":
        gamma = float(params.get("gamma", 1.0))
        return image ** gamma

    if method == "histogram_equalization":
        return exposure.equalize_hist(image)

    if method == "clahe":
        kernel_size = int(params.get("kernel_size", 51))
        clip_limit = float(params.get("clip_limit", 0.01))
        img_uint = (image * 65535).astype(np.uint16)
        result = exposure.equalize_adapthist(img_uint, kernel_size=kernel_size,
                                              clip_limit=clip_limit)
        result = result.astype(np.float64)
        # Normalize output to [0,1] (scikit-image version-dependent behavior)
        if result.max() > 1.0:
            result /= 65535.0
        return result

    return image


def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert pixel values. Works on any dtype."""
    return image.max() - image + image.min()
