from __future__ import annotations

import numpy as np
from skimage import measure


def extract_polygons(mask: np.ndarray, min_area: int = 5) -> list[tuple[int, np.ndarray]]:
    """Extract boundary polygons from a label mask.

    Returns:
        List of (label_value, contour_xy) tuples where contour_xy is (N, 2) in (row, col) order.
    """
    labels = np.unique(mask)
    labels = labels[labels > 0]
    polygons: list[tuple[int, np.ndarray]] = []
    for lbl in labels:
        binary = mask == lbl
        if np.sum(binary) < min_area:
            continue
        contours = measure.find_contours(binary, level=0.5)
        for contour in contours:
            polygons.append((int(lbl), contour))
    return polygons
