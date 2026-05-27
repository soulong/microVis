"""Tests for microVis.processing.overlay."""
from __future__ import annotations

import numpy as np

from microVis.processing.overlay import extract_polygons


def test_extract_polygons_empty_mask():
    mask = np.zeros((10, 10), dtype=np.int32)
    result = extract_polygons(mask)
    assert result == []


def test_extract_polygons_single_label():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:15, 5:15] = 1  # square region
    result = extract_polygons(mask)
    assert len(result) >= 1
    assert result[0][0] == 1


def test_extract_polygons_multiple_labels():
    mask = np.zeros((30, 30), dtype=np.int32)
    mask[2:8, 2:8] = 1
    mask[20:28, 20:28] = 2
    result = extract_polygons(mask)
    labels = {lbl for lbl, _ in result}
    assert 1 in labels
    assert 2 in labels


def test_extract_polygons_min_area_filter():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[0, 0] = 1  # tiny region (1 pixel)
    result = extract_polygons(mask, min_area=5)
    assert len(result) == 0


def test_extract_polygons_contour_shape():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:15, 5:15] = 1
    result = extract_polygons(mask)
    assert len(result) >= 1
    _, contour = result[0]
    assert contour.ndim == 2
    assert contour.shape[1] == 2  # (N, 2) coordinates
