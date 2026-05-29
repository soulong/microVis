"""Tests for microVis.processing.contrast."""
from __future__ import annotations

import numpy as np

from microVis.processing.contrast import apply_contrast, invert_image


def test_apply_contrast_none():
    img = np.random.rand(10, 10)
    result = apply_contrast(img, "none")
    np.testing.assert_array_equal(result, img)


def test_apply_contrast_gamma():
    img = np.random.rand(10, 10)
    result = apply_contrast(img, "gamma", gamma=2.0)
    assert result.shape == img.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_apply_contrast_histogram_equalization():
    img = np.random.rand(10, 10)
    result = apply_contrast(img, "histogram_equalization")
    assert result.shape == img.shape


def test_invert_image_float():
    img = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = invert_image(img)
    np.testing.assert_allclose(result, [1.0, 0.75, 0.5, 0.25, 0.0])


def test_invert_image_uint16():
    img = np.array([0, 100, 32768, 65535], dtype=np.uint16)
    result = invert_image(img)
    assert result[0] == 65535
    assert result[3] == 0
