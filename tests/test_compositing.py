"""Tests for microVis.processing.compositing."""
from __future__ import annotations

import numpy as np

from microVis.processing.compositing import composite_image


def test_composite_single_channel():
    h, w = 10, 10
    img = np.random.rand(h, w, 1).astype(np.float64)
    names = ["ch1"]
    config = {"ch1": {"enabled": True, "color": "green", "vmin": 0, "vmax": 1}}
    result = composite_image(img, names, config)
    assert result.shape == (h, w, 3)
    assert result.dtype == np.uint8


def test_composite_two_channels():
    h, w = 10, 10
    img = np.random.rand(h, w, 2).astype(np.float64)
    names = ["ch1", "ch2"]
    config = {
        "ch1": {"enabled": True, "color": "green", "vmin": 0, "vmax": 1},
        "ch2": {"enabled": True, "color": "red", "vmin": 0, "vmax": 1},
    }
    result = composite_image(img, names, config)
    assert result.shape == (h, w, 3)


def test_composite_disabled_channel():
    h, w = 10, 10
    img = np.ones((h, w, 2), dtype=np.float64)
    names = ["ch1", "ch2"]
    config = {
        "ch1": {"enabled": False, "color": "green", "vmin": 0, "vmax": 1},
        "ch2": {"enabled": True, "color": "red", "vmin": 0, "vmax": 1},
    }
    result = composite_image(img, names, config)
    # ch1 disabled, so green channel should be 0
    assert result[:, :, 1].sum() == 0  # green channel


def test_composite_output_is_uint8():
    img = np.random.rand(8, 8, 1).astype(np.float64)
    result = composite_image(img, ["ch1"], {"ch1": {"enabled": True, "color": "blue", "vmin": 0, "vmax": 1}})
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
