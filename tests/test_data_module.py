"""Tests for microVis.io.data_module (pure functions only)."""
from __future__ import annotations

from microVis.io.data_module import _infer_plate_dims


def test_infer_plate_dims_96well():
    wells = [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]
    rows, cols = _infer_plate_dims(wells)
    assert rows == 8
    assert cols == 12


def test_infer_plate_dims_24well():
    wells = [f"{r}{c}" for r in "ABCD" for c in range(1, 7)]
    rows, cols = _infer_plate_dims(wells)
    assert rows == 4
    assert cols == 6


def test_infer_plate_dims_single_well():
    wells = ["B2"]
    rows, cols = _infer_plate_dims(wells)
    assert rows == 2
    assert cols == 2


def test_infer_plate_dims_empty():
    rows, cols = _infer_plate_dims([])
    assert rows == 0
    assert cols == 0
