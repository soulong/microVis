"""Tests for microVis._settings constants."""
from __future__ import annotations

from microVis._settings import (
    AGG_METHODS,
    CMAP_OPTIONS,
    DEFAULT_CMAP,
    DEFAULT_PLATE,
    DTYPE_MAX,
    PLATE_FORMATS,
    QUALITATIVE_PALETTES,
)


def test_plate_formats_non_empty():
    assert len(PLATE_FORMATS) > 0


def test_default_plate_in_formats():
    assert DEFAULT_PLATE in PLATE_FORMATS


def test_plate_format_values_are_tuples():
    for _name, dims in PLATE_FORMATS.items():
        assert isinstance(dims, tuple)
        assert len(dims) == 2
        assert dims[0] > 0 and dims[1] > 0



def test_dtype_max_covers_common_types():
    assert "uint8" in DTYPE_MAX
    assert "uint16" in DTYPE_MAX
    assert "float32" in DTYPE_MAX


def test_agg_methods_non_empty():
    assert len(AGG_METHODS) > 0
    assert "mean" in AGG_METHODS


def test_cmap_options_non_empty():
    assert len(CMAP_OPTIONS) > 0
    assert DEFAULT_CMAP in CMAP_OPTIONS


def test_qualitative_palettes_non_empty():
    assert len(QUALITATIVE_PALETTES) > 0
