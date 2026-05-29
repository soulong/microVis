# microVis

[![Release](https://img.shields.io/github/v/release/soulong/microVis)](https://github.com/soulong/microVis/releases)
[![Last Commit](https://img.shields.io/github/last-commit/soulong/microVis)](https://github.com/soulong/microVis/commits/main)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/soulong/microVis)](LICENSE)

Interactive desktop GUI for visualizing microProfiler microscopy datasets — plate grids, multi-channel images, and profiling data.

## Requirements

- **Python** >= 3.10
- **OS**: Only Windows 11 (64-bit) was tested

## Installation

### Quick Install

```bash
git clone https://github.com/soulong/microVis.git
cd microVis
pip install .
```

## Quick Start

```bash
# Launch with folder selector
microvis

# Launch directly with a dataset
microvis "/path/to/Measurement 1"
```

### Windows Desktop Shortcut

After installing in the `micro` conda environment, create a desktop shortcut that launches microVis without opening a console window:

```bash
microvis-install-shortcut
```

This places a shortcut on the Desktop and in the Start Menu. The shortcut uses `pythonw.exe` from the `micro` environment for a clean, console-free launch.

## Features

### Well Plate Grid

Interactive scatter-plot grid visualizing well plates (24-, 96-, or 384-well). Color wells by any profiling column or merged metadata, with natural sorting, colormaps, and click-to-select.

### Multi-Channel Image Compositing

Composite multi-channel microscopy images with per-channel controls:

- Enable/disable individual channels
- Assign display colors (green, red, magenta, blue, cyan, yellow, white)
- Adjust vmin/vmax per channel or use auto-range (percentile-based, per-channel)
- Apply contrast transforms: gamma correction, histogram equalization, invert

### Object Overlay

Overlay segmentation masks and per-object profiling data onto images:

- Boundary contour visualization with color-coded fills
- Per-object tooltips showing label ID and profiling values
- Configurable colormap and alpha

### Metadata Integration

Import plate-shaped Excel metadata files, merge with profiling data, and use metadata columns for grid coloring or object overlay.

### Profiling Data Browser

Browse profiling tables from `results.db` with:

- Sortable columns (natural sort for numeric values)
- Radio-button table selector
- PyGwalker integration for interactive data exploration

### Pixel Inspector

Click any image thumbnail to see per-channel pixel intensities at that coordinate.

## Acknowledgements

microVis is a sibling package to [microProfiler](https://github.com/soulong/microProfiler) — a microscopy image preprocessing, segmentation, and profiling pipeline.

## Citation

If you use microVis in your research, please cite the repository:

```
@software{microVis,
  author = {Hao He},
  title = {microVis: Interactive Desktop GUI for Microscopy Dataset Visualization},
  year = {2026},
  url = {https://github.com/soulong/microVis}
}
```

## License

MIT
