# microVis

[![PyPI version](https://img.shields.io/pypi/v/microVis)](https://pypi.org/project/microVis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/pypi/pyversions/microVis)

Interactive desktop GUI for visualizing microProfiler microscopy datasets — plate grids, multi-channel images, and profiling data.

## Requirements

- **Python** >= 3.10
- **OS**: Windows 10/11 (64-bit) recommended

## Installation

### Quick Install

```bash
pip install microVis
```

### Conda Install

```bash
conda env create -f micro.yml
conda activate micro
```

### Development Install

```bash
git clone https://github.com/soulong/microVis.git
cd microVis
pip install -e ".[dev]"
```

## Quick Start

```bash
# Launch with folder selector
microvis

# Launch directly with a dataset
microvis "/path/to/Measurement 1"
```

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

## Key Modules

| Module | Purpose |
|--------|---------|
| `io.data_module` | Data access layer wrapping microProfiler ImageDataset + SQLite results.db |
| `processing.compositing` | Multi-channel image compositing with mask overlay |
| `processing.contrast` | Contrast enhancement (gamma, histogram equalization) |
| `processing.overlay` | Segmentation mask boundary extraction |
| `widgets.well_grid_canvas` | Matplotlib-based well plate grid visualization |
| `widgets.image_display` | Thumbnail grid with pan/zoom/overlay |
| `widgets.image_controls` | Channel settings, contrast, overlay controls |
| `widgets.data_view` | Profiling table browser with PyGwalker |
| `widgets.channel_controls` | Per-channel checkbox, color, vmin/vmax |
| `widgets.well_grid_controls` | Plate format, color-by, aggregation controls |

## System Requirements

- **OS**: Windows 10/11 (64-bit) — GUI is Windows-optimized
- **RAM**: 16 GB minimum (32 GB recommended for large datasets)

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
