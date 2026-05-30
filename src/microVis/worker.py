"""Background image processing worker for microVis."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal
from skimage.transform import resize as sk_resize

from microVis._settings import DTYPE_MAX
from microVis.log_utils import get_logger

_log = get_logger("microVis.worker")


def _downscale_image(img_data: np.ndarray, thumb_size: int) -> np.ndarray:
    """Downscale (H, W, C) array to fit within thumb_size."""
    h, w = img_data.shape[:2]
    if h <= thumb_size and w <= thumb_size:
        return img_data
    scale = thumb_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return sk_resize(
        img_data, (new_h, new_w, img_data.shape[2]),
        preserve_range=True, anti_aliasing=True,
    ).astype(img_data.dtype)


def _downscale_mask(mask_dict: dict, thumb_size: int) -> tuple:
    """Downscale first mask in dict. Returns (mask_name, small_mask) or (None, None)."""
    if not mask_dict:
        return None, None
    name = next(iter(mask_dict))
    first_mask = mask_dict[name]
    if first_mask is None:
        return None, None
    h, w = first_mask.shape
    if h <= thumb_size and w <= thumb_size:
        return name, first_mask
    scale = thumb_size / max(h, w)
    small = sk_resize(
        first_mask, (int(h * scale), int(w * scale)),
        order=0, preserve_range=True, anti_aliasing=False,
    ).astype(first_mask.dtype)
    return name, small


@dataclass
class ImageWorkerConfig:
    """Configuration for a single image processing worker."""

    row_idx: int
    well: str
    field: int
    stack: int
    tp: int
    raw_data: tuple
    thumb_size: int
    channel_names: list[str]
    ch_config: dict
    dmax: float
    contrast_method: str
    contrast_gamma: float
    invert: bool
    need_polygons: bool
    overlay_val: float | str | None = None
    overlay_col: str | None = None
    n_objects: int | None = None
    obj_values: dict = field(default_factory=dict)
    gen: int = 0
    sort_by_row: bool = False


class _WorkerSignals(QObject):
    finished = Signal(dict)
    error = Signal(str)


class ImageWorker(QRunnable):
    """Process a single image at thumbnail resolution in a background thread."""

    def __init__(self, config: ImageWorkerConfig):
        super().__init__()
        self.signals = _WorkerSignals()
        self._cfg = config
        self.setAutoDelete(False)

    @property
    def gen(self) -> int:
        return self._cfg.gen

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image
            from microVis.processing.contrast import apply_contrast, invert_image
            from microVis.processing.overlay import extract_polygons

            cfg = self._cfg
            img_data, mask_dict = cfg.raw_data

            # Downscale to thumbnail resolution
            img_small = _downscale_image(img_data, cfg.thumb_size)
            mask_name, mask_small = _downscale_mask(mask_dict, cfg.thumb_size)

            # Per-channel contrast on small image
            enhanced = np.zeros_like(img_small, dtype=np.float64)
            for ch_idx, ch_name in enumerate(cfg.channel_names):
                ch_cfg = cfg.ch_config.get(ch_name, {})
                if not ch_cfg.get("enabled", True):
                    enhanced[:, :, ch_idx] = img_small[:, :, ch_idx].astype(np.float64)
                    continue
                vmin = ch_cfg.get("vmin", 0)
                vmax = ch_cfg.get("vmax", cfg.dmax)
                band = img_small[:, :, ch_idx].astype(np.float64)
                band = np.clip((band - vmin) / max(vmax - vmin, 1e-10), 0, 1)
                if cfg.contrast_method == "gamma":
                    band = apply_contrast(band, "gamma", gamma=cfg.contrast_gamma)
                elif cfg.contrast_method == "histogram_equalization":
                    band = apply_contrast(band, "histogram_equalization")
                if cfg.invert:
                    band = invert_image(band)
                enhanced[:, :, ch_idx] = band

            # Composite
            comp_config = {ch: {**c, "vmin": 0, "vmax": 1}
                           for ch, c in cfg.ch_config.items()}
            rgb = composite_image(enhanced, cfg.channel_names, comp_config, None, None)

            # Polygons from downscaled mask
            polygons = None
            first_mask = None
            if cfg.need_polygons and mask_small is not None:
                first_mask = mask_small
                polygons = extract_polygons(first_mask)

            result = {
                "rgb": np.ascontiguousarray(rgb),
                "well": cfg.well, "field": cfg.field,
                "stack": cfg.stack, "tp": cfg.tp,
                "polygons": polygons,
                "overlay_val": cfg.overlay_val,
                "overlay_col": cfg.overlay_col,
                "n_objects": cfg.n_objects,
                "mask": first_mask,
                "obj_values": cfg.obj_values,
                "gen": cfg.gen,
                "thumb_size": cfg.thumb_size,
                "sort_by_row": cfg.sort_by_row,
            }
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
