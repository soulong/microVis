"""Background image processing worker for microVis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


# ── Crop Worker ───────────────────────────────────────────────────────────────


class _CropSignals(QObject):
    finished = Signal(object, object)  # np.ndarray (RGB uint8), ObjectKey
    error = Signal(str)


class CropWorker(QRunnable):
    """Background worker that crops and masks a single object from an image."""

    def __init__(
        self,
        img_data: np.ndarray,
        mask: np.ndarray,
        label: int,
        key: Any,
        channel_names: list[str],
        ch_config: dict,
        dmax: float,
        contrast_method: str = "none",
        contrast_gamma: float = 1.0,
        invert: bool = False,
        target_size: int = 64,
        padding: int = 4,
    ):
        super().__init__()
        self.signals = _CropSignals()
        self.setAutoDelete(False)
        self._img = img_data
        self._mask = mask
        self._label = label
        self._key = key
        self._ch_names = channel_names
        self._ch_config = ch_config
        self._dmax = dmax
        self._contrast = contrast_method
        self._gamma = contrast_gamma
        self._invert = invert
        self._target = target_size
        self._pad = padding

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image
            from microVis.processing.contrast import apply_contrast, invert_image

            mask = self._mask
            label = self._label

            # Find bounding box of the object
            ys, xs = np.where(mask == label)
            if len(ys) == 0:
                self.signals.finished.emit(None, self._key)
                return

            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())

            # Add padding
            h, w = mask.shape
            y_min = max(0, y_min - self._pad)
            y_max = min(h, y_max + self._pad + 1)
            x_min = max(0, x_min - self._pad)
            x_max = min(w, x_max + self._pad + 1)

            # Crop image and mask
            crop_img = self._img[y_min:y_max, x_min:x_max, :].astype(np.float64)
            crop_mask = mask[y_min:y_max, x_min:x_max]

            # Apply mask: zero out pixels not belonging to this object
            obj_mask = (crop_mask == label).astype(np.float64)
            for ch in range(crop_img.shape[2]):
                crop_img[:, :, ch] *= obj_mask

            # Per-channel contrast
            enhanced = np.zeros_like(crop_img, dtype=np.float64)
            for ch_idx, ch_name in enumerate(self._ch_names):
                if ch_idx >= crop_img.shape[2]:
                    break
                ch_cfg = self._ch_config.get(ch_name, {})
                if not ch_cfg.get("enabled", True):
                    enhanced[:, :, ch_idx] = crop_img[:, :, ch_idx]
                    continue
                vmin = ch_cfg.get("vmin", 0)
                vmax = ch_cfg.get("vmax", self._dmax)
                band = crop_img[:, :, ch_idx]
                band = np.clip((band - vmin) / max(vmax - vmin, 1e-10), 0, 1)
                if self._contrast == "gamma":
                    band = apply_contrast(band, "gamma", gamma=self._gamma)
                elif self._contrast == "histogram_equalization":
                    band = apply_contrast(band, "histogram_equalization")
                if self._invert:
                    band = invert_image(band)
                enhanced[:, :, ch_idx] = band

            # Composite
            comp_config = {
                ch: {**c, "vmin": 0, "vmax": 1}
                for ch, c in self._ch_config.items()
            }
            rgb = composite_image(enhanced, self._ch_names, comp_config, None, None)

            # Resize to target
            ch, cw = rgb.shape[:2]
            if ch > self._target or cw > self._target:
                from skimage.transform import resize as sk_resize
                scale = self._target / max(ch, cw)
                rgb = sk_resize(
                    rgb,
                    (int(ch * scale), int(cw * scale), 3),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.uint8)

            # Return as numpy array (QPixmap must be created on main thread)
            rgb = np.ascontiguousarray(rgb)

            self.signals.finished.emit(rgb, self._key)
        except Exception as e:
            self.signals.error.emit(str(e))


# -- Object Export Worker --


class _ExportSignals(QObject):
    progress = Signal(int, int)  # (current, total)
    finished = Signal(dict)  # {"count": N, "save_dir": path}
    error = Signal(str)


class ObjectExportWorker(QRunnable):
    """Background worker that exports cropped/masked objects to disk."""

    def __init__(
        self,
        dm: Any,
        wells: list[str],
        fields: list[int],
        stacks: list[int],
        timepoints: list[int],
        mask_name: str,
        channel_names: list[str],
        save_dir: str,
        object_mode: str,
        channel_mode: str,
        annotations: dict | None = None,
        gen: int = 0,
    ):
        super().__init__()
        self.signals = _ExportSignals()
        self.setAutoDelete(False)
        self._dm = dm
        self._wells = wells
        self._fields = fields
        self._stacks = stacks
        self._tps = timepoints
        self._mask_name = mask_name
        self._channel_names = channel_names
        self._save_dir = save_dir
        self._object_mode = object_mode
        self._channel_mode = channel_mode
        self._annotations = annotations
        self._gen = gen

    def run(self) -> None:
        try:
            from pathlib import Path
            import tifffile

            save_path = Path(self._save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            rows = self._dm.lookup_row_indices(
                self._wells, self._fields, self._stacks, self._tps
            )
            if not rows:
                self.signals.finished.emit({"count": 0, "save_dir": self._save_dir})
                return

            # Build annotation lookup if needed
            key_to_class = {}
            if self._annotations and self._object_mode == "All annotated":
                for cls_name, keys in self._annotations.items():
                    for key in keys:
                        lookup = (key.well, key.field, key.stack, key.tp, key.label)
                        key_to_class[lookup] = cls_name

            total = len(rows)
            exported_count = 0
            mask_col = f"mask_{self._mask_name}"
            _log.info("Export starting: %d images, mask=%s, mode=%s, channel_mode=%s",
                      total, mask_col, self._object_mode, self._channel_mode)

            for img_idx, (row_idx, well, field, stack, tp) in enumerate(rows):
                try:
                    img_data, mask_dict = self._dm.get_imageset_with_masks(
                        row_idx, channels=self._channel_names, masks=[mask_col]
                    )
                    _log.debug("Row %d: mask_dict keys=%s", row_idx, list(mask_dict.keys()))
                    # Try with prefix first, then without
                    mask = mask_dict.get(mask_col)
                    if mask is None:
                        mask = mask_dict.get(self._mask_name)
                    if mask is None and mask_dict:
                        # Fall back to first available mask
                        mask = next(iter(mask_dict.values()))
                    if mask is None:
                        _log.debug("Row %d: no mask found, skipping", row_idx)
                        continue

                    # Find image name from metadata
                    img_name = f"{well}_f{field}_z{stack}_t{tp}"

                    # Get unique labels in this image
                    unique_labels = np.unique(mask)
                    unique_labels = unique_labels[unique_labels > 0]
                    _log.info("Row %d (%s): mask shape=%s, dtype=%s, unique_labels=%d",
                              row_idx, img_name, mask.shape, mask.dtype, len(unique_labels))

                    for label_id in unique_labels:
                        # Check if we should export this object
                        lookup = (well, field, stack, tp, int(label_id))
                        if self._object_mode == "All annotated":
                            if lookup not in key_to_class:
                                _log.debug("Skipping object %d (not in annotations)", label_id)
                                continue

                        # Extract object crop
                        ys, xs = np.where(mask == label_id)
                        if len(ys) == 0:
                            continue

                        y_min, y_max = int(ys.min()), int(ys.max())
                        x_min, x_max = int(xs.min()), int(xs.max())

                        # Add padding
                        h, w = mask.shape
                        pad = 4
                        y_min_p = max(0, y_min - pad)
                        y_max_p = min(h, y_max + pad + 1)
                        x_min_p = max(0, x_min - pad)
                        x_max_p = min(w, x_max + pad + 1)

                        # Crop image (raw intensity values)
                        crop_img = img_data[y_min_p:y_max_p, x_min_p:x_max_p, :]
                        crop_mask = mask[y_min_p:y_max_p, x_min_p:x_max_p]

                        # Apply mask (zero background, keep raw values for object)
                        obj_mask = (crop_mask == label_id)
                        for ch in range(crop_img.shape[2]):
                            crop_img[:, :, ch] *= obj_mask

                        # Save based on channel mode
                        if self._channel_mode == "Single channel":
                            # Save each channel separately
                            for ch_idx, ch_name in enumerate(self._channel_names):
                                if ch_idx >= crop_img.shape[2]:
                                    break
                                ch_data = crop_img[:, :, ch_idx]
                                fname = f"{img_name}_{label_id}.tiff"
                                # Add channel name to filename if multiple channels
                                if len(self._channel_names) > 1:
                                    fname = f"{img_name}_{label_id}_{ch_name}.tiff"
                                tifffile.imwrite(str(save_path / fname), ch_data)
                        else:
                            # Save as multi-channel image (CYX format)
                            # crop_img is (H, W, C) -> transpose to (C, H, W)
                            multi_ch = np.moveaxis(crop_img, -1, 0)
                            fname = f"{img_name}_{label_id}.tiff"
                            tifffile.imwrite(str(save_path / fname), multi_ch)

                        exported_count += 1

                except Exception as e:
                    _log.warning("Failed to export from row %d: %s", row_idx, e)

                self.signals.progress.emit(img_idx + 1, total)

            _log.info("Export complete: %d objects exported to %s", exported_count, self._save_dir)
            self.signals.finished.emit({
                "count": exported_count,
                "save_dir": self._save_dir,
            })

        except Exception as e:
            _log.exception("Object export failed")
            self.signals.error.emit(str(e))
