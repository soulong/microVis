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


def _enhance_channels(
    img_data: np.ndarray,
    channel_names: list[str],
    ch_config: dict,
    dmax: float,
    contrast_method: str,
    contrast_gamma: float,
    invert: bool,
) -> np.ndarray:
    """Apply per-channel contrast enhancement to an (H, W, C) image.

    Returns a float64 array in [0, 1].
    """
    from microVis.processing.contrast import apply_contrast, invert_image

    enhanced = np.zeros_like(img_data, dtype=np.float64)
    n_channels = img_data.shape[2]
    for ch_idx, ch_name in enumerate(channel_names):
        if ch_idx >= n_channels:
            break
        ch_cfg = ch_config.get(ch_name, {})
        if not ch_cfg.get("enabled", True):
            enhanced[:, :, ch_idx] = 0.0
            continue
        vmin = ch_cfg.get("vmin", 0)
        vmax = ch_cfg.get("vmax", dmax)
        band = img_data[:, :, ch_idx].astype(np.float64)
        band = np.clip((band - vmin) / max(vmax - vmin, 1e-10), 0, 1)
        if contrast_method == "gamma":
            band = apply_contrast(band, "gamma", gamma=contrast_gamma)
        elif contrast_method == "histogram_equalization":
            band = apply_contrast(band, "histogram_equalization")
        if invert:
            band = invert_image(band)
        enhanced[:, :, ch_idx] = band
    return enhanced


@dataclass
class ImageWorkerConfig:
    """Configuration for a single image processing worker."""

    row_idx: int
    well: str
    field: int
    stack: int
    tp: int
    raw_data: tuple | None
    thumb_size: int
    channel_names: list[str]
    ch_config: dict
    dmax: float
    contrast_method: str
    contrast_gamma: float
    invert: bool
    need_polygons: bool
    dm: object | None = None
    overlay_val: float | str | None = None
    overlay_col: str | None = None
    n_objects: int | None = None
    obj_values: dict = field(default_factory=dict)
    overlay_vmin: float = 0.0
    overlay_vmax: float = 1.0
    gen: int = 0
    sort_by_row: bool = False
    mask_cache: np.ndarray | None = None
    polygons_cache: list | None = None


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
            from microVis.processing.overlay import extract_polygons

            cfg = self._cfg

            # Load from disk if not provided (runs on worker thread, not main)
            loaded_from_disk = False
            if cfg.raw_data is not None:
                img_data, mask_dict = cfg.raw_data
            else:
                img_data, mask_dict = cfg.dm.get_imageset(cfg.row_idx)
                loaded_from_disk = True

            # Downscale to thumbnail resolution
            img_small = _downscale_image(img_data, cfg.thumb_size)
            if cfg.mask_cache is not None:
                mask_small = cfg.mask_cache
                mask_name = None
            else:
                mask_name, mask_small = _downscale_mask(mask_dict, cfg.thumb_size)

            # Per-channel contrast enhancement
            enhanced = _enhance_channels(
                img_small, cfg.channel_names, cfg.ch_config, cfg.dmax,
                cfg.contrast_method, cfg.contrast_gamma, cfg.invert,
            )

            # Composite
            comp_config = {ch: {**c, "vmin": 0, "vmax": 1}
                           for ch, c in cfg.ch_config.items()}
            rgb = composite_image(enhanced, cfg.channel_names, comp_config, None, None)

            # Polygons from downscaled mask — use cache when available
            polygons = None
            first_mask = mask_small if cfg.need_polygons else None
            if cfg.polygons_cache is not None:
                polygons = cfg.polygons_cache
            elif cfg.need_polygons and mask_small is not None:
                polygons = extract_polygons(mask_small)

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
                "overlay_vmin": cfg.overlay_vmin,
                "overlay_vmax": cfg.overlay_vmax,
                "gen": cfg.gen,
                "thumb_size": cfg.thumb_size,
                "sort_by_row": cfg.sort_by_row,
                "row_idx": cfg.row_idx,
            }
            if loaded_from_disk:
                result["raw_data"] = (img_data, mask_dict)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class _FullResSignals(QObject):
    finished = Signal(object, str, int, int, int, int, object, object, str, str, float, float)
    # QPixmap, well, field, stack, tp, gen, mask, obj_values, overlay_col, overlay_cmap, overlay_vmin, overlay_vmax
    error = Signal(str)


class FullResWorker(QRunnable):
    """Load and composite a single image at full resolution."""

    def __init__(self, dm, row_idx: int, well: str, field: int, stack: int, tp: int,
                 channel_names: list[str], ch_config: dict, dmax: float,
                 contrast_method: str, contrast_gamma: float, invert: bool,
                 gen: int = 0, overlay_alpha: float = 0.4, overlay_cmap: str = "Viridis",
                 need_polygons: bool = True, obj_values: dict | None = None,
                 overlay_col: str | None = None,
                 overlay_vmin: float = 0.0, overlay_vmax: float = 1.0):
        super().__init__()
        self.signals = _FullResSignals()
        self.setAutoDelete(False)
        self._dm = dm
        self._row_idx = row_idx
        self._well = well
        self._field = field
        self._stack = stack
        self._tp = tp
        self._ch_names = channel_names
        self._ch_config = ch_config
        self._dmax = dmax
        self._contrast = contrast_method
        self._gamma = contrast_gamma
        self._invert = invert
        self._gen = gen
        self._overlay_alpha = overlay_alpha
        self._overlay_cmap = overlay_cmap
        self._need_polygons = need_polygons
        self._obj_values = obj_values or {}
        self._overlay_col = overlay_col
        self._overlay_vmin = overlay_vmin
        self._overlay_vmax = overlay_vmax

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image

            img_data, mask_dict = self._dm.get_imageset(self._row_idx)

            enhanced = _enhance_channels(
                img_data, self._ch_names, self._ch_config, self._dmax,
                self._contrast, self._gamma, self._invert,
            )

            comp_config = {ch: {**c, "vmin": 0, "vmax": 1}
                           for ch, c in self._ch_config.items()}
            rgb = composite_image(enhanced, self._ch_names, comp_config, None, None)
            rgb = np.ascontiguousarray(rgb)

            h, w, _ = rgb.shape
            from PySide6.QtGui import QImage, QPixmap
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())

            # Extract mask for interactivity (hover/drag) and optional overlay
            full_mask = None
            if self._need_polygons and mask_dict:
                first_key = next(iter(mask_dict))
                full_mask = mask_dict[first_key]
                if full_mask is not None:
                    from microVis.processing.overlay import extract_polygons
                    from microVis.widgets.image_display import _draw_polygon_overlays
                    polygons = extract_polygons(full_mask)
                    if polygons:
                        pixmap = _draw_polygon_overlays(
                            pixmap, polygons, self._overlay_alpha, self._overlay_cmap,
                            self._obj_values, self._overlay_vmin, self._overlay_vmax)

            self.signals.finished.emit(
                pixmap, self._well, self._field, self._stack, self._tp,
                self._gen, full_mask, self._obj_values,
                self._overlay_col or "", self._overlay_cmap,
                self._overlay_vmin, self._overlay_vmax)
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
            enhanced = _enhance_channels(
                crop_img, self._ch_names, self._ch_config, self._dmax,
                self._contrast, self._gamma, self._invert,
            )

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
        annotated_keys: set | None = None,
        gen: int = 0,
        max_objects_per_image: int = 0,
        well_subdir: bool = False,
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
        self._max_obj = max_objects_per_image
        self._annotations = annotations
        self._annotated_keys = annotated_keys
        self._gen = gen
        self._well_subdir = well_subdir

    def run(self) -> None:
        try:
            from pathlib import Path
            import tifffile

            save_path = Path(self._save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            rows = self._dm.lookup_row_indices(
                self._wells, self._fields, self._stacks, self._tps
            )
            # Filter to only annotated images when applicable
            if self._annotated_keys is not None:
                rows = [r for r in rows if (r[1], r[2], r[3], r[4]) in self._annotated_keys]
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

                    # Filter to eligible labels (annotation check)
                    eligible_labels = unique_labels
                    if self._object_mode == "All annotated":
                        eligible_labels = np.array([
                            lbl for lbl in unique_labels
                            if (well, field, stack, tp, int(lbl)) in key_to_class
                        ])
                        if len(eligible_labels) == 0:
                            continue

                    # Random sample if max objects is set
                    if self._max_obj > 0 and len(eligible_labels) > self._max_obj:
                        eligible_labels = np.random.default_rng().choice(
                            eligible_labels, size=self._max_obj, replace=False,
                        )

                    for label_id in eligible_labels:
                        lookup = (well, field, stack, tp, int(label_id))

                        # Determine save directory (class subfolder for annotated, well subdir)
                        obj_save_dir = save_path
                        if self._object_mode == "All annotated" and lookup in key_to_class:
                            class_name = key_to_class[lookup]
                            obj_save_dir = obj_save_dir / class_name
                        if self._well_subdir:
                            obj_save_dir = obj_save_dir / well
                        obj_save_dir.mkdir(parents=True, exist_ok=True)

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
                        crop_img = img_data[y_min_p:y_max_p, x_min_p:x_max_p, :].copy()
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
                                tifffile.imwrite(str(obj_save_dir / fname), ch_data)
                        else:
                            # Save as multi-channel image (CYX format)
                            # crop_img is (H, W, C) -> transpose to (C, H, W)
                            multi_ch = np.moveaxis(crop_img, -1, 0)
                            fname = f"{img_name}_{label_id}.tiff"
                            tifffile.imwrite(str(obj_save_dir / fname), multi_ch)

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
