"""Object crop dataset for deep learning training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.transform import resize as sk_resize

from microVis.log_utils import get_logger
from microVis.widgets.label_annotation import ObjectKey

_log = get_logger("microVis.model_dataset")


@dataclass
class CropRecord:
    """Metadata for a single object crop."""

    row_idx: int
    well: str
    field: int
    stack: int
    tp: int
    label_id: int
    bbox: tuple[int, int, int, int]  # y_min, y_max, x_min, x_max
    class_name: str | None = None
    class_idx: int | None = None
    split: str = "train"  # "train", "val", or "test"


@dataclass
class DatasetSummary:
    """Summary statistics for the prepared dataset."""

    total_objects: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    n_classes: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    crop_size: int = 64
    n_channels: int = 0
    channel_names: list[str] = field(default_factory=list)
    has_labels: bool = False


def extract_object_crops(
    img_data: np.ndarray,
    mask: np.ndarray,
    channel_names: list[str],
    crop_size: int,
    padding: int = 4,
) -> list[tuple[np.ndarray, int, tuple[int, int, int, int]]]:
    """Extract all object crops from a single image+mask pair.

    Returns list of (crop_tensor[C,H,W], label_id, bbox).
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    results = []
    h, w = mask.shape
    n_ch = img_data.shape[2] if img_data.ndim == 3 else 1

    for label_id in unique_labels:
        # Find bounding box
        ys, xs = np.where(mask == label_id)
        if len(ys) == 0:
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        # Add padding
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding + 1)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding + 1)

        # Crop image and mask
        if img_data.ndim == 3:
            crop_img = img_data[y_min:y_max, x_min:x_max, :].astype(np.float64)
        else:
            crop_img = img_data[y_min:y_max, x_min:x_max].astype(np.float64)
            crop_img = crop_img[:, :, np.newaxis]

        crop_mask = mask[y_min:y_max, x_min:x_max]

        # Apply mask: zero out background
        obj_mask = (crop_mask == label_id).astype(np.float64)
        for ch in range(crop_img.shape[2]):
            crop_img[:, :, ch] *= obj_mask

        # Normalize to [0, 1] per channel
        for ch in range(crop_img.shape[2]):
            ch_max = crop_img[:, :, ch].max()
            if ch_max > 0:
                crop_img[:, :, ch] /= ch_max

        # Resize to target size
        ch_h, ch_w = crop_img.shape[:2]
        if ch_h != crop_size or ch_w != crop_size:
            crop_img = sk_resize(
                crop_img,
                (crop_size, crop_size, crop_img.shape[2]),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float64)

        # Convert to (C, H, W) format
        crop_tensor = np.moveaxis(crop_img, -1, 0).astype(np.float32)
        bbox = (y_min, y_max, x_min, x_max)
        results.append((crop_tensor, int(label_id), bbox))

    return results


def prepare_dataset(
    dm: Any,
    wells: list[str],
    fields: list[int],
    stacks: list[int] | None,
    timepoints: list[int] | None,
    mask_name: str,
    channel_names: list[str],
    crop_size: int,
    annotations: dict[str, list[ObjectKey]] | None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[CropRecord], DatasetSummary]:
    """Prepare dataset records from selected wells/fields.

    Returns (records, summary). Records contain metadata only;
    actual crop loading happens in __getitem__.
    """
    from sklearn.model_selection import train_test_split

    rows = dm.lookup_row_indices(wells, fields, stacks, timepoints)
    if not rows:
        return [], DatasetSummary()

    # Build class mapping from annotations
    class_names: list[str] = []
    class_to_idx: dict[str, int] = {}
    if annotations:
        for cls_name, keys in annotations.items():
            if keys:  # only include non-empty classes
                class_names.append(cls_name)
        class_names.sort()
        class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Build ObjectKey → class_name lookup
    key_to_class: dict[tuple, str] = {}
    if annotations:
        for cls_name, keys in annotations.items():
            for key in keys:
                lookup = (key.well, key.field, key.stack, key.tp, key.label)
                key_to_class[lookup] = cls_name

    # Scan all images and collect crop records
    records: list[CropRecord] = []
    mask_col = f"mask_{mask_name}"

    for row_idx, well, field, stack, tp in rows:
        try:
            _, mask_dict = dm.get_imageset_with_masks(
                row_idx, channels=None, masks=[mask_col]
            )
            mask = mask_dict.get(mask_col)
            if mask is None:
                continue

            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels > 0]

            for label_id in unique_labels:
                ys, xs = np.where(mask == label_id)
                if len(ys) == 0:
                    continue

                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())

                # Look up class from annotations
                lookup = (well, field, stack, tp, int(label_id))
                class_name = key_to_class.get(lookup)
                class_idx = class_to_idx.get(class_name) if class_name else None

                # For SL mode: skip unlabeled objects
                if annotations is not None and class_name is None:
                    continue

                records.append(CropRecord(
                    row_idx=row_idx,
                    well=well,
                    field=field,
                    stack=stack,
                    tp=tp,
                    label_id=int(label_id),
                    bbox=(y_min, y_max, x_min, x_max),
                    class_name=class_name,
                    class_idx=class_idx,
                ))
        except Exception:
            _log.warning("Failed to process row %d (%s)", row_idx, (well, field, stack, tp))

    if not records:
        return [], DatasetSummary()

    # Split into train/val/test
    indices = list(range(len(records)))
    labels_for_split = [r.class_idx if r.class_idx is not None else 0 for r in records]

    if annotations and len(class_names) > 1:
        # Stratified split
        train_idx, temp_idx = train_test_split(
            indices, test_size=(1 - train_ratio),
            stratify=[labels_for_split[i] for i in indices],
            random_state=seed,
        )
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_ratio_adjusted),
            stratify=[labels_for_split[i] for i in temp_idx],
            random_state=seed,
        )
    else:
        train_idx, temp_idx = train_test_split(
            indices, test_size=(1 - train_ratio), random_state=seed
        )
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_ratio_adjusted), random_state=seed
        )

    # Assign split labels
    for i in train_idx:
        records[i].split = "train"
    for i in val_idx:
        records[i].split = "val"
    for i in test_idx:
        records[i].split = "test"

    # Build summary
    class_counts: dict[str, int] = {}
    for r in records:
        if r.class_name:
            class_counts[r.class_name] = class_counts.get(r.class_name, 0) + 1

    summary = DatasetSummary(
        total_objects=len(records),
        class_counts=class_counts,
        n_classes=len(class_names),
        train_count=len(train_idx),
        val_count=len(val_idx),
        test_count=len(test_idx),
        crop_size=crop_size,
        n_channels=len(channel_names),
        channel_names=channel_names,
        has_labels=annotations is not None,
    )

    return records, summary
