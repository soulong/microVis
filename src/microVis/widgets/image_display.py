from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QDrag, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

_MIME_TYPE = "application/x-microvis-object"


class _PassThroughPixmapItem(QGraphicsPixmapItem):
    """QGraphicsPixmapItem that ignores all mouse events, letting them fall through."""

    def mousePressEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()


class _ThumbnailView(QGraphicsView):
    """Single image thumbnail with optional overlay polygons."""

    pixel_clicked = Signal(str, int, int, int, int, int, int, int)  # well, field, stack, tp, x, y, pixmap_w, pixmap_h
    full_res_requested = Signal(str, int, int, int, int)  # well, field, stack, tp, gen

    def __init__(
        self,
        rgb: np.ndarray,
        well: str,
        field: int,
        stack: int,
        tp: int,
        polygons: list | None = None,
        overlay_alpha: float = 0.4,
        overlay_cmap: str = "Viridis",
        overlay_val: float | str | None = None,
        overlay_col: str | None = None,
        n_objects: int | None = None,
        mask: np.ndarray | None = None,
        obj_values: dict | None = None,
        thumb_size: int = 210,
        row_idx: int = -1,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._well = well
        self._field = field
        self._stack = stack
        self._tp = tp
        self._row_idx = row_idx
        self._mask = mask
        self._thumb_mask = mask  # saved for restore on zoom-out
        self._obj_values = obj_values or {}
        self._overlay_col = overlay_col
        self._is_full_res = False
        self._full_res_item = None
        self._fade_timer = None
        self._full_res_gen = 0  # generation counter to reject stale results

        # Create QPixmap from numpy RGB
        self._base_rgb = rgb.copy()
        pixmap = _array_to_qpixmap(rgb)

        # Draw overlay polygons
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap)

        self._thumb_pixmap = pixmap

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        # Configure view
        self.setFixedSize(thumb_size, thumb_size)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #333333;")
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._panning = False
        self._pan_start = None
        self._last_tip = ""
        self._drag_start_pos = None  # for distinguishing click vs drag
        self.setMouseTracking(True)

        # Static tooltip fallback (when no mask for per-object hover)
        # Only show when overlay_col is set (not when Color by is None)
        if self._mask is None and overlay_col:
            parts = []
            if overlay_val is not None:
                if isinstance(overlay_val, float):
                    parts.append(f"{overlay_col}: {overlay_val:.4f}")
                else:
                    parts.append(f"{overlay_col}: {overlay_val}")
            if n_objects is not None:
                parts.append(f"objects: {n_objects}")
            if parts:
                self.setToolTip("\n".join(parts))

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            # Load full-res when zoomed past native resolution
            zoom = self.transform().m11()
            if zoom > 1.0 and not self._is_full_res:
                self._is_full_res = True
                self._full_res_gen += 1
                self.full_res_requested.emit(self._well, self._field, self._stack, self._tp, self._full_res_gen)
            elif zoom <= 1.0 and self._is_full_res:
                self._is_full_res = False
                self.remove_full_res()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Check for drag initiation (left button held + moved beyond threshold)
        if self._drag_start_pos is not None and event.buttons() & Qt.LeftButton:
            dist = (event.pos() - self._drag_start_pos).manhattanLength()
            if dist > 4:
                zoom = self.transform().m11()
                # Always try object drag first if mask exists
                if self._mask is not None:
                    scene_pos = self.mapToScene(event.pos())
                    x = int(scene_pos.x())
                    y = int(scene_pos.y())
                    h, w = self._mask.shape
                    if 0 <= x < w and 0 <= y < h:
                        lbl = int(self._mask[y, x])
                        if lbl > 0:
                            self._start_object_drag(lbl)
                            return
                # Zoomed in + no object found → pan
                if zoom > 1.0:
                    self._panning = True
                    self._pan_start = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    return
                # Unzoomed: do nothing (no pan, no object to drag)
                return

        # Per-object tooltip on hover
        if self._mask is not None:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            h, w = self._mask.shape
            if 0 <= x < w and 0 <= y < h:
                lbl = int(self._mask[y, x])
                if lbl > 0 and lbl in self._obj_values:
                    val = self._obj_values[lbl]
                    if isinstance(val, float):
                        tip = f"label: {lbl}\n{self._overlay_col}: {val:.4f}"
                    else:
                        tip = f"label: {lbl}\n{self._overlay_col}: {val}"
                elif lbl > 0:
                    tip = f"label: {lbl}"
                else:
                    tip = ""
                if tip != self._last_tip:
                    self._last_tip = tip
                    if tip:
                        QToolTip.showText(QCursor.pos(), tip, self)
                    else:
                        QToolTip.hideText()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        elif event.button() == Qt.LeftButton:
            if self._panning:
                self._panning = False
                self._pan_start = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
            else:
                # Short click = emit pixel_clicked (preserves existing behavior)
                if self._drag_start_pos is not None:
                    scene_pos = self.mapToScene(event.pos())
                    x = int(scene_pos.x())
                    y = int(scene_pos.y())
                    # Scene coords are always in thumbnail pixel space (both zoomed and not).
                    # Use thumbnail pixmap for bounds check and coordinate conversion.
                    pw = self._pixmap_item.pixmap().width()
                    ph = self._pixmap_item.pixmap().height()
                    if 0 <= x < pw and 0 <= y < ph:
                        self.pixel_clicked.emit(self._well, self._field, self._stack, self._tp, x, y, pw, ph)
                self._drag_start_pos = None
                super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_zoom()
        event.accept()

    def _start_object_drag(self, label: int) -> None:
        """Initiate a QDrag carrying the ObjectKey for a mask object."""
        from microVis.widgets.label_annotation import ObjectKey, encode_object_key

        key = ObjectKey(
            well=self._well, field=self._field,
            stack=self._stack, tp=self._tp, label=label,
        )
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_MIME_TYPE, encode_object_key(key))
        drag.setMimeData(mime)

        # Create a small drag pixmap from the object region
        if self._mask is not None:
            ys, xs = np.where(self._mask == label)
            if len(ys) > 0:
                pix = self._pixmap_item.pixmap()
                # Scale factors between pixmap and original scene
                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())
                # Crop from the pixmap
                crop = pix.copy(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
                drag.setPixmap(crop.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._drag_start_pos = None
        drag.exec(Qt.CopyAction)

    def set_pixmap(self, rgb: np.ndarray, polygons: list | None = None,
                   overlay_alpha: float = 0.4, overlay_cmap: str = "Viridis") -> None:
        """Update the displayed image in-place, preserving zoom/pan state."""
        self._base_rgb = rgb.copy()
        pixmap = _array_to_qpixmap(rgb)
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap)
        self._thumb_pixmap = pixmap
        if not self._is_full_res:
            self._pixmap_item.setPixmap(pixmap)

    def render_overlay(self, overlay_alpha: float, overlay_cmap: str,
                       polygons: list | None = None) -> None:
        """Re-render overlay with new styling, using cached base RGB."""
        pixmap = _array_to_qpixmap(self._base_rgb)
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap)
        self._thumb_pixmap = pixmap
        if not self._is_full_res:
            self._pixmap_item.setPixmap(pixmap)

    def set_full_res_pixmap(self, pixmap: QPixmap, gen: int = 0,
                            mask: np.ndarray | None = None,
                            obj_values: dict | None = None,
                            overlay_col: str = "") -> None:
        """Cross-fade from thumbnail to full-resolution pixmap."""
        from PySide6.QtCore import QTimer
        # Reject stale results (gen=0 means always accept)
        if gen != 0 and gen != self._full_res_gen:
            return
        # Update mask and overlay data for hover/drag interactivity
        if mask is not None:
            # Downscale full-res mask to thumbnail coordinate space for correct hover lookup
            thumb_w = self._thumb_pixmap.width()
            thumb_h = self._thumb_pixmap.height()
            h, w = mask.shape
            if h <= thumb_h and w <= thumb_w:
                self._mask = mask
            else:
                from skimage.transform import resize as sk_resize
                scale = min(thumb_h / h, thumb_w / w)
                self._mask = sk_resize(
                    mask, (int(h * scale), int(w * scale)),
                    order=0, preserve_range=True, anti_aliasing=False,
                ).astype(mask.dtype)
        if obj_values is not None:
            self._obj_values = obj_values
        if overlay_col:
            self._overlay_col = overlay_col
        # Remove previous full-res item if any
        if self._full_res_item is not None:
            self._scene.removeItem(self._full_res_item)

        # Scale full-res to match thumbnail's scene coordinates
        thumb_w = self._thumb_pixmap.width()
        thumb_h = self._thumb_pixmap.height()
        full_w = pixmap.width()
        full_h = pixmap.height()

        self._full_res_item = _PassThroughPixmapItem(pixmap)
        self._full_res_item.setZValue(1)
        self._full_res_item.setOpacity(0.0)
        # Scale so full-res pixels align with thumbnail coordinate space
        if full_w > 0 and full_h > 0:
            self._full_res_item.setScale(min(thumb_w / full_w, thumb_h / full_h))
        self._scene.addItem(self._full_res_item)

        # Fade in with QTimer
        steps = 5
        interval = 30  # ms per step → 150ms total
        self._fade_step = 0
        self._fade_steps = steps

        def _tick():
            self._fade_step += 1
            if self._full_res_item is not None:
                self._full_res_item.setOpacity(self._fade_step / self._fade_steps)
            if self._fade_step >= self._fade_steps:
                self._fade_timer.stop()

        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(_tick)
        self._fade_timer.start(interval)

    def remove_full_res(self) -> None:
        """Remove full-res overlay, restoring thumbnail view."""
        if self._fade_timer is not None:
            self._fade_timer.stop()
            self._fade_timer = None
        if self._full_res_item is not None:
            self._scene.removeItem(self._full_res_item)
            self._full_res_item = None
        # Restore original downscaled mask for hover/drag
        self._mask = self._thumb_mask

    def reset_zoom(self) -> None:
        self._is_full_res = False
        self.remove_full_res()
        self._pixmap_item.setPixmap(self._thumb_pixmap)
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def get_view_state(self) -> dict:
        return {
            "transform": self.transform(),
            "h_scroll": self.horizontalScrollBar().value(),
            "v_scroll": self.verticalScrollBar().value(),
        }

    def restore_view_state(self, state: dict) -> None:
        self.setTransform(state["transform"])
        self.horizontalScrollBar().setValue(state["h_scroll"])
        self.verticalScrollBar().setValue(state["v_scroll"])


class ImageDisplay(QScrollArea):
    """Scrollable area displaying image thumbnails grouped by well."""

    pixel_clicked = Signal(str, int, int, int, int, int, int, int)  # well, field, stack, tp, x, y, pixmap_w, pixmap_h
    full_res_requested = Signal(str, int, int, int, int)  # well, field, stack, tp, gen

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        self.setWidget(container)

        # Result cache for sort/resize without reprocessing
        self._results_cache: list[dict] = []
        self._cached_thumb_size: int = 0
        # Incremental display tracking: group_key → (row_widget, h_layout)
        self._row_widgets: dict = {}

    def save_view_state(self) -> dict:
        return self._save_current_view_state()

    def clear(self) -> None:
        self._clear_layout()

    def reset_all_zoom(self) -> None:
        """Reset zoom on all visible thumbnails."""
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            for thumb in item.widget().findChildren(_ThumbnailView):
                thumb.reset_zoom()

    def show_loading(self, count: int) -> None:
        self._clear_layout()
        lbl = QLabel(f"Loading {count} images...")
        lbl.setProperty("class", "muted")
        lbl.setAlignment(Qt.AlignCenter)
        self._layout.insertWidget(0, lbl)

    def begin_results(self, thumb_size: int) -> None:
        """Clear display and prepare for progressive thumbnail insertion."""
        self._clear_layout()
        self._results_cache = []
        self._cached_thumb_size = thumb_size
        self._row_widgets = {}

    def add_result(self, result: dict, thumb_size: int, overlay_alpha: float,
                   overlay_cmap: str, saved_state: dict | None,
                   sort_by_row: bool) -> None:
        """Incrementally insert a single result as a thumbnail."""
        self._results_cache.append(result)

        # Determine grouping key
        if sort_by_row:
            group_key = result["well"]
        else:
            group_key = (result["field"], result["stack"], result["tp"])

        # Remove trailing stretch
        if self._layout.count() > 0:
            last = self._layout.itemAt(self._layout.count() - 1)
            if last and last.spacerItem():
                self._layout.takeAt(self._layout.count() - 1)

        # Find or create row widget
        if group_key not in self._row_widgets:
            row_widget = QWidget()
            row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addStretch()
            self._row_widgets[group_key] = (row_widget, row)
            # Insert row in sorted position (left-to-right, top-to-bottom)
            self._insert_row_sorted(group_key, row_widget)
        else:
            row_widget, row = self._row_widgets[group_key]

        # Insert thumbnail in sorted position within the row
        self._add_thumbnail_sorted(row, result, thumb_size, overlay_alpha,
                                   overlay_cmap, saved_state, sort_by_row)

        # Re-add trailing stretch
        self._layout.addStretch()

    def _rebuild_display(self, thumb_size, overlay_alpha, overlay_cmap,
                         saved_state, sort_by_row):
        """Rebuild the entire display from cached results."""
        old_state = self._save_current_view_state()

        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout():
                self._clear_sub_layout(item.layout())
        self._row_widgets = {}

        if old_state:
            if saved_state:
                old_state.update(saved_state)
            saved_state = old_state

        from collections import defaultdict

        results = self._results_cache

        if sort_by_row:
            groups = defaultdict(list)
            for r in results:
                groups[r["well"]].append(r)
            for well in sorted(groups.keys()):
                row_widget = QWidget()
                row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                self._row_widgets[well] = (row_widget, row)
                for r in sorted(groups[well],
                                key=lambda x: (x["stack"], x["tp"], x["field"])):
                    self._add_thumbnail_column(row, r, thumb_size, overlay_alpha,
                                               overlay_cmap, saved_state)
                row.addStretch()
                self._layout.addWidget(row_widget)
        else:
            combos_set = set()
            wells_set = set()
            for r in results:
                combos_set.add((r["field"], r["stack"], r["tp"]))
                wells_set.add(r["well"])
            combos = sorted(combos_set)
            wells = sorted(wells_set)
            idx = defaultdict(list)
            for r in results:
                idx[(r["well"], r["field"], r["stack"], r["tp"])].append(r)
            for combo in combos:
                field, stack, tp = combo
                row_widget = QWidget()
                row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                self._row_widgets[combo] = (row_widget, row)
                for well in wells:
                    matching = idx.get((well, field, stack, tp), [])
                    if matching:
                        self._add_thumbnail_column(row, matching[0], thumb_size,
                                                   overlay_alpha, overlay_cmap, saved_state)
                row.addStretch()
                self._layout.addWidget(row_widget)

        has_polygons = any(r.get("polygons") for r in results)
        if has_polygons:
            cbar_widget = _create_colorbar_widget(overlay_cmap)
            self._layout.addWidget(cbar_widget)
        self._layout.addStretch()

    def _save_current_view_state(self) -> dict:
        """Save view state of currently visible thumbnails."""
        state = {}
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            for child in item.widget().findChildren(_ThumbnailView):
                key = (child._well, child._field, child._stack, child._tp)
                state[key] = child.get_view_state()
        return state

    def show_results(
        self,
        results: list,
        overlay_alpha: float = 0.4,
        overlay_cmap: str = "Viridis",
        thumb_size: int = 210,
        saved_state: dict | None = None,
        sort_by_row: bool = False,
    ) -> None:
        """Show all results. Also caches them for sort/resize without reprocessing."""
        self._results_cache = list(results)
        self._cached_thumb_size = thumb_size
        self._rebuild_display(thumb_size, overlay_alpha, overlay_cmap,
                              saved_state, sort_by_row)

    def resort_cached(self, thumb_size, overlay_alpha, overlay_cmap,
                      saved_state, sort_by_row) -> bool:
        """Re-sort cached results without reprocessing. Returns True if cache was usable."""
        if not self._results_cache:
            return False
        self._rebuild_display(thumb_size, overlay_alpha, overlay_cmap,
                              saved_state, sort_by_row)
        return True

    def update_pixmaps_in_place(self, results: list[dict], overlay_alpha: float,
                                overlay_cmap: str,
                                remove_full_res: bool = True) -> None:
        """Update existing thumbnail pixmaps without rebuilding the layout."""
        # Build lookup: (well, field, stack, tp) → result
        result_map = {}
        for r in results:
            key = (r["well"], r["field"], r["stack"], r["tp"])
            result_map[key] = r

        for row_widget, _row_layout in self._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                key = (thumb._well, thumb._field, thumb._stack, thumb._tp)
                r = result_map.get(key)
                if r is not None:
                    thumb.set_pixmap(r["rgb"], r.get("polygons"),
                                     overlay_alpha, overlay_cmap)
                    if remove_full_res:
                        thumb.remove_full_res()

        # Merge into results cache (don't replace — keep all cached results)
        for i, cached in enumerate(self._results_cache):
            key = (cached["well"], cached["field"], cached["stack"], cached["tp"])
            if key in result_map:
                self._results_cache[i] = result_map[key]

    def restyle_overlay(self, overlay_alpha: float, overlay_cmap: str,
                        polygon_cache: dict) -> None:
        """Re-render overlay polygons with new styling, no worker dispatch."""
        for row_widget, _row_layout in self._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                polygons = polygon_cache.get(thumb._row_idx)
                thumb.render_overlay(overlay_alpha, overlay_cmap, polygons)

    def _insert_row_sorted(self, group_key, row_widget):
        """Insert row widget at the correct sorted position in the layout."""
        # Build list of existing group_keys in layout order
        existing = []
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            for gk, (rw, _) in self._row_widgets.items():
                if rw is item.widget():
                    existing.append(gk)
                    break

        # Find insertion index
        insert_idx = len(existing)
        for i, gk in enumerate(existing):
            if group_key < gk:
                insert_idx = i
                break

        # Remove trailing stretch, insert row, re-add stretch
        last = self._layout.itemAt(self._layout.count() - 1)
        has_stretch = last is not None and last.spacerItem() is not None
        if has_stretch:
            self._layout.takeAt(self._layout.count() - 1)
        self._layout.insertWidget(insert_idx, row_widget)
        if has_stretch:
            self._layout.addStretch()

    def _add_thumbnail_sorted(self, row, r, thumb_size, overlay_alpha, overlay_cmap,
                               saved_state, sort_by_row):
        """Insert thumbnail at sorted position within a row."""
        if sort_by_row:
            # Sort by (stack, tp, field) within a well row
            new_key = (r["stack"], r["tp"], r["field"])
        else:
            # Sort by well name within a (field, stack, tp) row
            new_key = r["well"]

        # Find insertion index: count existing columns before the stretch
        insert_idx = row.count() - 1  # before trailing stretch
        for i in range(row.count() - 1):  # skip trailing stretch
            item = row.itemAt(i)
            if item is None or not item.layout():
                continue
            # Extract metadata label from the column to get sort key
            col_layout = item.layout()
            if col_layout.count() > 0:
                meta_item = col_layout.itemAt(0)
                if meta_item and meta_item.widget() and isinstance(meta_item.widget(), QLabel):
                    text = meta_item.widget().text()
                    existing_key = self._parse_sort_key(text, sort_by_row)
                    if existing_key is not None and new_key < existing_key:
                        insert_idx = i
                        break

        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(0, 0, 0, 0)

        meta = QLabel(f"{r['well']} f{r['field']} z{r['stack']} t{r['tp']}")
        meta.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        meta.setAlignment(Qt.AlignCenter)
        col.addWidget(meta)

        thumb = _ThumbnailView(
            r["rgb"], r["well"], r["field"], r["stack"], r["tp"],
            r.get("polygons"), overlay_alpha, overlay_cmap,
            r.get("overlay_val"), r.get("overlay_col"),
            r.get("n_objects"), r.get("mask"), r.get("obj_values"),
            thumb_size, row_idx=r.get("row_idx", -1),
        )
        thumb.pixel_clicked.connect(self.pixel_clicked)
        thumb.full_res_requested.connect(self.full_res_requested)
        if saved_state:
            key = (r["well"], r["field"], r["stack"], r["tp"])
            if key in saved_state:
                thumb.restore_view_state(saved_state[key])
        col.addWidget(thumb)
        row.insertLayout(insert_idx, col)

    @staticmethod
    def _parse_sort_key(label_text, sort_by_row):
        """Parse 'A1 f0 z0 t0' label into a sort key tuple."""
        try:
            parts = label_text.split()
            well = parts[0]
            field = int(parts[1][1:])
            stack = int(parts[2][1:])
            tp = int(parts[3][1:])
            if sort_by_row:
                return (stack, tp, field)
            else:
                return well
        except (IndexError, ValueError):
            return None

    def _add_thumbnail_column(self, row, r, thumb_size, overlay_alpha, overlay_cmap,
                               saved_state):
        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(0, 0, 0, 0)

        meta = QLabel(f"{r['well']} f{r['field']} z{r['stack']} t{r['tp']}")
        meta.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        meta.setAlignment(Qt.AlignCenter)
        col.addWidget(meta)

        thumb = _ThumbnailView(
            r["rgb"],
            r["well"],
            r["field"],
            r["stack"],
            r["tp"],
            r.get("polygons"),
            overlay_alpha,
            overlay_cmap,
            r.get("overlay_val"),
            r.get("overlay_col"),
            r.get("n_objects"),
            r.get("mask"),
            r.get("obj_values"),
            thumb_size,
            row_idx=r.get("row_idx", -1),
        )
        thumb.pixel_clicked.connect(self.pixel_clicked)
        thumb.full_res_requested.connect(self.full_res_requested)
        if saved_state:
            key = (r["well"], r["field"], r["stack"], r["tp"])
            if key in saved_state:
                thumb.restore_view_state(saved_state[key])
        col.addWidget(thumb)
        row.addLayout(col)

    def _clear_layout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout():
                self._clear_sub_layout(item.layout())


    def _clear_sub_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
            sub = item.layout()
            if sub is not None:
                self._clear_sub_layout(sub)
                del sub


def _array_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert (H, W, 3) uint8 RGB array to QPixmap via copy-safe QImage."""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    # Create pixmap immediately while rgb reference lives
    return QPixmap.fromImage(qimage.copy())


def _draw_polygon_overlays(
    pixmap: QPixmap,
    polygons: list[tuple[int, np.ndarray]],
    alpha: float = 0.4,
    cmap_name: str = "Viridis",
) -> QPixmap:
    """Draw filled polygon overlays onto a copy of pixmap using QPainter."""
    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)

    cmap = plt.colormaps[cmap_name]
    gold_pen = QPen(QColor(0xFF, 0xD7, 0x00), 1.0)
    _PHI = 0.618033988749895  # golden-ratio conjugate for deterministic hashing

    for label_val, contour in polygons:
        if len(contour) < 3:
            continue

        path = QPainterPath()
        path.moveTo(contour[0, 1], contour[0, 0])  # x=col, y=row
        for pt in contour[1:]:
            path.lineTo(pt[1], pt[0])
        path.closeSubpath()

        # Deterministic color per label (golden-ratio hash)
        rgba = cmap((label_val * _PHI) % 1.0)
        fill_color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], alpha)

        painter.fillPath(path, fill_color)
        painter.setPen(gold_pen)
        painter.drawPath(path)

    painter.end()
    return result


def _create_colorbar_widget(cmap_name: str) -> QWidget:
    """Create a horizontal colorbar widget for object overlay mapping."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(4)

    fig = Figure(facecolor="#252536", figsize=(3.0, 0.4), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.55, bottom=0.35)
    ax = fig.add_subplot(111)

    cmap = plt.colormaps[cmap_name]
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=ax, orientation="horizontal")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_facecolor("#252536")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    canvas = FigureCanvasQTAgg(fig)
    canvas.setFixedWidth(300)
    canvas.setFixedHeight(40)
    canvas.setStyleSheet("background-color: #252536;")
    layout.addWidget(canvas)

    layout.addStretch()
    return widget
