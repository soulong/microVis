from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap
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


class _ThumbnailView(QGraphicsView):
    """Single image thumbnail with optional overlay polygons."""

    pixel_clicked = Signal(str, int, int, int, int, int)  # well, field, stack, tp, x, y

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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._well = well
        self._field = field
        self._stack = stack
        self._tp = tp
        self._mask = mask
        self._obj_values = obj_values or {}
        self._overlay_col = overlay_col

        # Create QPixmap from numpy RGB
        pixmap = _array_to_qpixmap(rgb)

        # Draw overlay polygons
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap)

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
        self.setMouseTracking(True)

        # Static tooltip fallback (when no mask for per-object hover)
        if self._mask is None:
            parts = []
            if overlay_val is not None and overlay_col:
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
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            h = self._pixmap_item.pixmap().height()
            w = self._pixmap_item.pixmap().width()
            if 0 <= x < w and 0 <= y < h:
                self.pixel_clicked.emit(self._well, self._field, self._stack, self._tp, x, y)
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
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        event.accept()

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

    pixel_clicked = Signal(str, int, int, int, int, int)  # well, field, stack, tp, x, y

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

    def save_view_state(self) -> dict:
        state = {}
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            row_widget = item.widget()
            for child in row_widget.findChildren(_ThumbnailView):
                key = (child._well, child._field, child._stack, child._tp)
                state[key] = child.get_view_state()
        return state

    def clear(self) -> None:
        self._clear_layout()

    def show_loading(self, count: int) -> None:
        self._clear_layout()
        lbl = QLabel(f"Loading {count} images...")
        lbl.setProperty("class", "muted")
        lbl.setAlignment(Qt.AlignCenter)
        self._layout.insertWidget(0, lbl)

    def show_results(
        self,
        results: list,
        overlay_alpha: float = 0.4,
        overlay_cmap: str = "Viridis",
        thumb_size: int = 210,
        saved_state: dict | None = None,
    ) -> None:
        self._clear_layout()

        # Group by well
        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            groups[r["well"]].append(r)

        for well in sorted(groups.keys()):
            # Thumbnail row
            row_widget = QWidget()
            row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            for r in groups[well]:
                col = QVBoxLayout()
                col.setSpacing(2)
                col.setContentsMargins(0, 0, 0, 0)

                meta = QLabel(f"{well} f{r['field']} z{r['stack']} t{r['tp']}")
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
                )
                thumb.pixel_clicked.connect(self.pixel_clicked)
                if saved_state:
                    key = (r["well"], r["field"], r["stack"], r["tp"])
                    if key in saved_state:
                        thumb.restore_view_state(saved_state[key])
                col.addWidget(thumb)
                row.addLayout(col)
            row.addStretch()
            self._layout.addWidget(row_widget)

        # Object color mapping colorbar
        has_polygons = any(r.get("polygons") for r in results)
        if has_polygons:
            cbar_widget = _create_colorbar_widget(overlay_cmap)
            self._layout.addWidget(cbar_widget)

        self._layout.addStretch()

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
    rng = np.random.RandomState(42)
    gold_pen = QPen(QColor(0xFF, 0xD7, 0x00), 1.0)

    for label_val, contour in polygons:
        if len(contour) < 3:
            continue

        path = QPainterPath()
        path.moveTo(contour[0, 1], contour[0, 0])  # x=col, y=row
        for pt in contour[1:]:
            path.lineTo(pt[1], pt[0])
        path.closeSubpath()

        # Random color per object
        rgba = cmap(rng.uniform(0, 1))
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

    fig = Figure(facecolor="#252536", figsize=(0.625, 0.25))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.05)
    ax = fig.add_subplot(111)

    cmap = plt.colormaps[cmap_name]
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=ax, orientation="horizontal")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(colors="white", labelsize=30)
    ax.set_facecolor("#252536")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    canvas = FigureCanvasQTAgg(fig)
    canvas.setFixedWidth(300)
    canvas.setFixedHeight(20)
    canvas.setStyleSheet("background-color: #252536;")
    layout.addWidget(canvas)

    layout.addStretch()
    return widget
