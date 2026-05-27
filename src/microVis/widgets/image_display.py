from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QScrollArea,
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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._well = well
        self._field = field
        self._stack = stack
        self._tp = tp

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
        self.setFixedSize(210, 210)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #333333;")
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        # Label
        lbl = QLabel(f"f{field} z{stack} t{tp}", self)
        lbl.setStyleSheet("color: #cccccc; background-color: rgba(0,0,0,120); font-size: 8pt; padding: 1px 4px;")
        lbl.move(4, 4)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            # Map click to image pixel coordinates
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            h = self._pixmap_item.pixmap().height()
            w = self._pixmap_item.pixmap().width()
            if 0 <= x < w and 0 <= y < h:
                self.pixel_clicked.emit(self._well, self._field, self._stack, self._tp, x, y)
        super().mousePressEvent(event)


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
    ) -> None:
        self._clear_layout()

        # Group by well
        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            groups[r["well"]].append(r)

        for well in sorted(groups.keys()):
            # Well header
            header = QLabel(well)
            header.setStyleSheet("font-weight: bold; color: #4cc9f0; padding-top: 4px;")
            self._layout.addWidget(header)

            # Thumbnail row
            row = QHBoxLayout()
            row.setSpacing(6)
            for r in groups[well]:
                thumb = _ThumbnailView(
                    r["rgb"],
                    r["well"],
                    r["field"],
                    r["stack"],
                    r["tp"],
                    r.get("polygons"),
                    overlay_alpha,
                    overlay_cmap,
                )
                thumb.pixel_clicked.connect(self.pixel_clicked)
                row.addWidget(thumb)
            row.addStretch()
            self._layout.addLayout(row)

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
            elif item.layout():
                self._clear_sub_layout(item.layout())


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
