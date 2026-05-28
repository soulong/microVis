from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QSizePolicy, QToolTip, QVBoxLayout, QWidget


def _get_cmap(name: str):
    """Case-insensitive colormap lookup."""
    import matplotlib.pyplot as plt
    try:
        return plt.colormaps[name]
    except KeyError:
        return plt.colormaps[name.lower()]


class WellGridCanvas(QWidget):
    """Well plate grid visualization using matplotlib scatter plot."""

    well_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._figure = Figure(facecolor="#252536")

        # Pre-allocate BOTH axes once with a fixed GridSpec so their positions
        # never change across redraws.  The colorbar axes is simply shown or
        # hidden; it never steals space from the main axes (which is the root
        # cause of the shrinking-grid bug when using figure.colorbar(ax=…)).
        gs = GridSpec(
            1, 2,
            figure=self._figure,
            width_ratios=[20, 1],
            left=0.06, right=0.97,
            top=0.90, bottom=0.03,
            wspace=0.05,
        )
        self._axes = self._figure.add_subplot(gs[0, 0])
        self._axes.set_facecolor("#252536")
        self._cbar_ax = self._figure.add_subplot(gs[0, 1])
        self._cbar_ax.set_visible(False)

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet("background-color: #252536;")

        self._well_indices: dict[str, int] = {}
        self._data_map: dict[str, float | str] = {}
        self._col_name: str | None = None
        self._row_labels: list[str] = []
        self._n_rows: int = 0
        self._n_cols: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def update_grid(
        self,
        dm,
        table_name: str,
        col_val: tuple | None,
        agg: str,
        cmap: str,
        palette: str,
        fmt_name: str,
        selected_wells: set[str],
        metadata_map: dict[str, float | str] | None = None,
    ) -> None:
        """Redraw the well grid with current parameters."""
        import matplotlib.pyplot as plt

        self._axes.clear()
        self._axes.set_facecolor("#252536")
        # Clear the colorbar axes and hide it until we know we need it.
        self._cbar_ax.clear()
        self._cbar_ax.set_visible(False)
        self._well_indices.clear()

        rows, cols = dm.get_plate_dims()
        fmt_rows, fmt_cols = _get_format_dims(fmt_name)
        rows = max(rows, fmt_rows)
        cols = max(cols, fmt_cols)

        row_labels = _row_labels(rows)
        self._row_labels = row_labels
        self._n_rows = rows
        self._n_cols = cols
        all_wells = dm.get_wells()
        well_set = set(all_wells)

        # Compute data mapping via DataModule.aggregate()
        data_map: dict[str, float | str] = {}
        is_numeric = False
        self._col_name = None
        if col_val and col_val[0] is not None and table_name:
            col_name, is_numeric = col_val
            self._col_name = col_name
            if metadata_map is not None:
                data_map = metadata_map
            else:
                try:
                    data_map = dm.aggregate(table_name, col_name, agg)
                except Exception:
                    data_map = {}
        self._data_map = data_map

        marker_size = _compute_marker_size(rows, cols)
        x_vals, y_vals, color_list = [], [], []

        has_data = bool(data_map)
        cmap_obj = None
        norm = None
        cat_to_color = {}

        if has_data and is_numeric:
            numeric_vals = [v for v in data_map.values() if isinstance(v, (int, float))]
            if numeric_vals:
                vmin = min(numeric_vals)
                vmax = max(numeric_vals)
                cmap_obj = _get_cmap(cmap)
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
            else:
                has_data = False

        if has_data and not is_numeric:
            unique_cats = sorted(set(str(v) for v in data_map.values()))
            base = _get_cmap(palette)
            n_cats = len(unique_cats)
            cat_to_color = {
                cat: base(i / max(n_cats - 1, 1))
                for i, cat in enumerate(unique_cats)
            }

        for r in range(rows):
            for c in range(cols):
                well = f"{row_labels[r]}{c + 1}"
                x_vals.append(c + 1)
                y_vals.append(r + 1)

                if has_data and well in data_map:
                    val = data_map[well]
                    if is_numeric and isinstance(val, (int, float)):
                        color_list.append(cmap_obj(norm(val)))
                    elif not is_numeric:
                        color_list.append(cat_to_color.get(str(val), "#3a3a5a"))
                    else:
                        color_list.append("#3a3a5a")
                elif well in well_set:
                    color_list.append("#3a3a5a")
                else:
                    color_list.append("#1a1a2a")

        self._axes.scatter(
            x_vals, y_vals,
            s=marker_size,
            c=color_list,
            edgecolors="#555555",
            linewidths=0.5,
            picker=True,
            pickradius=5,
            zorder=2,
        )

        for i in range(len(x_vals)):
            well_at_pos = f"{row_labels[i // cols]}{(i % cols) + 1}"
            self._well_indices[well_at_pos] = i

        for i in range(len(x_vals)):
            well_at_pos = f"{row_labels[i // cols]}{(i % cols) + 1}"
            if well_at_pos in well_set:
                self._axes.annotate(
                    well_at_pos, (x_vals[i], y_vals[i]),
                    fontsize=5, color="#aaaaaa", ha="center", va="center", zorder=3,
                )

        for well in selected_wells:
            if well in well_set and well in self._well_indices:
                idx = self._well_indices[well]
                self._axes.scatter(
                    [x_vals[idx]], [y_vals[idx]],
                    s=marker_size * 1.3,
                    facecolors="none",
                    edgecolors="#5a8a9a",
                    linewidths=2,
                    zorder=4,
                )

        self._axes.set_xlim(0.5, cols + 0.5)
        self._axes.set_ylim(0.5, rows + 0.5)
        self._axes.set_xticks(range(1, cols + 1))
        self._axes.set_xticklabels([str(c) for c in range(1, cols + 1)])
        self._axes.set_yticks(range(1, rows + 1))
        self._axes.set_yticklabels(row_labels)
        self._axes.tick_params(colors="#888888", labelsize=7)
        self._axes.invert_yaxis()
        self._axes.xaxis.set_ticks_position('top')
        self._axes.xaxis.set_label_position('top')
        for spine in self._axes.spines.values():
            spine.set_color("#333333")

        if has_data and is_numeric and len(color_list) > 0:
            sm = ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            # Use cax= to draw into the pre-allocated colorbar axes.
            # This is the key fix: figure.colorbar(ax=self._axes, …) internally
            # calls ax.set_position() to shrink the main axes and make room —
            # that shrinkage accumulates across redraws, causing the grid to
            # drift left and compress.  With cax= the main axes is never touched.
            self._figure.colorbar(sm, cax=self._cbar_ax)
            self._cbar_ax.set_visible(True)
            self._cbar_ax.tick_params(colors="#888888", labelsize=7)
            for spine in self._cbar_ax.spines.values():
                spine.set_color("#333333")

        self._canvas.draw()

    def _on_pick(self, event) -> None:
        ind = event.ind
        if ind is not None and len(ind) > 0:
            idx = ind[0]
            for well, i in self._well_indices.items():
                if i == idx:
                    QTimer.singleShot(0, lambda w=well: self.well_clicked.emit(w))
                    return

    def _on_motion(self, event) -> None:
        if event.inaxes != self._axes:
            QToolTip.hideText()
            return
        if event.xdata is None or event.ydata is None:
            QToolTip.hideText()
            return
        col = round(event.xdata)
        row = round(event.ydata)
        if 1 <= row <= self._n_rows and 1 <= col <= self._n_cols:
            well = f"{self._row_labels[row - 1]}{col}"
            if well in self._data_map and self._col_name:
                val = self._data_map[well]
                if isinstance(val, float):
                    text = f"{well}  {self._col_name}: {val:.4f}"
                else:
                    text = f"{well}  {self._col_name}: {val}"
            elif well in self._well_indices:
                text = well
            else:
                text = ""
            if text:
                QToolTip.showText(QCursor.pos(), text, self)
            else:
                QToolTip.hideText()
        else:
            QToolTip.hideText()


def _row_labels(n: int) -> list[str]:
    labels = []
    for i in range(n):
        label = ""
        num = i
        while True:
            label = chr(65 + num % 26) + label
            num = num // 26 - 1
            if num < 0:
                break
        labels.append(label)
    return labels


def _compute_marker_size(n_rows: int, n_cols: int) -> float:
    max_dim = max(n_rows, n_cols)
    if max_dim <= 6:
        return 400
    elif max_dim <= 12:
        return 200
    elif max_dim <= 24:
        return 80
    else:
        return 40


def _get_format_dims(fmt_name: str) -> tuple[int, int]:
    from microVis._settings import PLATE_FORMATS
    return PLATE_FORMATS.get(fmt_name, (16, 24))
