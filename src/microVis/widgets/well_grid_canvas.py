from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


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
        self._axes = self._figure.add_subplot(111)
        self._axes.set_facecolor("#252536")

        # Colorbar axes created once, never added/removed — avoids layout shifts
        self._cbar_ax = self._figure.add_axes([0.88, 0.08, 0.025, 0.86])
        self._cbar_ax.set_visible(False)
        self._cbar_ax.set_in_layout(False)
        self._colorbar = None

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet("background-color: #252536;")

        self._well_indices: dict[str, int] = {}

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
    ) -> None:
        """Redraw the well grid with current parameters."""
        import matplotlib.pyplot as plt

        self._axes.clear()
        self._well_indices.clear()

        rows, cols = dm.get_plate_dims()
        fmt_rows, fmt_cols = _get_format_dims(fmt_name)
        rows = max(rows, fmt_rows)
        cols = max(cols, fmt_cols)

        ratio = cols / max(rows, 1)
        self._figure.set_size_inches(max(4.5, 3.0 * ratio), 4.0)

        row_labels = _row_labels(rows)
        all_wells = dm.get_wells()
        well_set = set(all_wells)

        # Compute data mapping via DataModule.aggregate()
        data_map: dict[str, float | str] = {}
        is_numeric = False
        if col_val and col_val[0] is not None and table_name:
            col_name, is_numeric = col_val
            try:
                data_map = dm.aggregate(table_name, col_name, agg)
            except Exception:
                data_map = {}

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
                    edgecolors="#4cc9f0",
                    linewidths=2,
                    zorder=4,
                )

        # Colorbar: reuse the pre-created axes, just show/hide
        self._cbar_ax.clear()
        if has_data and is_numeric and len(color_list) > 0:
            sm = ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            self._colorbar = self._figure.colorbar(
                sm, cax=self._cbar_ax, orientation="vertical",
            )
            self._cbar_ax.tick_params(colors="#888888", labelsize=7)
            self._colorbar.outline.set_color("#333333")
            self._cbar_ax.set_visible(True)
        else:
            self._colorbar = None
            self._cbar_ax.set_visible(False)

        self._axes.set_xlim(0.5, cols + 0.5)
        self._axes.set_ylim(0.5, rows + 0.5)
        self._axes.set_xticks(range(1, cols + 1))
        self._axes.set_yticks(range(1, rows + 1))
        self._axes.set_yticklabels(row_labels)
        self._axes.tick_params(colors="#888888", labelsize=7)
        self._axes.invert_yaxis()
        self._axes.set_aspect("equal")
        for spine in self._axes.spines.values():
            spine.set_color("#333333")

        # Position main axes and colorbar to avoid overlap / shifting
        if has_data and is_numeric and len(color_list) > 0:
            main_left, main_bottom, main_width = 0.06, 0.08, 0.80
            cbar_left = main_left + main_width + 0.02
            self._axes.set_position([main_left, main_bottom, main_width, 0.86])
            self._cbar_ax.set_position([cbar_left, main_bottom, 0.025, 0.86])
        else:
            self._axes.set_position([0.06, 0.08, 0.88, 0.86])
            self._cbar_ax.set_visible(False)

        self._canvas.draw()

    def _on_pick(self, event) -> None:
        ind = event.ind
        if ind is not None and len(ind) > 0:
            idx = ind[0]
            for well, i in self._well_indices.items():
                if i == idx:
                    self.well_clicked.emit(well)
                    return


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
