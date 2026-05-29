from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Qt, QThreadPool, QTimer, Signal, QRunnable
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.transform import resize as sk_resize

from microVis._settings import (
    AGG_METHODS,
    CMAP_OPTIONS,
    DEFAULT_CHANNEL_COLORS,
    DEFAULT_CMAP,
    DEFAULT_PLATE,
    DTYPE_MAX,
    PLATE_FORMATS,
    QUALITATIVE_PALETTES,
)
from microVis.io.data_module import DataModule
from microVis.log_utils import get_logger
from microVis.processing.compositing import composite_image
from microVis.processing.contrast import apply_contrast
from microVis.processing.overlay import extract_polygons
from microVis.widgets._event_filter import RotatedLabel
from microVis.widgets.data_view import DataView
from microVis.widgets.image_controls import ImageControls
from microVis.widgets.image_display import ImageDisplay
from microVis.widgets.pixel_info import PixelInfo
from microVis.widgets.well_grid_canvas import WellGridCanvas
from microVis.widgets.well_grid_controls import WellGridControls

_log = get_logger("microVis.main_window")


# ── Background Worker ──────────────────────────────────────────────────────

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


class _WorkerSignals(QObject):
    finished = Signal(dict)
    error = Signal(str)


class _ImageWorker(QRunnable):
    """Process a single image at thumbnail resolution in a background thread."""

    def __init__(self, row_idx, well, field, stack, tp,
                 raw_data, thumb_size,
                 channel_names, ch_config, dmax,
                 contrast_method, contrast_gamma, invert,
                 need_polygons, overlay_val, overlay_col,
                 n_objects, obj_values, gen, sort_by_row):
        super().__init__()
        self.signals = _WorkerSignals()
        self._gen = gen
        self._row_idx = row_idx
        self._well = well
        self._field = field
        self._stack = stack
        self._tp = tp
        self._raw_data = raw_data
        self._thumb_size = thumb_size
        self._channel_names = channel_names
        self._ch_config = ch_config
        self._dmax = dmax
        self._contrast_method = contrast_method
        self._contrast_gamma = contrast_gamma
        self._invert = invert
        self._need_polygons = need_polygons
        self._overlay_val = overlay_val
        self._overlay_col = overlay_col
        self._n_objects = n_objects
        self._obj_values = obj_values
        self._sort_by_row = sort_by_row
        self.setAutoDelete(False)

    @property
    def gen(self) -> int:
        return self._gen

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image
            from microVis.processing.contrast import apply_contrast
            from microVis.processing.overlay import extract_polygons

            img_data, mask_dict = self._raw_data

            # Downscale to thumbnail resolution
            img_small = _downscale_image(img_data, self._thumb_size)
            mask_name, mask_small = _downscale_mask(mask_dict, self._thumb_size)

            # Per-channel contrast on small image
            enhanced = np.zeros_like(img_small, dtype=np.float64)
            for ch_idx, ch_name in enumerate(self._channel_names):
                ch_cfg = self._ch_config.get(ch_name, {})
                if not ch_cfg.get("enabled", True):
                    enhanced[:, :, ch_idx] = img_small[:, :, ch_idx].astype(np.float64)
                    continue
                vmin = ch_cfg.get("vmin", 0)
                vmax = ch_cfg.get("vmax", self._dmax)
                band = img_small[:, :, ch_idx].astype(np.float64)
                band = np.clip((band - vmin) / max(vmax - vmin, 1e-10), 0, 1)
                if self._contrast_method == "gamma":
                    band = apply_contrast(band, "gamma", gamma=self._contrast_gamma)
                elif self._contrast_method == "histogram_equalization":
                    band = apply_contrast(band, "histogram_equalization")
                if self._invert:
                    band = 1.0 - band
                enhanced[:, :, ch_idx] = band

            # Composite
            comp_config = {ch: {**cfg, "vmin": 0, "vmax": 1}
                           for ch, cfg in self._ch_config.items()}
            rgb = composite_image(enhanced, self._channel_names, comp_config, None, None)

            # Polygons from downscaled mask
            polygons = None
            first_mask = None
            if self._need_polygons and mask_small is not None:
                first_mask = mask_small
                polygons = extract_polygons(first_mask)

            result = {
                "rgb": np.ascontiguousarray(rgb),
                "well": self._well, "field": self._field,
                "stack": self._stack, "tp": self._tp,
                "polygons": polygons,
                "overlay_val": self._overlay_val,
                "overlay_col": self._overlay_col,
                "n_objects": self._n_objects,
                "mask": first_mask,
                "obj_values": self._obj_values,
                "gen": self._gen,
                "thumb_size": self._thumb_size,
                "sort_by_row": self._sort_by_row,
            }
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))




class MainWindow(QMainWindow):
    """Top-level application window for microVis."""

    def __init__(self, dataset_dir: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(1200, 800)
        self.setWindowTitle("microVis")

        # Data
        self._dm: DataModule | None = None
        self._dataset_dir: str | None = None

        # Selection state
        self._selected_wells: set[str] = set()

        # Channel config: {ch_name: {enabled, color: (r,g,b), vmin, vmax}}
        self._ch_config: dict = {}

        # Image display params
        self._contrast_method: str = "none"
        self._contrast_gamma: float = 1.0
        self._invert: bool = False
        self._overlay_table: str | None = None
        self._overlay_col: str | None = None
        self._overlay_cmap: str = "Viridis"
        self._overlay_alpha: float = 0.4

        # Image debounce timer
        self._debounce = QTimer(singleShot=True, interval=300, timeout=self._refresh_images)

        # Performance: caches and state tracking
        self._raw_cache: OrderedDict[int, tuple] = OrderedDict()  # LRU, row_idx → (img_data, mask_dict)
        self._raw_cache_max: int = 50  # max cached images
        self._gen: int = 0  # generation counter for cancelling stale workers
        self._last_state: dict = {}  # for change detection
        self._thread_pool = QThreadPool.globalInstance()
        self._pending_workers: int = 0
        self._shutting_down: bool = False

        # Metadata
        self._metadata_df: pd.DataFrame | None = None
        self._metadata_merged: pd.DataFrame | None = None
        self._current_table: str | None = None

        self._build_ui()
        self._connect_signals()

        # Load dataset if provided
        if dataset_dir:
            self._load_dataset(dataset_dir)

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_main_page(), stretch=1)

        # Status bar (hidden)
        self.statusBar().setVisible(False)
        self.statusBar().setMaximumHeight(0)

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── Left nav sidebar ──
        nav = QWidget()
        nav.setObjectName("sidebar")
        nav.setFixedWidth(32)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        self._nav_data = RotatedLabel("Data")
        self._nav_data.setProperty("class", "nav-tab")
        self._nav_data.setProperty("active", "true")
        _data_font = self._nav_data.font()
        _data_font.setBold(True)
        self._nav_data.setFont(_data_font)

        self._nav_plate = RotatedLabel("Image")
        self._nav_plate.setProperty("class", "nav-tab")
        self._nav_plate.setProperty("active", "false")
        _plate_font = self._nav_plate.font()
        _plate_font.setBold(True)
        self._nav_plate.setFont(_plate_font)

        nav_layout.addWidget(self._nav_data)
        nav_layout.addWidget(self._nav_plate)
        nav_layout.addStretch()

        # ── Stacked content ──
        self._stack_content = QStackedWidget()

        # Page 0: Data View
        self._data_view = DataView()
        self._stack_content.addWidget(self._data_view)

        # Page 1: Plate & Images
        self._stack_content.addWidget(self._build_plate_images_tab())

        self._nav_data.clicked.connect(lambda: self._switch_tab(0))
        self._nav_plate.clicked.connect(lambda: self._switch_tab(1))

        body.addWidget(nav)
        body.addWidget(self._stack_content, stretch=1)

        outer.addLayout(body, stretch=1)

        # Pixel info bar at bottom
        self._pixel_info = PixelInfo()
        outer.addWidget(self._pixel_info)

        return page

    def _switch_tab(self, index: int) -> None:
        self._stack_content.setCurrentIndex(index)
        self._nav_data.setProperty("active", index == 0)
        self._nav_plate.setProperty("active", index == 1)
        for w in (self._nav_data, self._nav_plate):
            w.style().unpolish(w)
            w.style().polish(w)

    def _build_plate_images_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Top splitter: Well Grid ──
        top_splitter = QSplitter(Qt.Horizontal)

        self._grid_controls = WellGridControls()
        self._grid_canvas = WellGridCanvas()

        top_splitter.addWidget(self._grid_controls)
        top_splitter.addWidget(self._grid_canvas)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([220, 600])

        # ── Bottom splitter: Image View ──
        bottom_splitter = QSplitter(Qt.Horizontal)

        self._image_controls = ImageControls()
        self._image_display = ImageDisplay()

        bottom_splitter.addWidget(self._image_controls)
        bottom_splitter.addWidget(self._image_display)
        bottom_splitter.setStretchFactor(0, 0)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([220, 600])

        # ── Vertical splitter between grid and image sections ──
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(top_splitter)
        v_splitter.addWidget(bottom_splitter)
        v_splitter.setSizes([300, 700])

        layout.addWidget(v_splitter)
        return tab

    # ── Signal Connections ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Well grid controls
        gw = self._grid_controls
        gw.plate_format.currentTextChanged.connect(self._on_grid_params_changed)
        gw.column.currentTextChanged.connect(self._on_grid_params_changed)
        gw.column.completer().activated.connect(self._on_grid_params_changed)
        gw.aggregation.currentTextChanged.connect(self._on_grid_params_changed)
        gw.colormap.currentTextChanged.connect(self._on_grid_params_changed)
        gw.palette.currentTextChanged.connect(self._on_grid_params_changed)
        gw.select_all_clicked.connect(self._on_select_all)
        gw.clear_clicked.connect(self._on_clear_selection)

        # Well grid canvas
        self._grid_canvas.well_clicked.connect(self._on_well_clicked)

        # Image controls (filter widget signals connected in _populate_image_controls)
        ic = self._image_controls
        ic.auto_all_clicked.connect(self._on_auto_all)
        ic.auto_range_changed.connect(self._on_auto_all)
        ic.reset_requested.connect(self._on_reset)
        ic.image_size_changed.connect(self._schedule_image_refresh)
        ic.channel_config_changed.connect(self._schedule_image_refresh)
        ic.contrast.currentTextChanged.connect(self._on_contrast_changed)
        ic.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        ic.overlay_col.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_col.completer().activated.connect(self._on_overlay_changed)
        ic.overlay_cmap.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_alpha.valueChanged.connect(self._on_overlay_changed)
        ic.sort_mode_changed.connect(self._schedule_image_refresh)

        # Image display (pixel click)
        self._image_display.pixel_clicked.connect(self._on_pixel_clicked)

        # Data view
        self._data_view.dataset_browse_clicked.connect(self._on_dataset_browse)
        self._data_view.pygwalker_open_clicked.connect(self._on_pygwalker_open)
        self._data_view.metadata_browse_clicked.connect(self._on_metadata_browse)
        self._data_view.metadata_merge_clicked.connect(self._on_metadata_merge)
        self._data_view.metadata_clear_clicked.connect(self._on_metadata_clear)
        self._data_view.write_to_db_clicked.connect(self._on_write_to_db)
        self._data_view.table_radio_selected.connect(self._on_data_table_changed)

    # ── Dataset Loading ──────────────────────────────────────────────────────

    def _on_dataset_browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            self._load_dataset(path)

    def _load_dataset(self, path: str) -> None:
        p = Path(path)
        if not p.is_dir():
            return
        if not (p / "image").is_dir():
            return

        try:
            self._dm = DataModule(str(p))
        except Exception:
            _log.exception("Failed to load dataset from %s", p)
            return

        self._dataset_dir = str(p)
        self._data_view.set_dataset_label(str(p))

        # Reset UI state for new dataset
        self._image_display.clear()
        self._raw_cache.clear()
        self._last_state.clear()
        self._grid_canvas.update_grid(
            self._dm, table_name="", col_val=(None, False), agg="mean",
            cmap="viridis", palette="Set1", fmt_name=DEFAULT_PLATE,
            selected_wells=set(),
        )

        # Init selection (no wells selected by default)
        self._selected_wells = set()

        # Init channel config
        self._init_channel_config()

        # Populate controls
        self._populate_grid_controls()
        self._populate_image_controls()
        self._populate_data_controls()

        # Initial render
        self._update_grid()
        self._schedule_image_refresh()

    # ── Channel Config ───────────────────────────────────────────────────────

    def _init_channel_config(self) -> None:
        self._ch_config.clear()
        channels = self._dm.channels
        for i, ch in enumerate(channels):
            color = DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]
            self._ch_config[ch] = {
                "enabled": True,
                "color": color,
                "vmin": 0,
                "vmax": 65535 if self._dm.img_dtype in ("uint16",) else 255,
            }

    def _auto_range(self, low_pct: float = 1.0, high_pct: float = 99.0,
                    ch_name: str | None = None) -> tuple[float, float]:
        """Compute percentile range across a sample of images."""
        dm = self._dm
        wells = dm.get_wells()[:5]
        fields = dm.get_fields()[:1]
        stacks = dm.get_stacks()[:1]
        tps = dm.get_timepoints()[:1]
        rows = dm.lookup_row_indices(wells, fields, stacks, tps)
        p_lo, p_hi = 0.0, 65535.0
        if not rows:
            return p_lo, p_hi
        try:
            if ch_name is not None:
                ch_idx = dm.channels.index(ch_name) if ch_name in dm.channels else -1
                if ch_idx < 0:
                    return p_lo, p_hi
                samples = []
                for row_idx, _, _, _, _ in rows:
                    img_data, _ = dm.get_imageset(row_idx)
                    if ch_idx < img_data.shape[2]:
                        samples.append(img_data[:, :, ch_idx].ravel())
                all_pixels = np.concatenate(samples)
            else:
                samples = []
                for row_idx, _, _, _, _ in rows:
                    img_data, _ = dm.get_imageset(row_idx)
                    samples.append(img_data.ravel())
                all_pixels = np.concatenate(samples)
            p_lo = float(np.percentile(all_pixels, low_pct))
            p_hi = float(np.percentile(all_pixels, high_pct))
        except Exception:
            _log.warning("Auto-range percentile computation failed", exc_info=True)
        return p_lo, p_hi

    # ── Populate Controls ────────────────────────────────────────────────────

    def _populate_grid_controls(self) -> None:
        gw = self._grid_controls

        # Plate formats
        gw.plate_format.blockSignals(True)
        gw.plate_format.clear()
        gw.plate_format.addItems(list(PLATE_FORMATS.keys()))
        idx = gw.plate_format.findText(DEFAULT_PLATE)
        if idx >= 0:
            gw.plate_format.setCurrentIndex(idx)
        gw.plate_format.blockSignals(False)

        self._update_grid_columns()

        gw.aggregation.blockSignals(True)
        gw.aggregation.clear()
        gw.aggregation.addItems(AGG_METHODS)
        gw.aggregation.blockSignals(False)

        gw.colormap.blockSignals(True)
        gw.colormap.clear()
        gw.colormap.addItems(CMAP_OPTIONS)
        gw.colormap.setCurrentText(DEFAULT_CMAP)
        gw.colormap.blockSignals(False)

        gw.palette.blockSignals(True)
        gw.palette.clear()
        gw.palette.addItems(QUALITATIVE_PALETTES)
        gw.palette.setCurrentText("Set1")
        gw.palette.blockSignals(False)

    def _update_grid_columns(self) -> None:
        if self._dm is None:
            return
        gw = self._grid_controls
        gw.column.blockSignals(True)
        gw.column.clear()
        gw.column.addItem("None")
        tables = self._dm.get_profiling_tables()
        for tname in tables:
            cols = self._dm.get_profiling_columns(tname)
            for name, _ctype, is_num in cols:
                gw.column.addItem(f"{tname}/{name}", (tname, name, is_num))
        # Add merged metadata columns
        if self._metadata_merged is not None:
            for col in self._metadata_merged.columns:
                if col != "well":
                    gw.column.addItem(f"metadata/{col}", ("metadata", col, False))
        gw.column.blockSignals(False)

    def _populate_image_controls(self) -> None:
        dm = self._dm
        ic = self._image_controls

        # Fields / Stacks / Timepoints
        ic.set_filter_options(
            [str(f) for f in dm.get_fields()],
            [str(s) for s in dm.get_stacks()],
            [str(t) for t in dm.get_timepoints()],
        )
        ic.fields_widget.selection_changed.connect(self._on_image_filter_changed)
        ic.stacks_widget.selection_changed.connect(self._on_image_filter_changed)
        ic.tps_widget.selection_changed.connect(self._on_image_filter_changed)

        # Channel controls
        ic.set_channels(self._ch_config)

        # Overlay column
        ic.overlay_col.blockSignals(True)
        ic.overlay_col.clear()
        ic.overlay_col.addItem("None")
        tables = dm.get_profiling_tables()
        for tname in tables:
            cols = dm.get_profiling_columns(tname)
            for name, _ctype, _is_num in cols:
                ic.overlay_col.addItem(f"{tname}/{name}", (tname, name))
        ic.overlay_col.blockSignals(False)

        ic.overlay_cmap.blockSignals(True)
        ic.overlay_cmap.clear()
        ic.overlay_cmap.addItems(CMAP_OPTIONS)
        ic.overlay_cmap.setCurrentText(DEFAULT_CMAP)
        ic.overlay_cmap.blockSignals(False)

    def _populate_data_controls(self) -> None:
        if self._dm is None:
            self._data_view.set_table_names([])
            self._data_view.clear_table()
            self._data_view.set_pygwalker_buttons(False)
            return
        tables = self._dm.get_profiling_tables()
        table_names = list(tables.keys())
        self._data_view.set_table_names(table_names)
        self._data_view.set_pygwalker_buttons(bool(tables))
        if not tables:
            self._data_view.clear_table()

    # ── Metadata ─────────────────────────────────────────────────────────────

    def _on_metadata_browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata File", "", "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            from microVis.io.data_module import parse_plate_metadata
            self._metadata_df = parse_plate_metadata(path)
            self._data_view.set_metadata_label(Path(path).name)

        except Exception:
            _log.exception("Failed to load metadata from %s", path)
            self._metadata_df = None
            self._data_view.set_metadata_label(None)

    def _on_metadata_merge(self) -> None:
        if self._metadata_df is None:
            return
        self._metadata_merged = self._metadata_df.copy()

        self._refresh_data_table()
        self._update_overlay_with_metadata()

    def _on_metadata_clear(self) -> None:
        self._metadata_df = None
        self._metadata_merged = None
        self._data_view.set_metadata_label(None)

        self._refresh_data_table()
        self._update_overlay_with_metadata()

    def _update_overlay_with_metadata(self) -> None:
        ic = self._image_controls
        ic.overlay_col.blockSignals(True)
        # Remove existing metadata items (tagged with "metadata/" prefix)
        for i in range(ic.overlay_col.count() - 1, -1, -1):
            data = ic.overlay_col.itemData(i)
            if data and isinstance(data, tuple) and data[0] == "metadata":
                ic.overlay_col.removeItem(i)
        # Add merged metadata columns
        if self._metadata_merged is not None:
            for col in self._metadata_merged.columns:
                if col != "well":
                    ic.overlay_col.addItem(f"metadata/{col}", ("metadata", col))
        ic.overlay_col.blockSignals(False)
        # Also update the well grid Color by dropdown
        self._update_grid_columns()

    def _on_write_to_db(self) -> None:
        if self._dm is None or self._metadata_merged is None:
            return
        try:
            tables = self._dm.get_profiling_tables()
            import sqlite3
            db_path = str(Path(self._dm._root_dir) / "results.db")
            conn = sqlite3.connect(db_path)
            for tname in tables:
                df = self._dm.get_table_df(tname)
                if df is not None and "well" in df.columns:
                    merged = self._join_metadata(df)
                    merged.to_sql(tname, conn, if_exists="replace", index=False)
            conn.close()

        except Exception:
            _log.exception("Failed to write metadata to database")

    def _refresh_data_table(self) -> None:
        if self._current_table:
            self._on_data_table_changed(self._current_table)

    # ── PyGwalker ────────────────────────────────────────────────────────────

    def _on_pygwalker_open(self) -> None:
        """Launch PyGwalker in the browser for the currently selected table."""
        if self._dm is None or not self._current_table:
            return
        df = self._dm.get_table_df(self._current_table)
        if df is None or df.empty:
            return
        df = df.copy()
        if self._metadata_merged is not None and "well" in df.columns:
            df = self._join_metadata(df)

        def _serve() -> None:
            from pygwalker.api.webserver import walk
            try:
                walk(
                    df,
                    gid="pgw",
                    theme_key="g2",
                    appearance="media",
                    show_cloud_tool=False,
                    kernel_computation=None,
                    cloud_computation=False,
                    default_tab="vis",
                    auto_open=True,
                    auto_shutdown=True,
                )
            except Exception:
                import traceback
                traceback.print_exc()

        threading.Thread(target=_serve, daemon=True).start()

    # ── Grid Handlers ────────────────────────────────────────────────────────

    def _on_grid_params_changed(self) -> None:
        self._update_grid()

    def _on_well_clicked(self, well: str) -> None:
        is_real = well in set(self._dm.get_wells())
        if well in self._selected_wells:
            self._selected_wells.discard(well)
        else:
            self._selected_wells.add(well)
        self._update_grid()
        if is_real:
            self._schedule_image_refresh()

    def _on_select_all(self) -> None:
        if self._dm is None:
            return
        self._selected_wells = set(self._dm.get_wells())
        self._update_grid()
        self._schedule_image_refresh()

    def _on_clear_selection(self) -> None:
        self._selected_wells.clear()
        self._update_grid()
        self._schedule_image_refresh()

    def _update_grid(self) -> None:
        if self._dm is None:
            return
        gw = self._grid_controls
        col_data = gw.column.currentData()
        if col_data and len(col_data) == 3:
            table_name, col_name, is_num = col_data
            col_val = (col_name, is_num)
        else:
            table_name = ""
            col_val = (None, False)

        # Pre-compute metadata data_map for grid (metadata isn't in the DB)
        metadata_map: dict[str, float | str] | None = None
        if (
            table_name == "metadata"
            and self._metadata_merged is not None
            and col_name in self._metadata_merged.columns
        ):
            metadata_map = dict(
                zip(self._metadata_merged["well"], self._metadata_merged[col_name], strict=True)
            )

        self._grid_canvas.update_grid(
            self._dm,
            table_name=table_name,
            col_val=col_val,
            agg=gw.aggregation.currentText(),
            cmap=gw.colormap.currentText(),
            palette=gw.palette.currentText(),
            fmt_name=gw.plate_format.currentText(),
            selected_wells=self._selected_wells,
            metadata_map=metadata_map,
        )

    # ── Image Handlers ───────────────────────────────────────────────────────

    def _on_image_filter_changed(self) -> None:
        self._schedule_image_refresh()

    def _on_auto_all(self) -> None:
        if self._dm is None:
            return
        ic = self._image_controls
        low_pct = ic.auto_low.value()
        high_pct = ic.auto_high.value()
        for ch, cfg in self._ch_config.items():
            if cfg.get("enabled", True):
                p_lo, p_hi = self._auto_range(low_pct, high_pct, ch_name=ch)
                cfg["vmin"] = p_lo
                cfg["vmax"] = p_hi
        self._image_controls.update_channel_values(self._ch_config)
        self._schedule_image_refresh()

    def _on_reset(self) -> None:
        if self._dm is None:
            return
        ic = self._image_controls
        # Reset Low/High to defaults
        ic.auto_low.blockSignals(True)
        ic.auto_high.blockSignals(True)
        ic.auto_low.setValue(0.01)
        ic.auto_high.setValue(99.99)
        ic.auto_low.blockSignals(False)
        ic.auto_high.blockSignals(False)
        # Re-init channel config to defaults
        self._init_channel_config()
        ic.set_channels(self._ch_config)
        # Reset image zoom
        self._image_display.reset_all_zoom()
        self._schedule_image_refresh()

    def _on_contrast_changed(self, method: str) -> None:
        self._contrast_method = method
        self._invert = (method == "invert")
        self._image_controls.set_gamma_visible(method == "gamma")
        self._schedule_image_refresh()

    def _on_gamma_changed(self, value: int) -> None:
        self._contrast_gamma = value / 100.0  # slider 10–300 → 0.1–3.0
        self._schedule_image_refresh()

    def _on_overlay_changed(self) -> None:
        ic = self._image_controls
        col_data = ic.overlay_col.currentData()
        if col_data and len(col_data) == 2:
            self._overlay_table, self._overlay_col = col_data
        else:
            self._overlay_table, self._overlay_col = None, None
        self._overlay_cmap = ic.overlay_cmap.currentText()
        self._overlay_alpha = ic.overlay_alpha.value() / 100.0
        self._schedule_image_refresh()

    def _schedule_image_refresh(self) -> None:
        self._debounce.start()

    def _cancel_workers(self) -> None:
        """Invalidate all pending background workers."""
        self._gen += 1
        self._pending_workers = 0

    def _cache_put(self, row_idx: int, data: tuple) -> None:
        """Add to LRU cache, evicting oldest if full."""
        if row_idx in self._raw_cache:
            self._raw_cache.move_to_end(row_idx)
        else:
            if len(self._raw_cache) >= self._raw_cache_max:
                self._raw_cache.popitem(last=False)
            self._raw_cache[row_idx] = data

    def _detect_change(self) -> str:
        """Compare current state to last state. Returns change category."""
        ic = self._image_controls
        new_state = {
            "wells": frozenset(self._selected_wells),
            "fields": tuple(ic.get_selected_fields()),
            "stacks": tuple(ic.get_selected_stacks()),
            "tps": tuple(ic.get_selected_tps()),
            "ch_config": str(sorted(self._ch_config.items())),
            "contrast": self._contrast_method,
            "gamma": self._contrast_gamma,
            "invert": self._invert,
            "overlay_col": self._overlay_col,
            "overlay_table": self._overlay_table,
            "overlay_cmap": self._overlay_cmap,
            "overlay_alpha": self._overlay_alpha,
            "sort_by_row": ic.sort_by_row.isChecked(),
            "thumb_size": int(ic.image_size.value()),
        }
        old = self._last_state
        self._last_state = new_state

        if not old:
            return "filters"

        if (old.get("wells") != new_state["wells"]
                or old.get("fields") != new_state["fields"]
                or old.get("stacks") != new_state["stacks"]
                or old.get("tps") != new_state["tps"]):
            return "filters"

        if (old.get("ch_config") != new_state["ch_config"]
                or old.get("contrast") != new_state["contrast"]
                or old.get("gamma") != new_state["gamma"]
                or old.get("invert") != new_state["invert"]):
            return "contrast"

        if (old.get("overlay_col") != new_state["overlay_col"]
                or old.get("overlay_table") != new_state["overlay_table"]
                or old.get("overlay_cmap") != new_state["overlay_cmap"]
                or old.get("overlay_alpha") != new_state["overlay_alpha"]):
            return "overlay"

        if old.get("sort_by_row") != new_state["sort_by_row"]:
            return "sort"

        if old.get("thumb_size") != new_state["thumb_size"]:
            return "image_size"

        return "none"

    def _refresh_images(self) -> None:
        if self._dm is None:
            return

        change = self._detect_change()
        if change == "none":
            return

        ic = self._image_controls
        fields = ic.get_selected_fields()
        stacks = ic.get_selected_stacks()
        tps = ic.get_selected_tps()

        if not fields or not stacks or not tps:
            self._image_display.clear()
            self._raw_cache.clear()
            return

        self._ch_config = ic.get_channel_config()

        if not self._selected_wells:
            self._image_display.clear()
            self._raw_cache.clear()
            return

        thumb_size = int(ic.image_size.value())
        sort_by_row = ic.sort_by_row.isChecked()
        saved_state = self._image_display.save_view_state()

        if change == "sort":
            self._refresh_resort(thumb_size, saved_state, sort_by_row)
            return

        if change == "image_size":
            self._refresh_resize(thumb_size, saved_state, sort_by_row)
            return

        if change == "overlay" and self._raw_cache:
            self._refresh_overlay(thumb_size, saved_state, sort_by_row)
            return

        if change == "contrast" and self._raw_cache:
            self._refresh_from_raw(thumb_size, saved_state, sort_by_row)
            return

        # Full pipeline: filters changed or no cache available
        self._refresh_full(thumb_size, saved_state, sort_by_row)

    def _refresh_resort(self, thumb_size, saved_state, sort_by_row) -> None:
        """Sort changed — just rearrange cached results, no reprocessing."""
        self._image_display.resort_cached(
            thumb_size, self._overlay_alpha, self._overlay_cmap,
            saved_state, sort_by_row,
        )

    def _refresh_resize(self, thumb_size, saved_state, sort_by_row) -> None:
        """Thumbnail size changed — reprocess from raw cache at new size."""
        if self._raw_cache:
            self._refresh_from_raw(thumb_size, saved_state, sort_by_row)
        else:
            self._refresh_full(thumb_size, saved_state, sort_by_row)

    def _refresh_overlay(self, thumb_size, saved_state, sort_by_row) -> None:
        """Overlay changed — reprocess from raw cache (skip disk read)."""
        if self._raw_cache:
            self._refresh_from_raw(thumb_size, saved_state, sort_by_row)
        else:
            self._refresh_full(thumb_size, saved_state, sort_by_row)

    def _refresh_from_raw(self, thumb_size, saved_state, sort_by_row) -> None:
        """Contrast/overlay changed — use cached raw images, skip disk I/O."""
        self._cancel_workers()
        self._ch_config = self._image_controls.get_channel_config()

        wells = sorted(self._selected_wells)
        fields_int = [int(f) for f in self._image_controls.get_selected_fields()]
        stacks_int = [int(s) for s in self._image_controls.get_selected_stacks()]
        tps_int = [int(t) for t in self._image_controls.get_selected_tps()]
        rows_info = self._dm.lookup_row_indices(wells, fields_int, stacks_int, tps_int)
        from natsort import natsort_key
        rows_info.sort(key=lambda r: (natsort_key(str(r[1])), r[2], r[3], r[4]))

        if not rows_info:
            self._image_display.clear()
            return

        overlay_values, object_counts, per_object_values = self._compute_overlay_data()

        self._image_display.begin_results(thumb_size)
        self._pending_workers = len(rows_info)
        gen = self._gen
        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        need_polygons = self._overlay_col is not None and self._overlay_table is not None

        for row_idx, well, field, stack, tp in rows_info:
            raw_data = self._raw_cache.get(row_idx)
            if raw_data is None:
                try:
                    raw_data = self._dm.get_imageset(row_idx)
                    self._cache_put(row_idx, raw_data)
                except Exception:
                    _log.warning("Failed to load image row %d", row_idx, exc_info=True)
                    self._pending_workers = max(0, self._pending_workers - 1)
                    continue

            worker = _ImageWorker(
                row_idx, well, field, stack, tp,
                raw_data, thumb_size,
                channel_names, self._ch_config, dmax,
                self._contrast_method, self._contrast_gamma, self._invert,
                need_polygons,
                overlay_values.get(well), self._overlay_col,
                object_counts.get(well), per_object_values.get(well, {}),
                gen, sort_by_row,
            )
            worker.signals.finished.connect(self._on_worker_finished, Qt.QueuedConnection)
            worker.signals.error.connect(self._on_worker_error, Qt.QueuedConnection)
            self._thread_pool.start(worker)

    def _refresh_full(self, thumb_size, saved_state, sort_by_row) -> None:
        """Full pipeline: load from disk, process, display."""
        self._cancel_workers()
        self._raw_cache.clear()
        self._ch_config = self._image_controls.get_channel_config()

        wells = sorted(self._selected_wells)
        fields_int = [int(f) for f in self._image_controls.get_selected_fields()]
        stacks_int = [int(s) for s in self._image_controls.get_selected_stacks()]
        tps_int = [int(t) for t in self._image_controls.get_selected_tps()]
        rows_info = self._dm.lookup_row_indices(wells, fields_int, stacks_int, tps_int)
        from natsort import natsort_key
        rows_info.sort(key=lambda r: (natsort_key(str(r[1])), r[2], r[3], r[4]))

        if not rows_info:
            self._image_display.clear()
            return

        overlay_values, object_counts, per_object_values = self._compute_overlay_data()

        self._image_display.begin_results(thumb_size)
        self._pending_workers = len(rows_info)
        gen = self._gen
        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        need_polygons = self._overlay_col is not None and self._overlay_table is not None

        for row_idx, well, field, stack, tp in rows_info:
            try:
                raw_data = self._dm.get_imageset(row_idx)
                self._cache_put(row_idx, raw_data)
            except Exception:
                _log.warning("Failed to load image row %d", row_idx, exc_info=True)
                self._pending_workers = max(0, self._pending_workers - 1)
                continue

            worker = _ImageWorker(
                row_idx, well, field, stack, tp,
                raw_data, thumb_size,
                channel_names, self._ch_config, dmax,
                self._contrast_method, self._contrast_gamma, self._invert,
                need_polygons,
                overlay_values.get(well), self._overlay_col,
                object_counts.get(well), per_object_values.get(well, {}),
                gen, sort_by_row,
            )
            worker.signals.finished.connect(self._on_worker_finished, Qt.QueuedConnection)
            worker.signals.error.connect(self._on_worker_error, Qt.QueuedConnection)
            self._thread_pool.start(worker)

    def _compute_overlay_data(self) -> tuple:
        """Pre-compute overlay values, object counts, and per-object values."""
        overlay_values: dict[str, float | str] = {}
        if self._overlay_col and self._overlay_table:
            if self._overlay_table == "metadata" and self._metadata_merged is not None:
                if self._overlay_col in self._metadata_merged.columns:
                    meta = self._metadata_merged
                    overlay_values = dict(
                        zip(meta["well"], meta[self._overlay_col], strict=True)
                    )
            else:
                try:
                    overlay_values = self._dm.aggregate(
                        self._overlay_table, self._overlay_col, "mean"
                    )
                except Exception:
                    _log.warning("Failed to aggregate overlay values", exc_info=True)

        object_counts: dict[str, int] = {}
        object_table: str | None = None
        for tname in self._dm.get_profiling_tables():
            cols = [c for c, _, _ in self._dm.get_profiling_columns(tname)]
            if "label" in cols and "well" in cols:
                object_table = tname
                try:
                    df = self._dm.get_table_df(tname)
                    if df is not None:
                        object_counts = df.groupby("well")["label"].nunique().to_dict()
                except Exception:
                    _log.warning("Failed to compute object counts", exc_info=True)
                break

        per_object_values: dict[str, dict[int, float | str]] = {}
        if object_table and self._overlay_col:
            try:
                odf = self._dm.get_table_df(object_table)
                if odf is not None and "label" in odf.columns and self._overlay_col in odf.columns:
                    for well, wdf in odf.groupby("well"):
                        per_object_values[well] = dict(zip(
                            wdf["label"].astype(int), wdf[self._overlay_col], strict=True
                        ))
            except Exception:
                _log.warning("Failed to compute per-object overlay values", exc_info=True)

        return overlay_values, object_counts, per_object_values

    def _on_worker_finished(self, result: dict) -> None:
        """Called on main thread when one image is processed."""
        if self._shutting_down:
            return
        # Discard stale results from previous generations
        if result.get("gen") != self._gen:
            return
        self._pending_workers = max(0, self._pending_workers - 1)
        self._image_display.add_result(
            result, result["thumb_size"], self._overlay_alpha,
            self._overlay_cmap, None, result["sort_by_row"],
        )

    def _on_worker_error(self, msg: str) -> None:
        if self._shutting_down:
            return
        self._pending_workers = max(0, self._pending_workers - 1)
        _log.warning("Image worker error: %s", msg)

    def _on_pixel_clicked(self, well: str, field: int, stack: int, tp: int, x: int, y: int) -> None:
        if self._dm is None:
            return
        try:
            rows = self._dm.lookup_row_indices([well], [field], [stack], [tp])
            if not rows:
                return
            row_idx = rows[0][0]
            img_data, _ = self._dm.get_imageset(row_idx)
            channels = self._dm.channels
            parts = [f"{well} f{field} z{stack} t{tp} @ ({x},{y})"]
            for i, ch in enumerate(channels):
                if i < img_data.shape[2]:
                    val = img_data[y, x, i]
                    parts.append(f"| {ch}: {val:.1f}")
            self._pixel_info.set_text("  ".join(parts))
        except Exception:
            _log.warning("Failed to read pixel info", exc_info=True)

    # ── Data View Handler ────────────────────────────────────────────────────

    def _on_data_table_changed(self, table_name: str) -> None:
        if self._dm is None or not table_name:
            return
        self._current_table = table_name
        try:
            df = self._dm.get_table_df(table_name)
            if df is not None and self._metadata_merged is not None and "well" in df.columns:
                df = self._join_metadata(df)
            self._data_view.set_dataframe(df)
        except Exception:
            _log.warning("Failed to load data table %s", table_name, exc_info=True)

    def _join_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        meta_cols = [c for c in self._metadata_merged.columns if c != "well"]
        merged = df.merge(self._metadata_merged, on="well", how="left")
        other_cols = [c for c in merged.columns if c not in meta_cols and c != "well"]
        return merged[["well"] + meta_cols + other_cols]

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._shutting_down = True
        self._cancel_workers()
        self._thread_pool.waitForDone(3000)
        if self._dm is not None:
            self._dm.close()
        super().closeEvent(event)
