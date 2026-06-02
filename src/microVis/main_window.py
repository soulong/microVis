from __future__ import annotations

import threading
from natsort import natsort_key
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
from microVis.widgets._event_filter import RotatedLabel
from microVis.widgets.data_view import DataView
from microVis.widgets.image_controls import ImageControls
from microVis.widgets.image_display import ImageDisplay
from microVis.widgets.label_annotation import LabelAnnotationPanel, ObjectKey
from microVis.widgets.pixel_info import PixelInfo
from microVis.widgets.well_grid_canvas import WellGridCanvas
from microVis.widgets.well_grid_controls import WellGridControls
from microVis.worker import CropWorker, ImageWorker, ImageWorkerConfig

_log = get_logger("microVis.main_window")




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
        self._raw_cache: dict[int, tuple] = {}  # row_idx → (img_data, mask_dict), unlimited, permanent until filter change
        self._mask_cache: dict[int, np.ndarray] = {}  # row_idx → downscaled mask
        self._polygon_cache: dict[int, list] = {}  # row_idx → extracted polygons
        self._gen: int = 0  # generation counter for cancelling stale workers
        self._last_state: dict = {}  # for change detection
        self._overlay_cache: tuple | None = None
        self._overlay_cache_key: str | None = None

        # PyGwalker server state
        self._pgw_httpd = None
        self._pgw_thread = None
        self._pgw_serving = False

        # Full-res zoom cache
        self._full_res_cache: dict[tuple, QPixmap] = {}  # (well, field, stack, tp) → QPixmap
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
        top_splitter.setSizes([280, 600])

        # ── Middle splitter: Image View ──
        middle_splitter = QSplitter(Qt.Horizontal)

        self._image_controls = ImageControls()
        self._image_display = ImageDisplay()

        middle_splitter.addWidget(self._image_controls)
        middle_splitter.addWidget(self._image_display)
        middle_splitter.setStretchFactor(0, 0)
        middle_splitter.setStretchFactor(1, 1)
        middle_splitter.setSizes([280, 600])

        # ── Label Annotation Panel ──
        self._label_panel = LabelAnnotationPanel()
        self._label_panel.setVisible(False)

        # ── Vertical splitter: grid + image + annotation ──
        self._v_splitter = QSplitter(Qt.Vertical)
        self._v_splitter.addWidget(top_splitter)
        self._v_splitter.addWidget(middle_splitter)
        self._v_splitter.addWidget(self._label_panel)
        self._v_splitter.setSizes([250, 750, 0])

        layout.addWidget(self._v_splitter)
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
        self._image_display.full_res_requested.connect(self._on_full_res_requested)

        # Label annotation controls
        ic.label_mask_selected.connect(self._on_label_mask_changed)
        ic.label_class_added.connect(self._on_label_class_added)
        ic.label_class_removed.connect(self._on_label_class_removed)
        ic.label_class_selection_changed.connect(self._on_label_class_selection_changed)
        ic.label_write_clicked.connect(self._on_label_write_to_db)

        # Object export
        ic.export_clicked.connect(self._on_export_clicked)

        # Label annotation panel crop requests
        self._label_panel.crop_requested.connect(self._on_crop_requested)

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

        # Flush message queue — DataModule init can be slow (disk I/O + SQLite)
        QApplication.processEvents()

        try:
            self._dataset_dir = str(p)
            self._data_view.set_dataset_label(str(p))

            # Reset UI state for new dataset
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._full_res_cache.clear()
            self._last_state.clear()
            self._overlay_cache = None
            self._overlay_cache_key = None
            self._label_panel.clear_all()
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
            self._populate_label_controls()

            # Pre-compute overlay data while DB is still open (avoids cold-start lag on first click)
            self._precompute_overlay_data()
            QApplication.processEvents()

            # Close DB connection — cached data remains available
            self._dm.close_db_only()

            # Initial render
            self._update_grid()
            self._schedule_image_refresh()
        except Exception:
            _log.exception("Failed to initialize UI for dataset %s", p)

    def _precompute_overlay_data(self) -> None:
        """Pre-load profiling tables into cache while DB is still open.

        This avoids expensive temporary SQLite connections on first well click.
        """
        try:
            for tname in self._dm.get_profiling_tables():
                self._dm.get_table_df(tname)  # populates _df_cache
            # Pre-compute overlay cache with current (empty) state
            self._compute_overlay_data()
        except Exception:
            _log.warning("Failed to pre-compute overlay data", exc_info=True)

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
        """Compute percentile range using downsampled images for speed."""
        from concurrent.futures import ThreadPoolExecutor

        from microVis.worker import _downscale_image

        dm = self._dm
        ic = self._image_controls
        wells = sorted(self._selected_wells) if self._selected_wells else dm.get_wells()
        fields = [int(f) for f in ic.get_selected_fields()] or dm.get_fields()
        stacks = [int(s) for s in ic.get_selected_stacks()] or dm.get_stacks()
        tps = [int(t) for t in ic.get_selected_tps()] or dm.get_timepoints()
        rows = dm.lookup_row_indices(wells, fields, stacks, tps)
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        p_lo, p_hi = 0.0, dmax
        if not rows:
            return p_lo, p_hi

        # Sample at most 16 rows for speed
        max_sample = 16
        if len(rows) > max_sample:
            step = len(rows) // max_sample
            rows = rows[::step][:max_sample]

        thumb_size = 128  # small for fast percentile computation
        try:
            ch_idx = -1
            if ch_name is not None:
                ch_idx = dm.channels.index(ch_name) if ch_name in dm.channels else -1
                if ch_idx < 0:
                    return p_lo, p_hi

            # Parallel load + downscale
            def _load_and_downscale(row_idx):
                raw_data = self._raw_cache.get(row_idx)
                if raw_data is None:
                    raw_data = dm.get_imageset(row_idx)
                    self._raw_cache[row_idx] = raw_data
                img_data, _ = raw_data
                return _downscale_image(img_data, thumb_size)

            with ThreadPoolExecutor(max_workers=4) as pool:
                thumbnails = list(pool.map(_load_and_downscale, [r[0] for r in rows]))

            samples = []
            for img_small in thumbnails:
                if ch_name is not None:
                    if ch_idx < img_small.shape[2]:
                        samples.append(img_small[:, :, ch_idx].ravel())
                else:
                    samples.append(img_small.ravel())

            if samples:
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

    def _populate_label_controls(self) -> None:
        """Populate mask dropdown in label annotation controls."""
        if self._dm is None:
            return
        self._image_controls.set_label_masks(self._dm.mask_names)
        self._label_panel.clear_all()
        # Initialize export "All annotated" option as disabled
        self._image_controls.update_export_annotated_option(False)

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
        self._overlay_cache = None

        self._refresh_data_table()
        self._update_overlay_with_metadata()

    def _on_metadata_clear(self) -> None:
        self._metadata_df = None
        self._metadata_merged = None
        self._overlay_cache = None
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
            for tname in tables:
                df = self._dm.get_table_df(tname)
                if df is not None and "well" in df.columns:
                    merged = self._join_metadata(df)
                    self._dm.write_merged_table(tname, merged)
            self._dm.invalidate_table_cache()

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

        # Shut down any previous PyGwalker server
        self._shutdown_pygwalker()

        df = self._dm.get_table_df(self._current_table)
        if df is None or df.empty:
            return
        df = df.copy()
        if self._metadata_merged is not None and "well" in df.columns:
            df = self._join_metadata(df)
        _log.info("PyGwalker: sending %d rows, columns=%s", len(df), list(df.columns))

        # Known profiling columns — always treat as categorical dimensions
        profiling_cols = {"directory", "well", "field", "stack", "timepoint", "label"}

        # Sample up to 200 rows per profiling group
        group_cols = [c for c in profiling_cols - {"label"} if c in df.columns]
        if group_cols and len(df) > 200:
            before = len(df)
            sampled_idx = (
                df.groupby(group_cols, group_keys=False)
                .apply(lambda g: g.sample(n=min(200, len(g)), random_state=42).index)
                .explode()
                .values
            )
            df = df.loc[sampled_idx].reset_index(drop=True)
            _log.info("PyGwalker: sampled %d → %d rows (max 200 per group)", before, len(df))

        try:
            # Disable PyGwalker network features to prevent SSL timeout on offline machines
            import os
            os.environ.setdefault("PYGWALKER_UPDATE_CHECK", "0")
            os.environ.setdefault("KANARIES_API_KEY", "")
            os.environ.setdefault("PYGWALKER_TELEMETRY", "0")
            from pygwalker.api.webserver import (
                BaseCommunication, CustomTCPServer, PygWalker,
                _GlobalState, _create_handler_with_walker, _open_browser, find_free_port,
            )

            from pygwalker.data_parsers.base import FieldSpec
            field_specs = [
                FieldSpec(fname=c, analytic_type="dimension" if c in profiling_cols else "?")
                for c in df.columns
            ]
            walker = PygWalker(
                gid="pgw",
                dataset=df,
                field_specs=field_specs,
                spec="",
                source_invoke_code="",
                theme_key="g2",
                appearance="media",
                show_cloud_tool=False,
                use_preview=False,
                kernel_computation=None,
                use_save_tool=True,
                gw_mode="explore",
                is_export_dataframe=True,
                kanaries_api_key="",
                default_tab="vis",
                cloud_computation=False,
            )
            walker._init_callback(BaseCommunication(str(walker.gid)))

            state = _GlobalState(auto_shutdown=False)
            handler = _create_handler_with_walker(walker, state)
            port = find_free_port()
            address = f"http://localhost:{port}"

            self._pgw_httpd = CustomTCPServer(("127.0.0.1", port), handler)
            self._pgw_serving = True

            def _serve():
                try:
                    with self._pgw_httpd:
                        threading.Thread(target=_open_browser, args=(address,), daemon=True).start()
                        self._pgw_httpd.serve_forever()
                except Exception:
                    _log.exception("PyGwalker server error")
                finally:
                    self._pgw_httpd = None
                    self._pgw_serving = False

            self._pgw_thread = threading.Thread(target=_serve)
            self._pgw_thread.start()

        except Exception:
            _log.exception("Failed to launch PyGwalker")

    def _shutdown_pygwalker(self) -> None:
        """Shut down the PyGwalker server if running."""
        if self._pgw_httpd is not None:
            try:
                self._pgw_httpd.shutdown()
            except Exception:
                pass
        if self._pgw_thread is not None and self._pgw_thread.is_alive():
            self._pgw_thread.join(timeout=3)
            self._pgw_thread = None
        self._pgw_httpd = None
        self._pgw_serving = False

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
            col_name = None
            col_val = (None, False)

        # Pre-compute metadata data_map for grid (metadata isn't in the DB)
        metadata_map: dict[str, float | str] | None = None
        if (
            table_name == "metadata"
            and col_name is not None
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
        from concurrent.futures import ThreadPoolExecutor

        from microVis.worker import _downscale_image

        ic = self._image_controls
        low_pct = ic.auto_low.value()
        high_pct = ic.auto_high.value()

        dm = self._dm
        channels = dm.channels
        enabled_chs = [ch for ch, cfg in self._ch_config.items() if cfg.get("enabled", True)]
        if not enabled_chs:
            return

        wells = sorted(self._selected_wells) if self._selected_wells else dm.get_wells()
        fields = [int(f) for f in ic.get_selected_fields()] or dm.get_fields()
        stacks = [int(s) for s in ic.get_selected_stacks()] or dm.get_stacks()
        tps = [int(t) for t in ic.get_selected_tps()] or dm.get_timepoints()
        rows = dm.lookup_row_indices(wells, fields, stacks, tps)
        if not rows:
            return

        max_sample = 16
        if len(rows) > max_sample:
            step = len(rows) // max_sample
            rows = rows[::step][:max_sample]

        thumb_size = 128
        ch_indices = {ch: channels.index(ch) for ch in enabled_chs if ch in channels}
        ch_samples: dict[str, list] = {ch: [] for ch in ch_indices}

        try:
            # Parallel load + downscale for uncached images
            def _load_and_downscale(row_idx):
                raw_data = self._raw_cache.get(row_idx)
                if raw_data is None:
                    raw_data = dm.get_imageset(row_idx)
                    self._raw_cache[row_idx] = raw_data
                img_data, _ = raw_data
                return _downscale_image(img_data, thumb_size)

            with ThreadPoolExecutor(max_workers=4) as pool:
                thumbnails = list(pool.map(_load_and_downscale, [r[0] for r in rows]))

            for img_small in thumbnails:
                for ch, idx in ch_indices.items():
                    if idx < img_small.shape[2]:
                        ch_samples[ch].append(img_small[:, :, idx].ravel())

            for ch, samples in ch_samples.items():
                if samples:
                    all_pixels = np.concatenate(samples)
                    self._ch_config[ch]["vmin"] = float(np.percentile(all_pixels, low_pct))
                    self._ch_config[ch]["vmax"] = float(np.percentile(all_pixels, high_pct))
        except Exception:
            _log.warning("Auto-range percentile computation failed", exc_info=True)

        self._image_controls.update_channel_values(self._ch_config)
        self._schedule_image_refresh()

    def _on_reset(self) -> None:
        if self._dm is None:
            return
        ic = self._image_controls
        # Reset Low/High to defaults
        ic.auto_low.blockSignals(True)
        ic.auto_high.blockSignals(True)
        ic.auto_low.setValue(0.5)
        ic.auto_high.setValue(99.5)
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

    def _detect_change(self) -> str:
        """Compare current state to last state. Returns change category."""
        ic = self._image_controls
        # Normalize ch_config so colors are always tuples (avoids list != tuple)
        normalized_ch = {
            ch: {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}
            for ch, cfg in self._ch_config.items()
        }
        new_state = {
            "wells": frozenset(self._selected_wells),
            "fields": tuple(ic.get_selected_fields()),
            "stacks": tuple(ic.get_selected_stacks()),
            "tps": tuple(ic.get_selected_tps()),
            "ch_config": normalized_ch,
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

        contrast_changed = (
            old.get("contrast") != new_state["contrast"]
            or old.get("gamma") != new_state["gamma"]
            or old.get("invert") != new_state["invert"]
        )
        ch_config_changed = old.get("ch_config") != new_state["ch_config"]

        if ch_config_changed or contrast_changed:
            # Distinguish channel-toggle (only enabled flags changed) from contrast change
            if not contrast_changed and ch_config_changed:
                old_cfg = old.get("ch_config", {})
                new_cfg = new_state["ch_config"]
                only_enabled_changed = all(
                    old_cfg.get(ch, {}).get("vmin") == new_cfg.get(ch, {}).get("vmin")
                    and old_cfg.get(ch, {}).get("vmax") == new_cfg.get(ch, {}).get("vmax")
                    for ch in new_cfg
                    if ch in old_cfg
                )
                if only_enabled_changed:
                    return "channel_toggle"
            return "contrast"

        if (old.get("overlay_col") != new_state["overlay_col"]
                or old.get("overlay_table") != new_state["overlay_table"]):
            return "overlay"

        if (old.get("overlay_cmap") != new_state["overlay_cmap"]
                or old.get("overlay_alpha") != new_state["overlay_alpha"]):
            return "overlay_styling"

        if old.get("sort_by_row") != new_state["sort_by_row"]:
            return "sort"

        if old.get("thumb_size") != new_state["thumb_size"]:
            return "image_size"

        return "none"

    def _refresh_images(self) -> None:
        if self._dm is None:
            return

        # Update channel config BEFORE change detection so toggles are detected
        self._ch_config = self._image_controls.get_channel_config()

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
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._full_res_cache.clear()
            return

        if not self._selected_wells:
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._full_res_cache.clear()
            return

        thumb_size = int(ic.image_size.value())
        sort_by_row = ic.sort_by_row.isChecked()
        saved_state = self._image_display.save_view_state()

        if change == "sort":
            self._image_display.resort_cached(
                thumb_size, self._overlay_alpha, self._overlay_cmap,
                saved_state, sort_by_row,
            )
            return

        if change == "overlay_styling" and self._polygon_cache:
            # Re-render cached polygons with new styling — no workers needed
            self._image_display.restyle_overlay(
                self._overlay_alpha, self._overlay_cmap, self._polygon_cache)
            return

        if change == "channel_toggle" and self._raw_cache:
            # Re-enhance from cached raw at thumbnail res — near-instant
            self._dispatch_image_workers(thumb_size, saved_state=saved_state,
                                         channel_toggle=True)
            return

        # For overlay/contrast/resize changes with cache available, skip disk I/O
        use_cache = (
            change in ("overlay", "contrast", "image_size")
            and bool(self._raw_cache)
        )
        if not use_cache:
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._full_res_cache.clear()

        self._dispatch_image_workers(thumb_size, saved_state=saved_state)

    def _dispatch_image_workers(self, thumb_size: int, saved_state: dict | None = None,
                                channel_toggle: bool = False) -> None:
        """Load images and dispatch background workers for processing."""
        self._cancel_workers()
        self._ch_config = self._image_controls.get_channel_config()
        self._saved_state = saved_state

        wells = sorted(self._selected_wells)
        fields_int = [int(f) for f in self._image_controls.get_selected_fields()]
        stacks_int = [int(s) for s in self._image_controls.get_selected_stacks()]
        tps_int = [int(t) for t in self._image_controls.get_selected_tps()]
        rows_info = self._dm.lookup_row_indices(wells, fields_int, stacks_int, tps_int)
        rows_info.sort(key=lambda r: (natsort_key(str(r[1])), r[2], r[3], r[4]))

        if not rows_info:
            self._image_display.clear()
            return

        # Use cached overlay data when overlay settings haven't changed
        meta_id = id(self._metadata_merged)
        overlay_key = f"{self._overlay_table}:{self._overlay_col}:{meta_id}"
        if self._overlay_cache is not None and self._overlay_cache_key == overlay_key:
            overlay_values, object_counts, per_object_values = self._overlay_cache
        else:
            overlay_values, object_counts, per_object_values = self._compute_overlay_data()
            self._overlay_cache = (overlay_values, object_counts, per_object_values)
            self._overlay_cache_key = overlay_key

        if channel_toggle:
            # Keep existing thumbnails visible — update in-place when workers finish
            self._pending_workers = len(rows_info)
            self._channel_toggle_results = []
        else:
            self._image_display.begin_results(thumb_size)
            self._pending_workers = len(rows_info)

        gen = self._gen
        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        need_polygons = self._overlay_col is not None and self._overlay_table is not None
        sort_by_row = self._image_controls.sort_by_row.isChecked()

        for row_idx, well, field, stack, tp in rows_info:
            raw_data = self._raw_cache.get(row_idx)

            # Use cached mask and polygons when available (skip re-extraction)
            mask_cache = self._mask_cache.get(row_idx)
            polygons_cache = self._polygon_cache.get(row_idx) if need_polygons else None

            config = ImageWorkerConfig(
                row_idx=row_idx, well=well, field=field, stack=stack, tp=tp,
                raw_data=raw_data, thumb_size=thumb_size,
                channel_names=channel_names, ch_config=self._ch_config, dmax=dmax,
                contrast_method=self._contrast_method,
                contrast_gamma=self._contrast_gamma, invert=self._invert,
                need_polygons=need_polygons,
                dm=self._dm,
                overlay_val=overlay_values.get(well),
                overlay_col=self._overlay_col,
                n_objects=object_counts.get(well),
                obj_values=per_object_values.get(well, {}),
                gen=gen, sort_by_row=sort_by_row,
                mask_cache=mask_cache,
                polygons_cache=polygons_cache,
            )
            worker = ImageWorker(config)
            worker.signals.finished.connect(
                self._on_worker_channel_toggle_finished if channel_toggle
                else self._on_worker_finished,
                Qt.QueuedConnection,
            )
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
        # Cache raw data if worker loaded from disk
        if "raw_data" in result:
            self._raw_cache[result["row_idx"]] = result["raw_data"]
        # Cache mask and polygons for overlay fast paths
        row_idx = result["row_idx"]
        if result.get("mask") is not None:
            self._mask_cache[row_idx] = result["mask"]
        if result.get("polygons") is not None:
            self._polygon_cache[row_idx] = result["polygons"]
        self._pending_workers = max(0, self._pending_workers - 1)
        saved_state = getattr(self, "_saved_state", None)
        self._image_display.add_result(
            result, result["thumb_size"], self._overlay_alpha,
            self._overlay_cmap, saved_state, result["sort_by_row"],
        )

    def _on_worker_channel_toggle_finished(self, result: dict) -> None:
        """Called on main thread when a channel-toggle worker completes."""
        if self._shutting_down:
            return
        if result.get("gen") != self._gen:
            return
        # Cache raw data if worker loaded from disk
        if "raw_data" in result:
            self._raw_cache[result["row_idx"]] = result["raw_data"]
        # Cache mask and polygons for overlay fast paths
        row_idx = result["row_idx"]
        if result.get("mask") is not None:
            self._mask_cache[row_idx] = result["mask"]
        if result.get("polygons") is not None:
            self._polygon_cache[row_idx] = result["polygons"]
        self._channel_toggle_results.append(result)
        self._pending_workers = max(0, self._pending_workers - 1)
        # When all workers done, update pixmaps in-place (no flash)
        if self._pending_workers == 0:
            # Collect thumbnails currently showing full-res
            full_res_keys = set()
            from microVis.widgets.image_display import _ThumbnailView
            for row_widget, _ in self._image_display._row_widgets.values():
                for thumb in row_widget.findChildren(_ThumbnailView):
                    if thumb._is_full_res and thumb._full_res_item is not None:
                        full_res_keys.add((thumb._well, thumb._field,
                                           thumb._stack, thumb._tp))

            self._image_display.update_pixmaps_in_place(
                self._channel_toggle_results,
                self._overlay_alpha, self._overlay_cmap,
                remove_full_res=False,  # keep full-res visible until re-composited
            )

            # Re-dispatch full-res workers for thumbnails that were zoomed in
            if full_res_keys:
                from microVis.widgets.image_display import _ThumbnailView
                for row_widget, _ in self._image_display._row_widgets.values():
                    for thumb in row_widget.findChildren(_ThumbnailView):
                        key = (thumb._well, thumb._field, thumb._stack, thumb._tp)
                        if key in full_res_keys:
                            thumb._full_res_gen += 1
                for key in full_res_keys:
                    self._on_full_res_requested(*key, gen=0)

            self._channel_toggle_results = []

    def _on_worker_error(self, msg: str) -> None:
        if self._shutting_down:
            return
        self._pending_workers = max(0, self._pending_workers - 1)
        _log.warning("Image worker error: %s", msg)

    def _on_pixel_clicked(self, well: str, field: int, stack: int, tp: int,
                           x: int, y: int, pixmap_w: int, pixmap_h: int) -> None:
        if self._dm is None:
            return
        try:
            rows = self._dm.lookup_row_indices([well], [field], [stack], [tp])
            if not rows:
                return
            row_idx = rows[0][0]
            img_data, _ = self._dm.get_imageset(row_idx)
            channels = self._dm.channels

            # Convert scene coordinates to raw image coordinates.
            # x, y are in pixmap pixel space (scene coords).
            # Scale by raw/pixmap ratio to get raw image coordinates.
            h_raw, w_raw = img_data.shape[:2]
            rx = min(int(x * w_raw / pixmap_w), w_raw - 1)
            ry = min(int(y * h_raw / pixmap_h), h_raw - 1)

            parts = [f"{well} f{field} z{stack} t{tp} @ ({rx},{ry})"]
            for i, ch in enumerate(channels):
                if i < img_data.shape[2]:
                    val = img_data[ry, rx, i]
                    parts.append(f"| {ch}: {val:.1f}")
            self._pixel_info.set_text("  ".join(parts))
        except Exception:
            _log.warning("Failed to read pixel info", exc_info=True)

    # ── Full-Res Zoom ───────────────────────────────────────────────────────

    def _on_full_res_requested(self, well: str, field: int, stack: int, tp: int,
                               gen: int = 0) -> None:
        """Load and display full-resolution image when user zooms past threshold."""
        if self._dm is None:
            return

        # Look up row index
        rows = self._dm.lookup_row_indices([well], [field], [stack], [tp])
        if not rows:
            return
        row_idx = rows[0][0]

        from microVis.worker import FullResWorker
        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)

        need_polygons = self._overlay_col is not None and self._overlay_table is not None
        obj_values = {}
        if need_polygons and self._overlay_cache is not None:
            _, _, per_object_values = self._overlay_cache
            obj_values = per_object_values.get(well, {})

        worker = FullResWorker(
            dm=self._dm, row_idx=row_idx,
            well=well, field=field, stack=stack, tp=tp,
            channel_names=channel_names, ch_config=self._ch_config,
            dmax=dmax,
            contrast_method=self._contrast_method,
            contrast_gamma=self._contrast_gamma,
            invert=self._invert,
            gen=gen,
            overlay_alpha=self._overlay_alpha,
            overlay_cmap=self._overlay_cmap,
            need_polygons=need_polygons,
            obj_values=obj_values,
            overlay_col=self._overlay_col,
        )
        worker.signals.finished.connect(self._on_full_res_finished)
        worker.signals.error.connect(
            lambda msg: _log.warning("Full-res worker error: %s", msg))
        self._thread_pool.start(worker)

    def _on_full_res_finished(self, pixmap, well: str, field: int,
                               stack: int, tp: int, gen: int,
                               mask=None, obj_values=None,
                               overlay_col="", overlay_cmap="") -> None:
        """Apply full-res pixmap to the matching thumbnail."""
        self._apply_full_res_pixmap(
            (well, field, stack, tp), pixmap, gen, mask,
            obj_values or {}, overlay_col)

    def _apply_full_res_pixmap(self, key: tuple, pixmap, gen: int = 0,
                               mask=None, obj_values=None,
                               overlay_col="") -> None:
        """Find the thumbnail widget and set its full-res pixmap."""
        from microVis.widgets.image_display import _ThumbnailView
        well, field, stack, tp = key
        for row_widget, _row_layout in self._image_display._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                if (thumb._well == well and thumb._field == field
                        and thumb._stack == stack and thumb._tp == tp):
                    thumb.set_full_res_pixmap(
                        pixmap, gen, mask=mask,
                        obj_values=obj_values, overlay_col=overlay_col)
                    return

    # ── Label Annotation Handlers ────────────────────────────────────────────

    def _on_label_mask_changed(self, mask_name: str) -> None:
        """Handle mask selection change in label controls."""
        pass  # Mask is used at crop time; no immediate action needed

    def _on_label_class_added(self, class_name: str) -> None:
        """Handle new class creation from sidebar."""
        self._label_panel.add_class(class_name)
        # Show class boxes panel on first class creation
        if not self._label_panel.isVisible():
            self._label_panel.setVisible(True)
            # Reallocate space: 25% well grid, 50% image view, 25% class boxes
            total = sum(self._v_splitter.sizes())
            self._v_splitter.setSizes([
                int(total * 0.25),
                int(total * 0.50),
                int(total * 0.25),
            ])
        # Update export "All annotated" option availability
        self._image_controls.update_export_annotated_option(True)

    def _on_label_class_removed(self, class_name: str) -> None:
        """Handle class deletion from sidebar."""
        self._label_panel.remove_class(class_name)
        # Hide panel if no classes remain
        if not self._label_panel.get_all_class_names():
            self._label_panel.setVisible(False)
            total = sum(self._v_splitter.sizes())
            self._v_splitter.setSizes([int(total * 0.30), int(total * 0.70), 0])
            # Update export "All annotated" option availability
            self._image_controls.update_export_annotated_option(False)

    def _on_label_class_selection_changed(self) -> None:
        """Handle change in which classes are selected for display."""
        selected = self._image_controls.get_selected_class_names()
        self._label_panel.set_visible_classes(selected)

    def _on_label_write_to_db(self) -> None:
        """Write all label annotations to the database."""
        if self._dm is None:
            return

        mask_name = self._image_controls.get_selected_label_mask()
        if not mask_name:
            return

        annotations = self._label_panel.get_annotations()
        if not annotations:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "No Annotations",
                "No labeled objects to write. Drag objects into class boxes first.",
            )
            return

        # Build DataFrame
        rows = []
        for class_name, keys in annotations.items():
            for key in keys:
                rows.append({
                    "well": key.well,
                    "field": key.field,
                    "stack": key.stack,
                    "tp": key.tp,
                    "label": key.label,
                    "class": class_name,
                })
        df = pd.DataFrame(rows)

        # Determine table name
        table_name = self._image_controls.get_label_table_name() or f"{mask_name}_label"

        # Confirmation dialog
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Write Label to DB",
            f"Write {len(df)} label annotations to table '{table_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._dm.write_label_table(table_name, df)
            _log.info("Wrote %d label annotations to '%s'", len(df), table_name)
            # Refresh data view to show new table
            self._populate_data_controls()
        except Exception:
            _log.exception("Failed to write label annotations")
            QMessageBox.warning(
                self, "Write Failed",
                "Failed to write label annotations to database. See log for details.",
            )

    def _on_crop_requested(self, key: ObjectKey, class_name: str) -> None:
        """Dispatch a CropWorker for a dropped object."""
        if self._dm is None:
            return

        mask_name = self._image_controls.get_selected_label_mask()
        if not mask_name:
            return

        # Look up row index
        rows = self._dm.lookup_row_indices(
            [key.well], [key.field], [key.stack], [key.tp]
        )
        if not rows:
            _log.warning("No row found for %s", key)
            return
        row_idx = rows[0][0]

        # Load raw image data
        raw_data = self._raw_cache.get(row_idx)
        if raw_data is None:
            try:
                raw_data = self._dm.get_imageset(row_idx)
                self._raw_cache[row_idx] = raw_data
            except Exception:
                _log.warning("Failed to load image for crop: row %d", row_idx, exc_info=True)
                return

        img_data, mask_dict = raw_data

        # Get the selected mask
        mask_full_name = f"mask_{mask_name}"
        mask = mask_dict.get(mask_full_name)
        if mask is None:
            # Try without prefix
            mask = mask_dict.get(mask_name)
        if mask is None and mask_dict:
            # Fall back to first available mask
            mask = next(iter(mask_dict.values()))
        if mask is None:
            _log.warning("No mask found for crop: %s", mask_name)
            return

        # Get current channel/contrast settings
        ch_config = self._image_controls.get_channel_config()
        channel_names = list(ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)

        # Dispatch crop worker
        worker = CropWorker(
            img_data=img_data,
            mask=mask,
            label=key.label,
            key=key,
            channel_names=channel_names,
            ch_config=ch_config,
            dmax=dmax,
            contrast_method=self._contrast_method,
            contrast_gamma=self._contrast_gamma,
            invert=self._invert,
            target_size=64,
            padding=4,
        )
        worker.signals.finished.connect(self._on_crop_finished, Qt.QueuedConnection)
        worker.signals.error.connect(
            lambda msg: _log.warning("Crop worker error: %s", msg),
            Qt.QueuedConnection,
        )
        self._thread_pool.start(worker)

    def _on_crop_finished(self, rgb: np.ndarray, key: ObjectKey) -> None:
        """Update the class box thumbnail with the cropped image."""
        if rgb is None:
            return
        # Convert numpy RGB to QPixmap on main thread
        from PySide6.QtGui import QImage, QPixmap
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        # Find which class box contains this key and update it
        for class_name in self._label_panel.get_all_class_names():
            box = self._label_panel.get_class_box(class_name)
            if box is not None and box.has_object(key):
                box.set_object_pixmap(key, pixmap)
                break

    # ── Object Export Handler ────────────────────────────────────────────────

    def _on_export_clicked(self) -> None:
        """Handle object export button click."""
        if self._dm is None:
            return

        ic = self._image_controls
        object_mode = ic.get_export_object_mode()
        channel_mode = ic.get_export_channel_mode()
        save_dir = ic.get_export_dir()

        # Determine mask name from export mask dropdown
        mask_name = ic.get_export_mask()
        if not mask_name:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", "No mask selected for object extraction.")
            return

        # Determine save directory
        if not save_dir:
            save_dir = str(Path(self._dataset_dir) / "objects_exported")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Get annotations if needed
        annotations = None
        if object_mode == "All annotated":
            annotations = self._label_panel.get_annotations()
            if not annotations:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "No Annotations",
                    "No annotated objects found. Create classes and annotate objects first.",
                )
                return

        # Get wells/fields based on mode
        annotated_keys = None  # set of (well, field, stack, tp) for "All annotated"
        if object_mode == "All displayed":
            wells = sorted(self._selected_wells) if self._selected_wells else self._dm.get_wells()
            fields = [int(f) for f in ic.get_selected_fields()]
            stacks = [int(s) for s in ic.get_selected_stacks()]
            tps = [int(t) for t in ic.get_selected_tps()]
        elif object_mode == "All annotated" and annotations:
            # Only scan images that contain annotated objects
            annotated_keys = set()
            for keys in annotations.values():
                for key in keys:
                    annotated_keys.add((key.well, key.field, key.stack, key.tp))
            wells = sorted({c[0] for c in annotated_keys})
            fields = sorted({c[1] for c in annotated_keys})
            stacks = sorted({c[2] for c in annotated_keys})
            tps = sorted({c[3] for c in annotated_keys})
        else:  # "All images"
            wells = self._dm.get_wells()
            fields = self._dm.get_fields()
            stacks = self._dm.get_stacks()
            tps = self._dm.get_timepoints()

        # Disable UI during export
        ic.set_export_enabled(False)
        self._pixel_info.set_text("Exporting objects...")

        # Run export in background thread
        self._export_gen = getattr(self, "_export_gen", 0) + 1
        gen = self._export_gen

        from microVis.worker import ObjectExportWorker

        max_objects = ic.get_export_max_objects()

        worker = ObjectExportWorker(
            dm=self._dm,
            wells=wells,
            fields=fields,
            stacks=stacks,
            timepoints=tps,
            mask_name=mask_name,
            channel_names=self._dm.channels,
            annotated_keys=annotated_keys,
            save_dir=str(save_path),
            object_mode=object_mode,
            channel_mode=channel_mode,
            annotations=annotations,
            gen=gen,
            max_objects_per_image=max_objects,
        )
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self._thread_pool.start(worker)

    def _on_export_progress(self, current: int, total: int) -> None:
        """Update export progress."""
        self._pixel_info.set_text(f"Exporting: {current}/{total} images...")

    def _on_export_finished(self, result: dict) -> None:
        """Handle export completion."""
        self._image_controls.set_export_enabled(True)
        count = result.get("count", 0)
        save_dir = result.get("save_dir", "")
        self._pixel_info.set_text(f"Exported {count} objects to {save_dir}")
        _log.info("Exported %d objects to %s", count, save_dir)

    def _on_export_error(self, msg: str) -> None:
        """Handle export error."""
        self._image_controls.set_export_enabled(True)
        self._pixel_info.set_text(f"Export error: {msg}")
        _log.warning("Export error: %s", msg)

    # ── Data View Handler ────────────────────────────────────────────────────

    def _on_data_table_changed(self, table_name: str) -> None:
        if self._dm is None or not table_name:
            return
        self._current_table = table_name
        try:
            # Fetch only 20 rows via SQL LIMIT — no full table scan
            df, total_rows = self._dm.get_table_preview(table_name, limit=20)
            if df is not None and self._metadata_merged is not None and "well" in df.columns:
                df = self._join_metadata(df)
            self._data_view.set_dataframe(df)
            self._data_view.set_preview_hint(total_rows)
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
        self._shutdown_pygwalker()
        self._cancel_workers()
        self._thread_pool.waitForDone(2000)
        # Flush Windows message queue to clear "Not Responding" state
        QApplication.processEvents()
        if self._dm is not None:
            self._dm.close()
        super().closeEvent(event)
