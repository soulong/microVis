from __future__ import annotations

import threading
from collections import OrderedDict
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
from microVis.widgets.model_controls import ModelControls
from microVis.widgets.model_view import ModelView

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

        # Model training state
        self._model_records: list | None = None
        self._model_summary: dict | None = None
        self._model_state: dict | None = None
        self._model_config: object | None = None
        self._train_worker = None

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

        self._nav_model = RotatedLabel("Model")
        self._nav_model.setProperty("class", "nav-tab")
        self._nav_model.setProperty("active", "false")
        _model_font = self._nav_model.font()
        _model_font.setBold(True)
        self._nav_model.setFont(_model_font)

        nav_layout.addWidget(self._nav_data)
        nav_layout.addWidget(self._nav_plate)
        nav_layout.addWidget(self._nav_model)
        nav_layout.addStretch()

        # ── Stacked content ──
        self._stack_content = QStackedWidget()

        # Page 0: Data View
        self._data_view = DataView()
        self._stack_content.addWidget(self._data_view)

        # Page 1: Plate & Images
        self._stack_content.addWidget(self._build_plate_images_tab())

        # Page 2: Model Training
        self._stack_content.addWidget(self._build_model_tab())

        self._nav_data.clicked.connect(lambda: self._switch_tab(0))
        self._nav_plate.clicked.connect(lambda: self._switch_tab(1))
        self._nav_model.clicked.connect(lambda: self._switch_tab(2))

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
        self._nav_model.setProperty("active", index == 2)
        for w in (self._nav_data, self._nav_plate, self._nav_model):
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

    def _build_model_tab(self) -> QWidget:
        """Build the Model training tab with sidebar controls and wizard area."""
        splitter = QSplitter(Qt.Horizontal)

        self._model_controls = ModelControls()
        self._model_view = ModelView()

        splitter.addWidget(self._model_controls)
        splitter.addWidget(self._model_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 800])

        return splitter

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

        # Model controls
        mc = self._model_controls
        mc.prepare_clicked.connect(self._on_model_prepare)
        mc.train_clicked.connect(self._on_model_train)
        mc.stop_clicked.connect(self._on_model_stop)
        mc.save_model_clicked.connect(self._on_model_save)
        mc.apply_model_clicked.connect(self._on_model_apply)
        mc.save_results_clicked.connect(self._on_model_save_results)

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

        try:
            self._dataset_dir = str(p)
            self._data_view.set_dataset_label(str(p))

            # Reset UI state for new dataset
            self._image_display.clear()
            self._raw_cache.clear()
            self._last_state.clear()
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
            self._populate_model_controls()

            # Initial render
            self._update_grid()
            self._schedule_image_refresh()
        except Exception:
            _log.exception("Failed to initialize UI for dataset %s", p)

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
        """Compute percentile range across currently displayed images."""
        dm = self._dm
        ic = self._image_controls
        wells = sorted(self._selected_wells) if self._selected_wells else dm.get_wells()
        fields = [int(f) for f in ic.get_selected_fields()] or dm.get_fields()
        stacks = [int(s) for s in ic.get_selected_stacks()] or dm.get_stacks()
        tps = [int(t) for t in ic.get_selected_tps()] or dm.get_timepoints()
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
                    raw_data = self._raw_cache.get(row_idx)
                    if raw_data is None:
                        raw_data = dm.get_imageset(row_idx)
                        self._cache_put(row_idx, raw_data)
                    img_data, _ = raw_data
                    if ch_idx < img_data.shape[2]:
                        samples.append(img_data[:, :, ch_idx].ravel())
                all_pixels = np.concatenate(samples)
            else:
                samples = []
                for row_idx, _, _, _, _ in rows:
                    raw_data = self._raw_cache.get(row_idx)
                    if raw_data is None:
                        raw_data = dm.get_imageset(row_idx)
                        self._cache_put(row_idx, raw_data)
                    img_data, _ = raw_data
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
            db_path = str(Path(self._dm.root_dir) / "results.db")
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
            self._image_display.resort_cached(
                thumb_size, self._overlay_alpha, self._overlay_cmap,
                saved_state, sort_by_row,
            )
            return

        # For overlay/contrast/resize changes with cache available, skip disk I/O
        use_cache = (
            change in ("overlay", "contrast", "image_size")
            and bool(self._raw_cache)
        )
        if not use_cache:
            self._raw_cache.clear()

        self._dispatch_image_workers(thumb_size, saved_state=saved_state)

    def _dispatch_image_workers(self, thumb_size: int, saved_state: dict | None = None) -> None:
        """Load images and dispatch background workers for processing."""
        self._cancel_workers()
        self._ch_config = self._image_controls.get_channel_config()
        self._saved_state = saved_state

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
        sort_by_row = self._image_controls.sort_by_row.isChecked()

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

            config = ImageWorkerConfig(
                row_idx=row_idx, well=well, field=field, stack=stack, tp=tp,
                raw_data=raw_data, thumb_size=thumb_size,
                channel_names=channel_names, ch_config=self._ch_config, dmax=dmax,
                contrast_method=self._contrast_method,
                contrast_gamma=self._contrast_gamma, invert=self._invert,
                need_polygons=need_polygons,
                overlay_val=overlay_values.get(well),
                overlay_col=self._overlay_col,
                n_objects=object_counts.get(well),
                obj_values=per_object_values.get(well, {}),
                gen=gen, sort_by_row=sort_by_row,
            )
            worker = ImageWorker(config)
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
        saved_state = getattr(self, "_saved_state", None)
        self._image_display.add_result(
            result, result["thumb_size"], self._overlay_alpha,
            self._overlay_cmap, saved_state, result["sort_by_row"],
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
            self, "Save Labels",
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
                self._cache_put(row_idx, raw_data)
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
        if object_mode == "All displayed":
            wells = sorted(self._selected_wells) if self._selected_wells else self._dm.get_wells()
            fields = [int(f) for f in ic.get_selected_fields()]
            stacks = [int(s) for s in ic.get_selected_stacks()]
            tps = [int(t) for t in ic.get_selected_tps()]
        else:  # "All images" or "All annotated"
            wells = self._dm.get_wells()
            fields = self._dm.get_fields()
            stacks = self._dm.get_stacks()
            tps = self._dm.get_timepoints()

        # Disable UI during export
        ic.set_export_enabled(False)
        ic.set_status("Exporting objects...")

        # Run export in background thread
        self._export_gen = getattr(self, "_export_gen", 0) + 1
        gen = self._export_gen

        from microVis.worker import ObjectExportWorker

        worker = ObjectExportWorker(
            dm=self._dm,
            wells=wells,
            fields=fields,
            stacks=stacks,
            timepoints=tps,
            mask_name=mask_name,
            channel_names=self._dm.channels,
            save_dir=str(save_path),
            object_mode=object_mode,
            channel_mode=channel_mode,
            annotations=annotations,
            gen=gen,
        )
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self._thread_pool.start(worker)

    def _on_export_progress(self, current: int, total: int) -> None:
        """Update export progress."""
        self._image_controls.set_status(f"Exporting: {current}/{total} images...")

    def _on_export_finished(self, result: dict) -> None:
        """Handle export completion."""
        self._image_controls.set_export_enabled(True)
        count = result.get("count", 0)
        save_dir = result.get("save_dir", "")
        self._image_controls.set_status(f"Exported {count} objects")
        _log.info("Exported %d objects to %s", count, save_dir)

    def _on_export_error(self, msg: str) -> None:
        """Handle export error."""
        self._image_controls.set_export_enabled(True)
        self._image_controls.set_status(f"Export error: {msg}")
        _log.warning("Export error: %s", msg)

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

    # ── Model Tab Handlers ────────────────────────────────────────────────────

    def _populate_model_controls(self) -> None:
        """Populate model controls when dataset is loaded."""
        if self._dm is None:
            return
        mc = self._model_controls
        mc.set_masks(self._dm.mask_names)
        mc.set_gpu_info(self._detect_gpu())
        mc.set_state("idle")
        self._model_view.clear()

    def _detect_gpu(self) -> str:
        """Detect available GPU devices."""
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                lines = [f"CUDA available: {n} device(s)"]
                for i in range(n):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
                    lines.append(f"  [{i}] {name} ({mem:.1f} GB)")
                return "\n".join(lines)
            return "CUDA not available — using CPU"
        except ImportError:
            return "PyTorch not installed"

    def _on_model_prepare(self) -> None:
        """Prepare dataset for training."""
        if self._dm is None:
            return

        config = self._model_controls.get_config()
        mask_name = config["mask_name"]
        if not mask_name:
            self._model_controls.set_status("Please select a mask.")
            return

        mode = config["mode"]
        crop_size = config["crop_size"]
        channels = self._dm.channels

        # Get annotations for supervised mode
        annotations = None
        if mode == "supervised":
            if config["source"] == "Image tab annotations":
                annotations = self._label_panel.get_annotations()
                if not annotations:
                    self._model_controls.set_status(
                        "No annotations found. Create classes and annotate objects in the Image tab first."
                    )
                    return
            # else: all objects, no labels — annotations stays None for SSL

        # Get wells to use
        wells = sorted(self._selected_wells) if self._selected_wells else self._dm.get_wells()
        fields = self._dm.get_fields()
        stacks = self._dm.get_stacks()
        tps = self._dm.get_timepoints()

        self._model_controls.set_status("Preparing dataset...")
        self._model_controls.set_state("idle")

        # Run preparation in background
        from microVis.processing.model_dataset import prepare_dataset

        try:
            records, summary = prepare_dataset(
                self._dm, wells, fields, stacks, tps,
                mask_name, channels, crop_size, annotations,
            )

            if not records:
                self._model_controls.set_status("No objects found for training.")
                return

            self._model_records = records
            self._model_summary = {
                "total_objects": summary.total_objects,
                "class_counts": summary.class_counts,
                "n_classes": summary.n_classes,
                "train_count": summary.train_count,
                "val_count": summary.val_count,
                "test_count": summary.test_count,
                "crop_size": summary.crop_size,
                "n_channels": summary.n_channels,
                "channel_names": summary.channel_names,
                "has_labels": summary.has_labels,
            }

            # Generate sample crops for preview
            sample_crops = self._generate_sample_crops(records[:16], crop_size)

            # Update view
            self._model_view.set_preview_data(self._model_summary, sample_crops)
            self._model_view.set_step(0)

            # Update num_classes in config
            config["num_classes"] = summary.n_classes

            self._model_controls.set_state("prepared")
            self._model_controls.set_status(
                f"Prepared: {summary.total_objects} objects, "
                f"{summary.n_classes} classes, "
                f"Train={summary.train_count} Val={summary.val_count} Test={summary.test_count}"
            )

        except Exception as e:
            _log.exception("Failed to prepare model dataset")
            self._model_controls.set_status(f"Error: {e}")

    def _generate_sample_crops(self, records, crop_size):
        """Generate sample crop thumbnails for preview."""
        from microVis.processing.model_dataset import extract_object_crops
        import numpy as np

        crops = []
        seen_rows = set()
        for rec in records:
            if rec.row_idx in seen_rows:
                continue
            seen_rows.add(rec.row_idx)
            try:
                img_data, mask_dict = self._dm.get_imageset_with_masks(
                    rec.row_idx, channels=self._dm.channels,
                    masks=[f"mask_{self._model_controls.get_config()['mask_name']}"],
                )
                mask = mask_dict.get(f"mask_{self._model_controls.get_config()['mask_name']}")
                if mask is None:
                    continue
                results = extract_object_crops(
                    img_data, mask, self._dm.channels, crop_size,
                )
                for crop_tensor, label_id, bbox in results[:4]:
                    crops.append(crop_tensor)
                if len(crops) >= 16:
                    break
            except Exception:
                continue
        return crops[:16]

    def _on_model_train(self) -> None:
        """Start model training."""
        if self._model_records is None:
            return

        config = self._model_controls.get_config()
        self._model_controls.set_state("training")
        self._model_view.set_step(1)
        self._model_view.init_training_charts()

        # Build train/val record lists
        train_records = [r for r in self._model_records if r.split == "train"]
        val_records = [r for r in self._model_records if r.split == "val"]

        from microVis.processing.model_worker import TrainConfig, TrainWorker

        train_config = TrainConfig(
            mode=config["mode"],
            backbone=config["backbone"],
            num_classes=config.get("num_classes", 2),
            in_channels=len(self._dm.channels),
            crop_size=config["crop_size"],
            pretrained=config["pretrained"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            optimizer=config["optimizer"],
            scheduler=config["scheduler"],
            early_stopping_patience=config["early_stopping_patience"],
            ssl_method=config["ssl_method"],
            ssl_temperature=config["ssl_temperature"],
            device=config["device"],
        )

        self._train_worker = TrainWorker(
            train_config, train_records, val_records,
            self._dm, config["mask_name"], self._dm.channels,
        )

        # Connect signals
        w = self._train_worker
        w.signals.epoch_done.connect(self._on_train_epoch_done)
        w.signals.progress.connect(self._on_train_progress)
        w.signals.log_message.connect(self._on_train_log)
        w.signals.finished.connect(self._on_train_finished)
        w.signals.error.connect(self._on_train_error)

        self._on_train_log("Starting training...")
        self._thread_pool.start(w)

    def _on_model_stop(self) -> None:
        """Request training stop."""
        if self._train_worker is not None:
            self._train_worker.request_stop()
            self._on_train_log("Stop requested...")

    def _on_train_epoch_done(self, epoch: int, metrics: dict) -> None:
        """Update UI with per-epoch metrics."""
        self._model_view.update_epoch(epoch, metrics)

    def _on_train_progress(self, current: int, total: int) -> None:
        """Update progress bar."""
        self._model_view.update_progress(current, total)

    def _on_train_log(self, message: str) -> None:
        """Append to training log."""
        self._model_view.append_log(message)

    def _on_train_finished(self, result: dict) -> None:
        """Handle training completion."""
        self._model_state = result.get("model_state")
        self._model_config = result.get("config")
        self._train_worker = None

        self._model_controls.set_state("trained")
        self._model_controls.set_status(
            f"Training complete. Best val_acc={result.get('best_val_acc', 0):.3f}"
        )

        # Show results
        self._model_view.set_step(2)

        config = self._model_config
        mode = getattr(config, "mode", None) or (config.get("mode") if isinstance(config, dict) else None)
        if mode == "supervised":
            # Run final validation for confusion matrix
            self._show_sl_results(result)
        else:
            self._show_ssl_results(result)

    def _show_sl_results(self, result: dict) -> None:
        """Display supervised learning results."""
        from microVis.processing.model_eval import compute_classification_metrics
        from microVis.processing.model_worker import _CropTorchDataset
        import numpy as np
        import torch

        config = result["config"]
        val_records = [r for r in self._model_records if r.split == "val"]

        if not val_records:
            return

        # Quick validation pass for confusion matrix
        ds = _CropTorchDataset(
            val_records, self._dm, self._model_controls.get_config()["mask_name"],
            self._dm.channels, config.crop_size, augment=False,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=False)

        model = None
        try:
            from microVis.processing.model_arch import create_sl_model
            model = create_sl_model(
                config.backbone, config.num_classes, config.in_channels, pretrained=False,
            )
            model.load_state_dict(result["model_state"])
            device = torch.device(config.device)
            model = model.to(device)
            model.eval()

            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            class_names = sorted(set(
                r.class_name for r in self._model_records if r.class_name
            ))
            if not class_names:
                class_names = [f"Class {i}" for i in range(config.num_classes)]

            metrics = compute_classification_metrics(
                np.array(all_labels), np.array(all_preds), class_names,
            )

            # Get sample crops for display
            sample_crops = []
            sample_preds = []
            for i, rec in enumerate(val_records[:12]):
                try:
                    ds_item = ds[i]
                    if isinstance(ds_item, tuple):
                        crop = ds_item[0].numpy()
                    else:
                        crop = ds_item.numpy()
                    sample_crops.append(crop)
                    pred_name = class_names[all_preds[i]] if all_preds[i] < len(class_names) else "?"
                    true_name = rec.class_name or "?"
                    sample_preds.append(f"P:{pred_name}\nT:{true_name}")
                except Exception:
                    pass

            self._model_view.set_results_sl(metrics, class_names, sample_crops, sample_preds)

        except Exception as e:
            _log.warning("Failed to generate SL results: %s", e)
            self._model_view._results_summary.setText(f"Results unavailable: {e}")

    def _show_ssl_results(self, result: dict) -> None:
        """Display SSL results with UMAP embedding plot."""
        import numpy as np

        config = result["config"]
        val_records = [r for r in self._model_records if r.split == "val"]

        if not val_records or self._model_state is None:
            return

        try:
            from microVis.processing.model_arch import create_embedding_model
            from microVis.processing.model_worker import _CropTorchDataset
            from microVis.processing.model_eval import compute_embedding_quality, reduce_embeddings
            import torch

            model = create_embedding_model(
                config.backbone, config.in_channels, pretrained=False,
            )
            model.load_state_dict(self._model_state)
            device = torch.device(config.device)
            model = model.to(device)
            model.eval()

            ds = _CropTorchDataset(
                val_records, self._dm, self._model_controls.get_config()["mask_name"],
                self._dm.channels, config.crop_size, augment=False,
            )
            loader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=False)

            all_embeddings = []
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    inputs = inputs.to(device)
                    features = model(inputs)
                    all_embeddings.append(features.cpu().numpy())

            embeddings = np.concatenate(all_embeddings, axis=0)

            # Get labels if available
            labels = None
            class_names = None
            if any(r.class_name for r in val_records):
                label_map = {r.class_name: i for i, r in enumerate(val_records) if r.class_name}
                class_names = sorted(label_map.keys())
                labels = np.array([
                    label_map.get(r.class_name, 0) for r in val_records
                ])

            # Reduce to 2D
            embeddings_2d = reduce_embeddings(embeddings, method="umap")

            # Compute quality metrics
            quality = compute_embedding_quality(embeddings, labels)

            self._model_view.set_results_ssl(
                embeddings_2d, labels, class_names, quality,
            )

        except Exception as e:
            _log.warning("Failed to generate SSL results: %s", e)
            self._model_view._results_summary.setText(f"Results unavailable: {e}")

    def _on_train_error(self, msg: str) -> None:
        """Handle training error."""
        self._train_worker = None
        self._model_controls.set_state("prepared")
        self._model_controls.set_status(f"Training error: {msg}")
        self._on_train_log(f"ERROR: {msg}")

    def _on_model_save(self) -> None:
        """Save the trained model to disk."""
        if self._model_state is None or self._model_config is None:
            return

        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import torch
        from datetime import datetime

        # Default save path
        default_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        default_dir = str(Path(self._dataset_dir) / "models") if self._dataset_dir else ""

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", str(Path(default_dir) / default_name),
            "PyTorch Model (*.pt)",
        )
        if not path:
            return

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            save_data = {
                "model_state": self._model_state,
                "config": self._model_config.__dict__,
                "channels": self._dm.channels,
                "mask_name": self._model_controls.get_config()["mask_name"],
            }
            torch.save(save_data, path)
            self._model_controls.set_status(f"Model saved to {path}")
            self._on_train_log(f"Model saved to {path}")
        except Exception as e:
            _log.exception("Failed to save model")
            QMessageBox.warning(self, "Save Failed", f"Failed to save model: {e}")

    def _on_model_apply(self) -> None:
        """Apply the trained model to data."""
        if self._model_state is None or self._model_config is None:
            return

        config = self._model_controls.get_config()
        mask_name = config["mask_name"]

        # Get records for inference
        wells = sorted(self._selected_wells) if self._selected_wells else self._dm.get_wells()
        fields = self._dm.get_fields()
        stacks = self._dm.get_stacks()
        tps = self._dm.get_timepoints()

        self._model_controls.set_status("Running inference...")

        from microVis.processing.model_dataset import prepare_dataset
        from microVis.processing.model_worker import InferenceWorker, TrainConfig

        try:
            # Use None annotations to get all objects
            records, _ = prepare_dataset(
                self._dm, wells, fields, stacks, tps,
                mask_name, self._dm.channels,
                self._model_config.crop_size, None,
            )

            if not records:
                self._model_controls.set_status("No objects found for inference.")
                return

            # Build TrainConfig from saved config
            cfg = self._model_config
            if isinstance(cfg, dict):
                train_config = TrainConfig(**cfg)
            else:
                train_config = cfg

            mode = "embed" if config["mode"] == "self_supervised" else "predict"

            worker = InferenceWorker(
                train_config, self._model_state, records,
                self._dm, mask_name, self._dm.channels, mode=mode,
            )
            worker.signals.progress.connect(self._on_train_progress)
            worker.signals.log_message.connect(self._on_train_log)
            worker.signals.finished.connect(self._on_inference_finished)
            worker.signals.error.connect(self._on_train_error)

            self._thread_pool.start(worker)

        except Exception as e:
            _log.exception("Failed to run inference")
            self._model_controls.set_status(f"Inference error: {e}")

    def _on_inference_finished(self, result) -> None:
        """Handle inference completion."""
        self._model_view.set_apply_results(result)
        self._model_view.set_step(3)
        self._model_controls.set_state("applied")
        self._model_controls.set_status(f"Inference complete: {len(result)} objects")

        # Store for DB save
        self._inference_result = result

    def _on_model_save_results(self) -> None:
        """Save inference results to database."""
        if not hasattr(self, "_inference_result") or self._inference_result is None:
            return

        from PySide6.QtWidgets import QMessageBox

        df = self._inference_result
        config = self._model_controls.get_config()

        if config["mode"] == "self_supervised":
            table_name = "model_embeddings"
        else:
            table_name = "model_predictions"

        reply = QMessageBox.question(
            self, "Save Results",
            f"Write {len(df)} rows to table '{table_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._dm.write_label_table(table_name, df)
            self._populate_data_controls()
            self._model_controls.set_status(f"Results saved to '{table_name}'")
        except Exception as e:
            _log.exception("Failed to save results")
            QMessageBox.warning(self, "Save Failed", f"Failed to save: {e}")

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._shutting_down = True
        self._cancel_workers()
        self._thread_pool.waitForDone(3000)
        if self._dm is not None:
            self._dm.close()
        super().closeEvent(event)
