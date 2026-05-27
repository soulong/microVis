from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import (
    AGG_METHODS,
    CMAP_OPTIONS,
    CONTRAST_METHODS,
    DEFAULT_CHANNEL_COLORS,
    DEFAULT_CMAP,
    DEFAULT_PLATE,
    DTYPE_MAX,
    PLATE_FORMATS,
    QUALITATIVE_PALETTES,
)
from microVis.io.data_module import DataModule
from microVis.processing.compositing import composite_image
from microVis.processing.contrast import apply_contrast, invert_image
from microVis.processing.overlay import extract_polygons
from microVis.widgets.data_view import DataView
from microVis.widgets.folder_selector import FolderSelector
from microVis.widgets.image_controls import ImageControls
from microVis.widgets.image_display import ImageDisplay
from microVis.widgets.pixel_info import PixelInfo
from microVis.widgets.well_grid_canvas import WellGridCanvas
from microVis.widgets.well_grid_controls import WellGridControls


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
        self._image_fields: list[str] = []
        self._image_stacks: list[str] = []
        self._image_tps: list[str] = []
        self._contrast_method: str = "none"
        self._contrast_gamma: float = 1.0
        self._invert: bool = False
        self._overlay_mask: str = "None"
        self._overlay_col: str | None = None
        self._overlay_cmap: str = "Viridis"
        self._overlay_alpha: float = 0.4

        # Image debounce timer
        self._debounce = QTimer(singleShot=True, interval=300, timeout=self._refresh_images)

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

        # Header
        header = QWidget()
        header.setObjectName("header")
        hlayout = QHBoxLayout(header)
        hlayout.setContentsMargins(12, 6, 12, 6)
        self._title_label = QLabel("microVis")
        self._title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4cc9f0;")
        self._path_label = QLabel("")
        self._path_label.setProperty("class", "muted")
        hlayout.addWidget(self._title_label)
        hlayout.addWidget(self._path_label)
        hlayout.addStretch()

        # Stack: folder selector → main content
        self._stack = QStackedWidget()

        self._folder_page = FolderSelector()
        self._main_page = self._build_main_page()

        self._stack.addWidget(self._folder_page)  # index 0
        self._stack.addWidget(self._main_page)     # index 1
        self._stack.setCurrentIndex(0)

        root.addWidget(header)
        root.addWidget(self._stack, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()

        # Tab 0: Plate & Images
        plate_tab = self._build_plate_images_tab()
        self._tabs.addTab(plate_tab, "Plate && Images")

        # Tab 1: Data View
        self._data_view = DataView()
        self._tabs.addTab(self._data_view, "Data View")

        layout.addWidget(self._tabs, stretch=1)

        # Pixel info bar at bottom
        self._pixel_info = PixelInfo()
        layout.addWidget(self._pixel_info)

        return page

    def _build_plate_images_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Top splitter: Well Grid ──
        top_splitter = QSplitter(Qt.Horizontal)

        self._grid_controls = WellGridControls()
        self._grid_canvas = WellGridCanvas()

        top_splitter.addWidget(self._grid_controls)
        top_splitter.addWidget(self._grid_canvas)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([260, 600])

        # ── Bottom splitter: Image View ──
        bottom_splitter = QSplitter(Qt.Horizontal)

        self._image_controls = ImageControls()
        self._image_display = ImageDisplay()

        bottom_splitter.addWidget(self._image_controls)
        bottom_splitter.addWidget(self._image_display)
        bottom_splitter.setStretchFactor(0, 0)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([260, 600])

        # ── Vertical splitter between grid and image sections ──
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(top_splitter)
        v_splitter.addWidget(bottom_splitter)
        v_splitter.setSizes([300, 400])

        layout.addWidget(v_splitter)
        return tab

    # ── Signal Connections ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Folder selector
        self._folder_page.load_requested.connect(self._on_load_requested)

        # Well grid controls
        gw = self._grid_controls
        gw.plate_format.currentTextChanged.connect(self._on_grid_params_changed)
        gw.table.currentTextChanged.connect(self._on_grid_table_changed)
        gw.column.currentTextChanged.connect(self._on_grid_params_changed)
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
        ic.channel_config_changed.connect(self._schedule_image_refresh)
        ic.contrast.currentTextChanged.connect(self._on_contrast_changed)
        ic.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        ic.invert_button.toggled.connect(self._on_invert_toggled)
        ic.overlay_mask.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_col.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_cmap.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_alpha.valueChanged.connect(self._on_overlay_changed)

        # Image display (pixel click)
        self._image_display.pixel_clicked.connect(self._on_pixel_clicked)

        # Data view
        self._data_view.table_selector.currentTextChanged.connect(self._on_data_table_changed)

    # ── Dataset Loading ──────────────────────────────────────────────────────

    def _on_load_requested(self, path: str) -> None:
        self._load_dataset(path)

    def _load_dataset(self, path: str) -> None:
        p = Path(path)
        if not p.is_dir():
            self.statusBar().showMessage(f"Invalid directory: {path}")
            self._folder_page.show_error(f"Not a valid directory: {path}")
            return
        if not (p / "image").is_dir():
            self.statusBar().showMessage(f"No 'image/' subdirectory in: {path}")
            self._folder_page.show_error(f"No 'image/' subdirectory found in:\n{path}")
            return

        try:
            self._dm = DataModule(str(p))
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load dataset: {e}")
            self._folder_page.show_error(f"Failed to load dataset:\n{e}")
            return

        self._dataset_dir = str(p)
        self._path_label.setText(str(p))
        self._folder_page.clear_error()

        # Init selection
        all_wells = self._dm.get_wells()
        self._selected_wells = set(all_wells)

        # Init channel config
        self._init_channel_config()

        # Populate controls
        self._populate_grid_controls()
        self._populate_image_controls()
        self._populate_data_controls()

        # Show main page
        self._stack.setCurrentIndex(1)

        # Initial render
        self._update_grid()
        self._schedule_image_refresh()

        self.statusBar().showMessage(f"Loaded: {p.name}  |  {len(all_wells)} wells")

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

    def _auto_range(self, ch_name: str | None = None) -> tuple[float, float]:
        """Compute 1st–99th percentile range across a sample of images.

        Args:
            ch_name: If given, compute for a single channel. Otherwise across all pixels.
        """
        import numpy as np
        dm = self._dm
        wells = dm.get_wells()[:5]
        fields = dm.get_fields()[:1]
        stacks = dm.get_stacks()[:1]
        tps = dm.get_timepoints()[:1]
        rows = dm.lookup_row_indices(wells, fields, stacks, tps)
        p1, p99 = 0.0, 65535.0
        if not rows:
            return p1, p99
        try:
            if ch_name is not None:
                ch_idx = dm.channels.index(ch_name) if ch_name in dm.channels else -1
                if ch_idx < 0:
                    return p1, p99
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
            p1 = float(np.percentile(all_pixels, 1))
            p99 = float(np.percentile(all_pixels, 99))
        except Exception:
            pass
        return p1, p99

    # ── Populate Controls ────────────────────────────────────────────────────

    def _populate_grid_controls(self) -> None:
        gw = self._grid_controls
        gm = self._dm

        # Plate formats
        gw.plate_format.blockSignals(True)
        gw.plate_format.clear()
        gw.plate_format.addItems(list(PLATE_FORMATS.keys()))
        idx = gw.plate_format.findText(DEFAULT_PLATE)
        if idx >= 0:
            gw.plate_format.setCurrentIndex(idx)
        gw.plate_format.blockSignals(False)

        # Tables
        gw.table.blockSignals(True)
        gw.table.clear()
        tables = gm.get_profiling_tables()
        gw.table.addItems(list(tables.keys()))
        gw.table.blockSignals(False)

        if tables:
            self._update_grid_columns(list(tables.keys())[0])

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

    def _update_grid_columns(self, table_name: str) -> None:
        if self._dm is None:
            return
        cols = self._dm.get_profiling_columns(table_name)
        gw = self._grid_controls
        gw.column.blockSignals(True)
        gw.column.clear()
        for name, ctype, is_num in cols:
            tag = "[num]" if is_num else "[cat]"
            gw.column.addItem(f"{name} {tag}", (name, is_num))
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

        # Overlay masks
        masks = dm.mask_names or []
        ic.overlay_mask.blockSignals(True)
        ic.overlay_mask.clear()
        ic.overlay_mask.addItem("None")
        ic.overlay_mask.addItems(masks)
        ic.overlay_mask.blockSignals(False)

        # Overlay column
        ic.overlay_col.blockSignals(True)
        ic.overlay_col.clear()
        tables = dm.get_profiling_tables()
        for tname in tables:
            if tname in ("image", "metadata"):
                continue
            cols = dm.get_profiling_columns(tname)
            for name, ctype, is_num in cols:
                ic.overlay_col.addItem(f"{tname}/{name}", (tname, name))
        ic.overlay_col.blockSignals(False)

        ic.overlay_cmap.blockSignals(True)
        ic.overlay_cmap.clear()
        ic.overlay_cmap.addItems(CMAP_OPTIONS)
        ic.overlay_cmap.setCurrentText(DEFAULT_CMAP)
        ic.overlay_cmap.blockSignals(False)

    def _populate_data_controls(self) -> None:
        if self._dm is None:
            return
        tables = self._dm.get_profiling_tables()
        self._data_view.table_selector.blockSignals(True)
        self._data_view.table_selector.clear()
        self._data_view.table_selector.addItems(list(tables.keys()))
        self._data_view.table_selector.blockSignals(False)
        if tables:
            self._on_data_table_changed(list(tables.keys())[0])

    # ── Grid Handlers ────────────────────────────────────────────────────────

    def _on_grid_params_changed(self) -> None:
        self._update_grid()

    def _on_grid_table_changed(self, table_name: str) -> None:
        if table_name:
            self._update_grid_columns(table_name)
        self._update_grid()

    def _on_well_clicked(self, well: str) -> None:
        if well in self._selected_wells:
            self._selected_wells.discard(well)
        else:
            self._selected_wells.add(well)
        self._update_grid()
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
        col_val = col_data if col_data else (None, False)
        self._grid_canvas.update_grid(
            self._dm,
            table_name=gw.table.currentText(),
            col_val=col_val,
            agg=gw.aggregation.currentText(),
            cmap=gw.colormap.currentText(),
            palette=gw.palette.currentText(),
            fmt_name=gw.plate_format.currentText(),
            selected_wells=self._selected_wells,
        )

    # ── Image Handlers ───────────────────────────────────────────────────────

    def _on_image_filter_changed(self) -> None:
        self._schedule_image_refresh()

    def _on_auto_all(self) -> None:
        if self._dm is None:
            return
        p1, p99 = self._auto_range()
        for ch, cfg in self._ch_config.items():
            if cfg.get("enabled", True):
                cfg["vmin"] = p1
                cfg["vmax"] = p99
        self._image_controls.update_channel_values(self._ch_config)
        self._schedule_image_refresh()

    def _on_contrast_changed(self, method: str) -> None:
        self._contrast_method = method
        self._image_controls.set_gamma_visible(method == "gamma")
        self._schedule_image_refresh()

    def _on_gamma_changed(self, value: int) -> None:
        self._contrast_gamma = value / 100.0  # slider 10–300 → 0.1–3.0
        self._schedule_image_refresh()

    def _on_invert_toggled(self, checked: bool) -> None:
        self._invert = checked
        self._schedule_image_refresh()

    def _on_overlay_changed(self) -> None:
        ic = self._image_controls
        self._overlay_mask = ic.overlay_mask.currentText()
        col_data = ic.overlay_col.currentData()
        self._overlay_col = col_data[1] if col_data else None
        self._overlay_cmap = ic.overlay_cmap.currentText()
        self._overlay_alpha = ic.overlay_alpha.value() / 100.0
        self._schedule_image_refresh()

    def _schedule_image_refresh(self) -> None:
        self._debounce.start()

    def _refresh_images(self) -> None:
        if self._dm is None:
            return
        ic = self._image_controls
        fields = ic.get_selected_fields()
        stacks = ic.get_selected_stacks()
        tps = ic.get_selected_tps()

        if not fields or not stacks or not tps:
            return

        self._ch_config = ic.get_channel_config()

        wells = sorted(self._selected_wells) if self._selected_wells else self._dm.get_wells()
        fields_int = [int(f) for f in fields]
        stacks_int = [int(s) for s in stacks]
        tps_int = [int(t) for t in tps]
        rows_info = self._dm.lookup_row_indices(wells, fields_int, stacks_int, tps_int)
        rows_info.sort(key=lambda r: (r[1], r[4], r[3], r[2]))

        if not rows_info:
            self._image_display.clear()
            return

        self._image_display.show_loading(len(rows_info))
        self.statusBar().showMessage(f"Loading {len(rows_info)} images...")

        # Process synchronously on the main thread
        import numpy as np

        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        results = []

        for row_idx, well, field, stack, tp in rows_info:
            try:
                img_data, mask_dict = self._dm.get_imageset(row_idx)

                # Per-channel contrast
                n_ch = img_data.shape[2]
                enhanced = np.zeros_like(img_data, dtype=np.float64)
                for ch_idx, ch_name in enumerate(channel_names):
                    ch_cfg = self._ch_config.get(ch_name, {})
                    if not ch_cfg.get("enabled", True):
                        enhanced[:, :, ch_idx] = img_data[:, :, ch_idx].astype(np.float64)
                        continue
                    vmin = ch_cfg.get("vmin", 0)
                    vmax = ch_cfg.get("vmax", dmax)
                    band = img_data[:, :, ch_idx].astype(np.float64)
                    band = np.clip((band - vmin) / max(vmax - vmin, 1e-10), 0, 1)
                    if self._contrast_method == "gamma":
                        band = apply_contrast(band, "gamma", gamma=self._contrast_gamma)
                    elif self._contrast_method == "histogram_equalization":
                        band = apply_contrast(band, "histogram_equalization")
                    if self._invert:
                        band = 1.0 - band
                    enhanced[:, :, ch_idx] = band

                # Data is already [0,1]; tell composite_image vmin=0, vmax=1
                comp_config = {ch: {**cfg, "vmin": 0, "vmax": 1}
                               for ch, cfg in self._ch_config.items()}
                rgb = composite_image(enhanced, channel_names, comp_config,
                                      None, None)

                polygons = None
                if self._overlay_mask and self._overlay_mask != "None":
                    # microProfiler strips "mask_" prefix from dict keys
                    if self._overlay_mask in mask_dict:
                        polygons = extract_polygons(mask_dict[self._overlay_mask])

                results.append({
                    "rgb": np.ascontiguousarray(rgb),
                    "well": well, "field": field, "stack": stack, "tp": tp,
                    "polygons": polygons,
                })

                # Keep UI responsive during long loads
                QApplication.processEvents()
            except Exception as e:
                self.statusBar().showMessage(f"Error: [{well} f{field}] {e}")

        if results:
            self._image_display.show_results(
                results,
                overlay_alpha=self._overlay_alpha,
                overlay_cmap=self._overlay_cmap,
            )
            self.statusBar().showMessage("Ready")

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
            pass

    # ── Data View Handler ────────────────────────────────────────────────────

    def _on_data_table_changed(self, table_name: str) -> None:
        if self._dm is None or not table_name:
            return
        try:
            df = self._dm.get_table_df(table_name)
            self._data_view.set_dataframe(df)
        except Exception as e:
            self.statusBar().showMessage(f"Error loading table: {e}")

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._dm is not None:
            self._dm.close()
        super().closeEvent(event)
