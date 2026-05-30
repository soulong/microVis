from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CONTRAST_METHODS
from microVis.widgets._event_filter import NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSlider
from microVis.widgets.channel_controls import ChannelControls


class _MultiSelectCombo(QWidget):
    """Compact multi-select with checkboxes in a horizontal layout + Select All / Clear."""

    selection_changed = Signal()

    def __init__(self, label: str, items: list[str], checked_first: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Label + buttons row
        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(QLabel(label))

        sel_all = QPushButton("All")
        sel_all.setProperty("class", "secondary")
        sel_all.setFixedSize(42, 18)
        sel_all.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        sel_all.clicked.connect(lambda: self.set_all_checked(True))
        header.addWidget(sel_all)

        clear_btn = QPushButton("Clear")
        clear_btn.setProperty("class", "secondary")
        clear_btn.setFixedSize(48, 18)
        clear_btn.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        clear_btn.clicked.connect(lambda: self.set_all_checked(False))
        header.addWidget(clear_btn)
        header.addStretch()
        layout.addLayout(header)

        # Checkboxes in a scrollable horizontal row
        checks_scroll = QScrollArea()
        checks_scroll.setWidgetResizable(True)
        checks_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        checks_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        checks_scroll.setMaximumHeight(24)
        checks_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        checks_inner = QWidget()
        self._checks_layout = QHBoxLayout(checks_inner)
        self._checks_layout.setSpacing(4)
        self._checks_layout.setContentsMargins(0, 0, 0, 0)
        self._checkboxes: dict[str, QCheckBox] = {}
        for i, item_text in enumerate(items):
            cb = QCheckBox(item_text)
            cb.setStyleSheet("QCheckBox { font-size: 7pt; spacing: 2px; } QCheckBox::indicator { width: 12px; height: 12px; }")
            cb.setChecked(i == 0 if checked_first else True)
            cb.toggled.connect(lambda: self.selection_changed.emit())
            self._checks_layout.addWidget(cb)
            self._checkboxes[item_text] = cb
        self._checks_layout.addStretch()

        checks_scroll.setWidget(checks_inner)
        layout.addWidget(checks_scroll)

    def get_selected(self) -> list[str]:
        return [t for t, cb in self._checkboxes.items() if cb.isChecked()]

    def set_all_checked(self, checked: bool) -> None:
        for cb in self._checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.selection_changed.emit()


class ImageControls(QScrollArea):
    """Left sidebar controls for the image viewer."""

    auto_all_clicked = Signal()
    auto_range_changed = Signal()
    image_size_changed = Signal()
    channel_config_changed = Signal()
    reset_requested = Signal()
    sort_mode_changed = Signal()

    # Class labeling signals
    label_mask_selected = Signal(str)       # mask_name
    label_class_added = Signal(str)         # class_name
    label_class_selection_changed = Signal()  # selected classes changed
    label_write_clicked = Signal()          # write to db requested

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(200)
        self.setMaximumWidth(260)

        container = QWidget()
        container.setStyleSheet("""
            QComboBox, QDoubleSpinBox, QSlider {
                min-height: 18px;
                max-height: 22px;
                font-size: 8pt;
                padding: 2px 3px;
                min-width: 0;
            }
            QLabel {
                font-size: 8pt;
            }
            QPushButton {
                font-size: 9pt;
                padding: 2px 6px;
            }
        """)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(3)

        # ── Image Filters ──
        self._section("Image Filters")
        self._filter_container = QVBoxLayout()
        self._filter_container.setSpacing(2)
        self._layout.addLayout(self._filter_container)
        self._fields_widget: _MultiSelectCombo | None = None
        self._stacks_widget: _MultiSelectCombo | None = None
        self._tps_widget: _MultiSelectCombo | None = None
        self._layout.addSpacing(6)

        # ── Channel Controls ──
        self._section("Channel Setting")
        self._ch_layout = QVBoxLayout()
        self._ch_layout.setSpacing(2)
        self._layout.addLayout(self._ch_layout)
        self._channel_widgets: dict[str, ChannelControls] = {}
        self._layout.addSpacing(6)

        # ── Global Adjustments ──
        def _row(label_text, widget):
            r = QHBoxLayout()
            r.setSpacing(4)
            r.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(60)
            r.addWidget(lbl)
            r.addWidget(widget, stretch=1)
            self._layout.addLayout(r)

        # Low / High
        lowhigh_row = QHBoxLayout()
        lowhigh_row.setSpacing(4)
        lbl_lo = QLabel("Low")
        lbl_lo.setFixedWidth(28)
        lowhigh_row.addWidget(lbl_lo)
        self._auto_low = NoScrollDoubleSpinBox()
        self._auto_low.setRange(0.0, 100.0)
        self._auto_low.setValue(0.01)
        self._auto_low.setDecimals(2)
        self._auto_low.setButtonSymbols(QDoubleSpinBox.NoButtons)
        lowhigh_row.addWidget(self._auto_low, stretch=1)
        lbl_hi = QLabel("High")
        lbl_hi.setFixedWidth(28)
        lowhigh_row.addWidget(lbl_hi)
        self._auto_high = NoScrollDoubleSpinBox()
        self._auto_high.setRange(0.0, 100.0)
        self._auto_high.setValue(99.99)
        self._auto_high.setDecimals(2)
        self._auto_high.setButtonSymbols(QDoubleSpinBox.NoButtons)
        lowhigh_row.addWidget(self._auto_high, stretch=1)
        self._layout.addLayout(lowhigh_row)

        self._auto_low.valueChanged.connect(lambda: self.auto_range_changed.emit())
        self._auto_high.valueChanged.connect(lambda: self.auto_range_changed.emit())

        # Auto / Reset (centered)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        self._auto_all_btn = QPushButton("Auto")
        self._auto_all_btn.setProperty("class", "secondary")
        self._auto_all_btn.setFixedSize(64, 24)
        self._auto_all_btn.clicked.connect(self.auto_all_clicked)
        btn_row.addWidget(self._auto_all_btn)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setProperty("class", "secondary")
        self._reset_btn.setFixedSize(64, 24)
        self._reset_btn.clicked.connect(self.reset_requested)
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch()
        self._layout.addLayout(btn_row)

        # Image size
        self._image_size = NoScrollDoubleSpinBox()
        self._image_size.setRange(50, 500)
        self._image_size.setValue(250)
        self._image_size.setDecimals(0)
        self._image_size.setSingleStep(10)
        self._image_size.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._image_size.valueChanged.connect(lambda: self.image_size_changed.emit())
        _row("Img size", self._image_size)

        # Contrast
        self._contrast = NoScrollComboBox()
        self._contrast.addItems(CONTRAST_METHODS)
        _row("Transform", self._contrast)

        # Sort mode
        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        sort_row.setContentsMargins(0, 0, 0, 0)
        sort_lbl = QLabel("Sort well")
        sort_lbl.setFixedWidth(60)
        sort_row.addWidget(sort_lbl)
        self._sort_by_col = QRadioButton("By Col")
        self._sort_by_col.setStyleSheet("font-size: 8pt; spacing: 4px;")
        self._sort_by_col.toggled.connect(lambda: self.sort_mode_changed.emit())
        sort_row.addWidget(self._sort_by_col)
        self._sort_by_row = QRadioButton("By Row")
        self._sort_by_row.setChecked(True)
        self._sort_by_row.setStyleSheet("font-size: 8pt; spacing: 4px;")
        self._sort_by_row.toggled.connect(lambda: self.sort_mode_changed.emit())
        sort_row.addWidget(self._sort_by_row)
        self._layout.addLayout(sort_row)

        self._gamma_slider = NoScrollSlider(Qt.Horizontal)
        self._gamma_slider.setRange(10, 300)
        self._gamma_slider.setSingleStep(10)
        self._gamma_slider.setPageStep(10)
        self._gamma_slider.setValue(100)
        self._gamma_slider.setVisible(False)
        self._gamma_slider_label = QLabel("Gamma: 1.00")
        self._gamma_slider_label.setProperty("class", "muted")
        self._gamma_slider_label.setVisible(False)
        self._gamma_slider.valueChanged.connect(
            lambda v: self._gamma_slider_label.setText(f"Gamma: {v / 100:.2f}")
        )
        self._layout.addWidget(self._gamma_slider_label)
        self._layout.addWidget(self._gamma_slider)

        # ── Object Overlay ──
        self._section("Object Overlay")

        self._overlay_col = NoScrollComboBox()
        self._overlay_col.setEditable(True)
        self._overlay_col.setInsertPolicy(QComboBox.NoInsert)
        self._overlay_col.completer().setFilterMode(Qt.MatchContains)
        self._overlay_col.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._overlay_col.lineEdit().setPlaceholderText("Type to filter...")
        _row("Color by", self._overlay_col)

        self._overlay_cmap = NoScrollComboBox()
        _row("Colors", self._overlay_cmap)

        self._overlay_alpha = NoScrollSlider(Qt.Horizontal)
        self._overlay_alpha.setRange(0, 100)
        self._overlay_alpha.setValue(40)
        _row("Alpha", self._overlay_alpha)

        # ── Class Labeling ──
        self._section("Class Labeling")

        # Mask selector
        self._label_mask = NoScrollComboBox()
        self._label_mask.currentTextChanged.connect(self.label_mask_selected)
        _row("Mask", self._label_mask)

        # Class name input + Add button
        class_input_row = QHBoxLayout()
        class_input_row.setSpacing(4)
        class_input_row.setContentsMargins(0, 0, 0, 0)
        lbl_class = QLabel("Class")
        lbl_class.setFixedWidth(60)
        class_input_row.addWidget(lbl_class)
        self._class_input = QLineEdit()
        self._class_input.setPlaceholderText("New class name...")
        self._class_input.setStyleSheet(
            "min-height: 18px; max-height: 22px; font-size: 8pt; padding: 2px 3px;"
        )
        self._class_input.returnPressed.connect(self._on_add_class)
        class_input_row.addWidget(self._class_input, stretch=1)
        self._add_class_btn = QPushButton("Add")
        self._add_class_btn.setProperty("class", "secondary")
        self._add_class_btn.setFixedSize(36, 20)
        self._add_class_btn.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        self._add_class_btn.clicked.connect(self._on_add_class)
        class_input_row.addWidget(self._add_class_btn)
        self._layout.addLayout(class_input_row)

        # Selected classes (multi-select dropdown)
        self._class_select_label = QLabel("Selected classes")
        self._class_select_label.setStyleSheet(
            "font-size: 8pt; color: #aaaaaa; padding-top: 2px;"
        )
        self._class_select_label.setVisible(False)
        self._layout.addWidget(self._class_select_label)

        self._class_select_scroll = QScrollArea()
        self._class_select_scroll.setWidgetResizable(True)
        self._class_select_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._class_select_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._class_select_scroll.setMaximumHeight(24)
        self._class_select_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._class_select_scroll.setVisible(False)

        self._class_select_container = QWidget()
        self._class_select_layout = QHBoxLayout(self._class_select_container)
        self._class_select_layout.setSpacing(4)
        self._class_select_layout.setContentsMargins(0, 0, 0, 0)
        self._class_checkboxes: dict[str, QCheckBox] = {}
        self._class_select_layout.addStretch()
        self._class_select_scroll.setWidget(self._class_select_container)
        self._layout.addWidget(self._class_select_scroll)

        # Table name input
        table_row = QHBoxLayout()
        table_row.setSpacing(4)
        table_row.setContentsMargins(0, 0, 0, 0)
        lbl_table = QLabel("Table")
        lbl_table.setFixedWidth(60)
        table_row.addWidget(lbl_table)
        self._label_table_name = QLineEdit()
        self._label_table_name.setPlaceholderText("{mask}_label")
        self._label_table_name.setStyleSheet(
            "min-height: 18px; max-height: 22px; font-size: 8pt; padding: 2px 3px;"
        )
        table_row.addWidget(self._label_table_name, stretch=1)
        self._layout.addLayout(table_row)

        # Write to DB button (matching Auto button style)
        write_btn_row = QHBoxLayout()
        write_btn_row.setSpacing(8)
        self._write_labels_btn = QPushButton("Write to DB")
        self._write_labels_btn.setProperty("class", "secondary")
        self._write_labels_btn.setFixedSize(64, 24)
        self._write_labels_btn.clicked.connect(self.label_write_clicked)
        write_btn_row.addWidget(self._write_labels_btn)
        write_btn_row.addStretch()
        self._layout.addLayout(write_btn_row)

        self._layout.addStretch()
        self.setWidget(container)

    def _section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; color: #5a8a9a; font-size: 8pt; padding-top: 2px;")
        self._layout.addWidget(lbl)

    def set_filter_options(self, fields: list[str], stacks: list[str], tps: list[str]) -> None:
        for w in [self._fields_widget, self._stacks_widget, self._tps_widget]:
            if w is not None:
                w.deleteLater()
        while self._filter_container.count():
            item = self._filter_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._fields_widget = _MultiSelectCombo("Fields", fields, checked_first=True)
        self._stacks_widget = _MultiSelectCombo("Stacks", stacks, checked_first=True)
        self._tps_widget = _MultiSelectCombo("Timepoints", tps, checked_first=True)

        self._filter_container.addWidget(self._fields_widget)
        self._filter_container.addWidget(self._stacks_widget)
        self._filter_container.addWidget(self._tps_widget)

    def get_selected_fields(self) -> list[str]:
        return self._fields_widget.get_selected() if self._fields_widget else []

    def get_selected_stacks(self) -> list[str]:
        return self._stacks_widget.get_selected() if self._stacks_widget else []

    def get_selected_tps(self) -> list[str]:
        return self._tps_widget.get_selected() if self._tps_widget else []

    def set_channels(self, ch_config: dict) -> None:
        for w in self._channel_widgets.values():
            w.deleteLater()
        self._channel_widgets.clear()
        while self._ch_layout.count():
            item = self._ch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for ch_name, cfg in ch_config.items():
            row = ChannelControls(ch_name, cfg)
            row.config_changed.connect(lambda ch=ch_name: self.channel_config_changed.emit())
            self._ch_layout.addWidget(row)
            self._channel_widgets[ch_name] = row

    def get_channel_config(self) -> dict:
        result = {}
        for ch_name, w in self._channel_widgets.items():
            result[ch_name] = w.get_config()
        return result

    def update_channel_values(self, ch_config: dict) -> None:
        for ch_name, cfg in ch_config.items():
            if ch_name in self._channel_widgets:
                self._channel_widgets[ch_name].set_values(
                    cfg.get("vmin", 0), cfg.get("vmax", 65535)
                )

    def set_gamma_visible(self, visible: bool) -> None:
        self._gamma_slider.setVisible(visible)
        self._gamma_slider_label.setVisible(visible)

    # ── Public API ──────────────────────────────────────────────────

    @property
    def contrast(self) -> NoScrollComboBox:
        return self._contrast

    @property
    def gamma_slider(self) -> NoScrollSlider:
        return self._gamma_slider

    @property
    def auto_low(self) -> NoScrollDoubleSpinBox:
        return self._auto_low

    @property
    def auto_high(self) -> NoScrollDoubleSpinBox:
        return self._auto_high

    @property
    def image_size(self) -> NoScrollDoubleSpinBox:
        return self._image_size

    @property
    def overlay_col(self) -> NoScrollComboBox:
        return self._overlay_col

    @property
    def overlay_cmap(self) -> NoScrollComboBox:
        return self._overlay_cmap

    @property
    def overlay_alpha(self) -> NoScrollSlider:
        return self._overlay_alpha

    @property
    def sort_by_row(self) -> QRadioButton:
        return self._sort_by_row

    @property
    def fields_widget(self) -> _MultiSelectCombo | None:
        return self._fields_widget

    @property
    def stacks_widget(self) -> _MultiSelectCombo | None:
        return self._stacks_widget

    @property
    def tps_widget(self) -> _MultiSelectCombo | None:
        return self._tps_widget

    # ── Class Labeling API ─────────────────────────────────────────

    def set_label_masks(self, mask_names: list[str]) -> None:
        """Populate the mask dropdown for label annotation."""
        self._label_mask.blockSignals(True)
        self._label_mask.clear()
        self._label_mask.addItems(mask_names)
        self._label_mask.blockSignals(False)

    def get_selected_label_mask(self) -> str:
        return self._label_mask.currentText()

    def get_label_table_name(self) -> str:
        """Return the custom table name, or '' if default ({mask}_label)."""
        return self._label_table_name.text().strip()

    def get_all_class_names(self) -> list[str]:
        """Return all added class names."""
        return list(self._class_checkboxes.keys())

    def get_selected_class_names(self) -> list[str]:
        """Return currently checked class names."""
        return [n for n, cb in self._class_checkboxes.items() if cb.isChecked()]

    def _on_add_class(self) -> None:
        name = self._class_input.text().strip()
        if not name or name in self._class_checkboxes:
            return
        self._class_input.clear()

        # Show the multi-select section on first class
        self._class_select_label.setVisible(True)
        self._class_select_scroll.setVisible(True)

        cb = QCheckBox(name)
        cb.setStyleSheet(
            "QCheckBox { font-size: 7pt; spacing: 2px; } "
            "QCheckBox::indicator { width: 12px; height: 12px; }"
        )
        cb.setChecked(True)
        cb.toggled.connect(lambda: self.label_class_selection_changed.emit())
        # Insert before the trailing stretch
        idx = self._class_select_layout.count() - 1
        self._class_select_layout.insertWidget(idx, cb)
        self._class_checkboxes[name] = cb

        self.label_class_added.emit(name)
