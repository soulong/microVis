from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CMAP_OPTIONS, CONTRAST_METHODS, DEFAULT_CMAP
from microVis.widgets.channel_controls import ChannelControls
from microVis.widgets._event_filter import NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSlider


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

        # Checkboxes in a horizontal row
        self._checks_layout = QHBoxLayout()
        self._checks_layout.setSpacing(4)
        self._checks_layout.setContentsMargins(0, 0, 0, 0)
        self._checkboxes: dict[str, QCheckBox] = {}
        for i, item_text in enumerate(items):
            cb = QCheckBox(item_text)
            cb.setChecked(i == 0 if checked_first else True)
            cb.toggled.connect(lambda: self.selection_changed.emit())
            self._checks_layout.addWidget(cb)
            self._checkboxes[item_text] = cb
        self._checks_layout.addStretch()
        layout.addLayout(self._checks_layout)

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
    channel_config_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)

        container = QWidget()
        container.setStyleSheet("""
            QComboBox, QDoubleSpinBox, QSlider {
                min-height: 20px;
                max-height: 24px;
                font-size: 9pt;
            }
            QLabel {
                font-size: 9pt;
            }
        """)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)

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
        self._ch_layout = QVBoxLayout()
        self._ch_layout.setSpacing(2)
        self._layout.addLayout(self._ch_layout)
        self._channel_widgets: dict[str, ChannelControls] = {}
        self._layout.addSpacing(6)

        # ── Global Adjustments ──
        auto_row = QHBoxLayout()
        auto_row.setSpacing(4)
        auto_row.addWidget(QLabel("Low"))
        self._auto_low = NoScrollDoubleSpinBox()
        self._auto_low.setRange(0.0, 100.0)
        self._auto_low.setValue(0.01)
        self._auto_low.setDecimals(2)
        self._auto_low.setButtonSymbols(QDoubleSpinBox.NoButtons)
        auto_row.addWidget(self._auto_low)
        auto_row.addWidget(QLabel("High"))
        self._auto_high = NoScrollDoubleSpinBox()
        self._auto_high.setRange(0.0, 100.0)
        self._auto_high.setValue(99.99)
        self._auto_high.setDecimals(2)
        self._auto_high.setButtonSymbols(QDoubleSpinBox.NoButtons)
        auto_row.addWidget(self._auto_high)
        self._layout.addLayout(auto_row)

        self._auto_all_btn = QPushButton("Auto All")
        self._auto_all_btn.setProperty("class", "secondary")
        self._auto_all_btn.clicked.connect(self.auto_all_clicked)
        self._layout.addWidget(self._auto_all_btn)

        self._layout.addSpacing(4)

        cl = QHBoxLayout()
        cl.addWidget(QLabel("Contrast"))
        self._contrast = NoScrollComboBox()
        self._contrast.addItems(CONTRAST_METHODS)
        cl.addWidget(self._contrast, stretch=1)
        self._layout.addLayout(cl)

        self._gamma_slider = NoScrollSlider(Qt.Horizontal)
        self._gamma_slider.setRange(10, 300)
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

        self._layout.addSpacing(4)

        self._invert_btn = QPushButton("Invert")
        self._invert_btn.setCheckable(True)
        self._invert_btn.setChecked(False)
        self._invert_btn.setProperty("class", "secondary")
        self._layout.addWidget(self._invert_btn)

        self._layout.addSpacing(6)

        # ── Object Overlay ──
        self._section("Object Overlay")

        self._layout.addWidget(QLabel("Map Column"))
        self._overlay_col = NoScrollComboBox()
        self._overlay_col.setEditable(True)
        self._overlay_col.setInsertPolicy(QComboBox.NoInsert)
        self._overlay_col.completer().setFilterMode(Qt.MatchContains)
        self._overlay_col.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._overlay_col.lineEdit().setPlaceholderText("Type to filter...")
        self._layout.addWidget(self._overlay_col)

        self._layout.addWidget(QLabel("Colors"))
        self._overlay_cmap = NoScrollComboBox()
        self._layout.addWidget(self._overlay_cmap)

        self._layout.addWidget(QLabel("Alpha"))
        self._overlay_alpha = NoScrollSlider(Qt.Horizontal)
        self._overlay_alpha.setRange(0, 100)
        self._overlay_alpha.setValue(40)
        self._layout.addWidget(self._overlay_alpha)

        self._layout.addStretch()
        self.setWidget(container)

    def _section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; color: #5a8a9a; padding-top: 4px;")
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
    def invert_button(self) -> QPushButton:
        return self._invert_btn

    @property
    def auto_low(self) -> NoScrollDoubleSpinBox:
        return self._auto_low

    @property
    def auto_high(self) -> NoScrollDoubleSpinBox:
        return self._auto_high

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
    def fields_widget(self) -> _MultiSelectCombo | None:
        return self._fields_widget

    @property
    def stacks_widget(self) -> _MultiSelectCombo | None:
        return self._stacks_widget

    @property
    def tps_widget(self) -> _MultiSelectCombo | None:
        return self._tps_widget
