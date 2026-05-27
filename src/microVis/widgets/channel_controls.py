from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CHANNEL_COLORS
from microVis.widgets._event_filter import NoScrollComboBox, NoScrollDoubleSpinBox


class ChannelControls(QWidget):
    """Per-channel control block: toggle + color, then vmin/vmax below."""

    config_changed = Signal(str)

    def __init__(self, ch_name: str, cfg: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._ch_name = ch_name

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(2)

        # ── Top row: checkbox + color dropdown ──
        top = QHBoxLayout()
        top.setSpacing(6)

        self._toggle = QCheckBox(ch_name)
        self._toggle.setChecked(cfg.get("enabled", True))
        self._toggle.toggled.connect(lambda: self.config_changed.emit(self._ch_name))
        top.addWidget(self._toggle)

        self._color = NoScrollComboBox()
        color_names = list(CHANNEL_COLORS.keys())
        self._color.addItems(color_names)
        current_color = cfg.get("color", "green")
        for i, cname in enumerate(color_names):
            if isinstance(current_color, str):
                if cname == current_color:
                    self._color.setCurrentIndex(i)
                    break
            elif CHANNEL_COLORS[cname] == tuple(current_color):
                self._color.setCurrentIndex(i)
                break
        self._color.currentTextChanged.connect(lambda: self.config_changed.emit(self._ch_name))
        self._color.setFixedWidth(80)
        top.addWidget(self._color)

        top.addStretch()
        root.addLayout(top)

        # ── Bottom row: vmin / vmax (indented) ──
        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        bottom.setContentsMargins(20, 0, 0, 0)

        bottom.addWidget(QLabel("vmin"))
        self._vmin = NoScrollDoubleSpinBox()
        self._vmin.setRange(0, 65535)
        self._vmin.setDecimals(0)
        self._vmin.setValue(cfg.get("vmin", 0))
        self._vmin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._vmin.valueChanged.connect(lambda: self.config_changed.emit(self._ch_name))
        self._vmin.setFixedWidth(70)
        self._vmin.setContextMenuPolicy(Qt.NoContextMenu)
        bottom.addWidget(self._vmin)

        bottom.addWidget(QLabel("vmax"))
        self._vmax = NoScrollDoubleSpinBox()
        self._vmax.setRange(0, 65535)
        self._vmax.setDecimals(0)
        self._vmax.setValue(cfg.get("vmax", 65535))
        self._vmax.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._vmax.valueChanged.connect(lambda: self.config_changed.emit(self._ch_name))
        self._vmax.setFixedWidth(70)
        self._vmax.setContextMenuPolicy(Qt.NoContextMenu)
        bottom.addWidget(self._vmax)

        bottom.addStretch()
        root.addLayout(bottom)

    def get_config(self) -> dict:
        color_name = self._color.currentText()
        return {
            "enabled": self._toggle.isChecked(),
            "color": CHANNEL_COLORS.get(color_name, (0, 1, 0)),
            "vmin": self._vmin.value(),
            "vmax": self._vmax.value(),
        }

    def set_values(self, vmin: float, vmax: float) -> None:
        self._vmin.blockSignals(True)
        self._vmax.blockSignals(True)
        self._vmin.setValue(vmin)
        self._vmax.setValue(vmax)
        self._vmin.blockSignals(False)
        self._vmax.blockSignals(False)

    def set_range(self, vmin: float, vmax: float) -> None:
        self._vmin.blockSignals(True)
        self._vmax.blockSignals(True)
        self._vmin.setRange(vmin, vmax)
        self._vmax.setRange(vmin, vmax)
        self._vmin.blockSignals(False)
        self._vmax.blockSignals(False)
