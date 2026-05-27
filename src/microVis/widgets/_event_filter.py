from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSlider


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores scroll-wheel.

    Overrides event() — wheelEvent does not work because QAbstractSpinBox
    processes wheel events through an internal event filter on its child
    QLineEdit, completely outside the normal wheelEvent dispatch chain.
    """

    def event(self, event):
        if event.type() == QEvent.Type.Wheel:
            return True
        return super().event(event)


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores scroll-wheel."""

    def event(self, event):
        if event.type() == QEvent.Type.Wheel:
            return True
        return super().event(event)


class NoScrollSlider(QSlider):
    """QSlider that ignores scroll-wheel."""

    def event(self, event):
        if event.type() == QEvent.Type.Wheel:
            return True
        return super().event(event)
