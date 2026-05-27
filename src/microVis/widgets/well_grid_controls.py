from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from microVis.widgets._event_filter import NoScrollComboBox






class WellGridControls(QScrollArea):
    """Left sidebar controls for the well plate grid."""

    select_all_clicked = Signal()
    clear_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Plate format
        layout.addWidget(QLabel("Format"))
        self._plate_fmt = NoScrollComboBox()
        layout.addWidget(self._plate_fmt)

        layout.addSpacing(6)

        # Table
        layout.addWidget(QLabel("Table"))
        self._table = NoScrollComboBox()
        layout.addWidget(self._table)

        layout.addSpacing(6)

        # Column
        layout.addWidget(QLabel("Column"))
        self._column = NoScrollComboBox()
        layout.addWidget(self._column)

        layout.addSpacing(6)

        # Aggregation
        layout.addWidget(QLabel("Aggregation"))
        self._agg = NoScrollComboBox()
        layout.addWidget(self._agg)

        layout.addSpacing(6)

        # Colors (continuous)
        layout.addWidget(QLabel("Colors"))
        self._cmap = NoScrollComboBox()
        layout.addWidget(self._cmap)

        layout.addSpacing(6)

        # Palette (categorical)
        layout.addWidget(QLabel("Palette"))
        self._palette = NoScrollComboBox()
        layout.addWidget(self._palette)

        layout.addSpacing(8)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #333333; max-height: 1px;")
        layout.addWidget(sep)
        layout.addSpacing(4)

        # Select All / Clear
        btn_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.setProperty("class", "secondary")
        self._select_all_btn.clicked.connect(self.select_all_clicked)
        btn_row.addWidget(self._select_all_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setProperty("class", "secondary")
        self._clear_btn.clicked.connect(self.clear_clicked)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

        self.setWidget(container)

    # ── Public API ──────────────────────────────────────────────────

    @property
    def plate_format(self) -> NoScrollComboBox:
        return self._plate_fmt

    @property
    def table(self) -> NoScrollComboBox:
        return self._table

    @property
    def column(self) -> NoScrollComboBox:
        return self._column

    @property
    def aggregation(self) -> NoScrollComboBox:
        return self._agg

    @property
    def colormap(self) -> NoScrollComboBox:
        return self._cmap

    @property
    def palette(self) -> NoScrollComboBox:
        return self._palette
