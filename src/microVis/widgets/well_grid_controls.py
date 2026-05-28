from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
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
        self.setMinimumWidth(200)
        self.setMaximumWidth(260)

        container = QWidget()
        container.setStyleSheet("""
            QComboBox, QDoubleSpinBox, QSlider {
                min-height: 18px;
                max-height: 22px;
                font-size: 8pt;
            }
            QLabel {
                font-size: 8pt;
            }
            QPushButton {
                font-size: 8pt;
                padding: 1px 4px;
            }
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        def _row(label_text, widget):
            r = QHBoxLayout()
            r.setSpacing(4)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(52)
            r.addWidget(lbl)
            r.addWidget(widget, stretch=1)
            layout.addLayout(r)

        # Format
        self._plate_fmt = NoScrollComboBox()
        _row("Format", self._plate_fmt)

        # Color by
        self._column = NoScrollComboBox()
        self._column.setEditable(True)
        self._column.setInsertPolicy(QComboBox.NoInsert)
        self._column.completer().setFilterMode(Qt.MatchContains)
        self._column.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._column.lineEdit().setPlaceholderText("Type to filter...")
        _row("Color by", self._column)

        # Aggregation
        self._agg = NoScrollComboBox()
        _row("Agg", self._agg)

        # Select All / Clear (centered)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        self._select_all_btn = QPushButton("All")
        self._select_all_btn.setProperty("class", "secondary")
        self._select_all_btn.setFixedSize(48, 18)
        self._select_all_btn.clicked.connect(self.select_all_clicked)
        btn_row.addWidget(self._select_all_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setProperty("class", "secondary")
        self._clear_btn.setFixedSize(48, 18)
        self._clear_btn.clicked.connect(self.clear_clicked)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Colors
        self._cmap = NoScrollComboBox()
        _row("Colors", self._cmap)

        # Palette
        self._palette = NoScrollComboBox()
        _row("Palette", self._palette)

        layout.addStretch()

        self.setWidget(container)

    # ── Public API ──────────────────────────────────────────────────

    @property
    def plate_format(self) -> NoScrollComboBox:
        return self._plate_fmt

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
