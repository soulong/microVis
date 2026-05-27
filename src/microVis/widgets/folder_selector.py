from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FolderSelector(QWidget):
    """Landing page with directory path input and Load button."""

    load_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        inner = QWidget()
        inner.setFixedWidth(520)
        ilayout = QVBoxLayout(inner)
        ilayout.setSpacing(12)

        title = QLabel("microVis")
        title.setStyleSheet("font-size: 28pt; font-weight: bold; color: #4cc9f0;")
        title.setAlignment(Qt.AlignCenter)
        ilayout.addWidget(title)

        subtitle = QLabel("Interactive visualization for microProfiler microscopy datasets")
        subtitle.setProperty("class", "muted")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        ilayout.addWidget(subtitle)

        ilayout.addSpacing(12)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Path to measurement directory...")
        row.addWidget(self._path_input, stretch=1)

        self._load_btn = QPushButton("Load")
        self._load_btn.setProperty("class", "primary")
        self._load_btn.clicked.connect(self._on_load)
        row.addWidget(self._load_btn)
        ilayout.addLayout(row)

        self._error_label = QLabel("")
        self._error_label.setProperty("class", "error")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        ilayout.addWidget(self._error_label)

        ilayout.addStretch()
        layout.addWidget(inner)

    def _on_load(self) -> None:
        path = self._path_input.text().strip()
        if path:
            self.load_requested.emit(path)

    def show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    def clear_error(self) -> None:
        self._error_label.setText("")
        self._error_label.setVisible(False)
