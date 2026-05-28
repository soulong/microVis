from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FolderSelector(QWidget):
    """Landing page with Browse button and path display."""

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
        title.setStyleSheet("font-size: 28pt; font-weight: bold; color: #5a8a9a;")
        title.setAlignment(Qt.AlignCenter)
        ilayout.addWidget(title)

        subtitle = QLabel("Interactive visualization for microProfiler microscopy datasets")
        subtitle.setProperty("class", "muted")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        ilayout.addWidget(subtitle)

        ilayout.addSpacing(12)

        # Browse button (centered)
        browse_row = QHBoxLayout()
        browse_row.addStretch()
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setProperty("class", "primary")
        self._browse_btn.clicked.connect(self._on_browse)
        browse_row.addWidget(self._browse_btn)
        browse_row.addStretch()
        ilayout.addLayout(browse_row)

        # Path label (centered, below button)
        self._path_label = QLabel("")
        self._path_label.setProperty("class", "muted")
        self._path_label.setAlignment(Qt.AlignCenter)
        self._path_label.setWordWrap(True)
        ilayout.addWidget(self._path_label)

        self._error_label = QLabel("")
        self._error_label.setProperty("class", "error")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        ilayout.addWidget(self._error_label)

        ilayout.addStretch()
        layout.addWidget(inner)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            self._path_label.setText(path)
            self.load_requested.emit(path)

    def show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    def clear_error(self) -> None:
        self._error_label.setText("")
        self._error_label.setVisible(False)
