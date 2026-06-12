from __future__ import annotations

import pandas as pd
from natsort import natsort_key
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class _PandasTableModel(QAbstractTableModel):
    """QAbstractTableModel wrapping a pandas DataFrame."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._df = pd.DataFrame()

    def setDataFrame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return len(self._df)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.4g}"
            return str(val)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])


class _NatSortProxyModel(QSortFilterProxyModel):
    """Proxy model with natural sorting (numbers sort correctly)."""

    def lessThan(self, left, right):
        left_val = self.sourceModel().data(left, Qt.DisplayRole)
        right_val = self.sourceModel().data(right, Qt.DisplayRole)
        try:
            return float(left_val) < float(right_val)
        except (ValueError, TypeError):
            return natsort_key(str(left_val)) < natsort_key(str(right_val))


class DataView(QWidget):
    """Data tab: dataset + metadata browsing, table view, PyGwalker integration."""

    dataset_browse_clicked = Signal()
    dataset_selected = Signal(str)
    reload_clicked = Signal()
    pygwalker_open_clicked = Signal()
    metadata_browse_clicked = Signal()
    metadata_merge_clicked = Signal()
    metadata_clear_clicked = Signal()
    write_to_db_clicked = Signal()
    table_radio_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        self.setStyleSheet("QPushButton { font-size: 9pt; padding: 0px 4px; }")

        # ── Row 1: Dataset browse + PyGwalker ──
        row1 = QHBoxLayout()
        row1.setAlignment(Qt.AlignBottom)
        self._btn_dataset_browse = QPushButton("Select Dataset Directory")
        self._btn_dataset_browse.setProperty("class", "primary")
        self._btn_dataset_browse.setFixedHeight(12)
        self._btn_dataset_browse.clicked.connect(self.dataset_browse_clicked)
        row1.addWidget(self._btn_dataset_browse)

        self._dataset_label = QLabel("")
        self._dataset_label.setStyleSheet("color: #aaaaaa;")
        self._dataset_label.setAlignment(Qt.AlignBottom)
        row1.addWidget(self._dataset_label)
        row1.addStretch()

        self._btn_pgw_open = QPushButton("Open in PyGwalker")
        self._btn_pgw_open.setProperty("class", "primary")
        self._btn_pgw_open.setFixedHeight(12)
        self._btn_pgw_open.setEnabled(False)
        self._btn_pgw_open.clicked.connect(self.pygwalker_open_clicked)
        row1.addWidget(self._btn_pgw_open)
        layout.addLayout(row1)

        # ── Pattern inputs (visible after dataset selected) ──
        self._pattern_widgets: list[QWidget] = []

        pat_header = QLabel("Image Pattern")
        pat_header.setStyleSheet("font-weight: bold; color: #7a9aaa; margin-top: 4px;")
        layout.addWidget(pat_header)
        self._pattern_widgets.append(pat_header)

        self._pattern_image_edit = QLineEdit()
        self._pattern_image_edit.setPlaceholderText(
            r"e.g. (?P<field>\d+)...ch(?P<channel>\d+)\.tiff"
        )
        self._pattern_image_edit.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 9pt;"
        )
        layout.addWidget(self._pattern_image_edit)
        self._pattern_widgets.append(self._pattern_image_edit)

        mask_header = QLabel("Mask Pattern")
        mask_header.setStyleSheet("font-weight: bold; color: #7a9aaa; margin-top: 2px;")
        layout.addWidget(mask_header)
        self._pattern_widgets.append(mask_header)

        self._pattern_mask_edit = QLineEdit()
        self._pattern_mask_edit.setPlaceholderText(
            r"e.g. ...cp_masks_(?P<mask_name>.+)\.png"
        )
        self._pattern_mask_edit.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 9pt;"
        )
        layout.addWidget(self._pattern_mask_edit)
        self._pattern_widgets.append(self._pattern_mask_edit)

        subdir_header = QLabel("Image Subdir Pattern")
        subdir_header.setStyleSheet("font-weight: bold; color: #7a9aaa; margin-top: 2px;")
        layout.addWidget(subdir_header)
        self._pattern_widgets.append(subdir_header)

        self._pattern_subdir_edit = QLineEdit()
        self._pattern_subdir_edit.setPlaceholderText("e.g. Images/  (leave empty to scan root)")
        self._pattern_subdir_edit.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 9pt;"
        )
        layout.addWidget(self._pattern_subdir_edit)
        self._pattern_widgets.append(self._pattern_subdir_edit)

        self._btn_reload = QPushButton("Reload Dataset")
        self._btn_reload.setProperty("class", "primary")
        self._btn_reload.setFixedHeight(12)
        self._btn_reload.setEnabled(False)
        self._btn_reload.clicked.connect(self.reload_clicked)
        layout.addWidget(self._btn_reload)
        self._pattern_widgets.append(self._btn_reload)

        self._set_pattern_visible(False)

        # ── Row 2: Metadata browse + Merge/Clear/Write ──
        row2 = QHBoxLayout()
        row2.setAlignment(Qt.AlignBottom)
        self._btn_meta_browse = QPushButton("Select Metadata")
        self._btn_meta_browse.setProperty("class", "primary")
        self._btn_meta_browse.setFixedHeight(12)
        self._btn_meta_browse.clicked.connect(self.metadata_browse_clicked)
        row2.addWidget(self._btn_meta_browse)

        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet("color: #aaaaaa;")
        self._meta_label.setAlignment(Qt.AlignBottom)
        row2.addWidget(self._meta_label)
        row2.addStretch()

        self._btn_merge = QPushButton("Merge")
        self._btn_merge.setProperty("class", "primary")
        self._btn_merge.setFixedHeight(12)
        self._btn_merge.setEnabled(False)
        self._btn_merge.clicked.connect(self.metadata_merge_clicked)
        row2.addWidget(self._btn_merge)

        self._btn_meta_clear = QPushButton("Clear")
        self._btn_meta_clear.setProperty("class", "primary")
        self._btn_meta_clear.setFixedHeight(12)
        self._btn_meta_clear.setEnabled(False)
        self._btn_meta_clear.clicked.connect(self.metadata_clear_clicked)
        row2.addWidget(self._btn_meta_clear)

        self._btn_write_db = QPushButton("Write to DB")
        self._btn_write_db.setProperty("class", "primary")
        self._btn_write_db.setFixedHeight(12)
        self._btn_write_db.setEnabled(False)
        self._btn_write_db.clicked.connect(self._on_write_to_db)
        row2.addWidget(self._btn_write_db)
        layout.addLayout(row2)

        # ── Row 3: Table radio buttons ──
        self._radio_row = QHBoxLayout()
        self._radio_row.setSpacing(12)
        self._tables_label = QLabel("Tables")
        self._tables_label.setStyleSheet("font-weight: bold; color: #7a9aaa;")
        self._radio_row.addWidget(self._tables_label)
        self._radio_group = QButtonGroup(self)
        self._radio_group.setExclusive(True)
        self._radio_group.idClicked.connect(self._on_radio_clicked)
        self._radio_container = QWidget()
        self._radio_container.setLayout(self._radio_row)
        layout.addWidget(self._radio_container)

        # ── Row 4: Table view ──
        self._model = _PandasTableModel(self)
        self._proxy = _NatSortProxyModel(self)
        self._proxy.setSourceModel(self._model)

        self._table_view = QTableView()
        self._table_view.setModel(self._proxy)
        self._table_view.setSortingEnabled(True)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.setStyleSheet(
            "QTableView { background-color: #1e1e2e; gridline-color: #333333; }"
            "QTableView::item { padding: 2px 8px; }"
            "QHeaderView::section {"
            "  background-color: #2d2d44; color: #e0e0e0;"
            "  padding: 4px 8px; border: 1px solid #333333;"
            "}"
        )
        layout.addWidget(self._table_view, 1)

        # ── Row 5: Preview hint ──
        self._preview_hint = QLabel("")
        self._preview_hint.setStyleSheet("color: #777777; font-size: 8pt;")
        self._preview_hint.setVisible(False)
        layout.addWidget(self._preview_hint)

    # ── Public methods ─────────────────────────────────────────────────────

    def _set_pattern_visible(self, visible: bool) -> None:
        for w in self._pattern_widgets:
            w.setVisible(visible)

    def set_patterns(self, image: str, mask: str, subdir: str) -> None:
        self._pattern_image_edit.setText(image)
        self._pattern_mask_edit.setText(mask)
        self._pattern_subdir_edit.setText(subdir)
        self._set_pattern_visible(True)
        self._btn_reload.setEnabled(True)

    def get_patterns(self) -> tuple[str, str, str]:
        return (
            self._pattern_image_edit.text().strip(),
            self._pattern_mask_edit.text().strip(),
            self._pattern_subdir_edit.text().strip(),
        )

    def set_dataset_label(self, text: str) -> None:
        self._dataset_label.setText(text)

    def set_metadata_label(self, text: str | None) -> None:
        self._meta_label.setText(text or "")
        has_meta = text is not None
        self._btn_merge.setEnabled(has_meta)
        self._btn_meta_clear.setEnabled(has_meta)
        self._btn_write_db.setEnabled(has_meta)

    def set_table_names(self, names: list[str]) -> None:
        # Clear existing radios (keep the "Tables" label)
        for btn in self._radio_group.buttons():
            self._radio_group.removeButton(btn)
            btn.deleteLater()
        # Remove everything after the "Tables" label
        while self._radio_row.count() > 1:
            item = self._radio_row.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        self._radio_container.setVisible(bool(names))
        _rb_style = (
            "QRadioButton { color: #7a9aaa; border: none; padding: 2px 8px; }"
            "QRadioButton:checked { background-color: #4a6a7a; color: #e0e0e0;"
            " border-radius: 3px; }"
        )
        for i, name in enumerate(names):
            rb = QRadioButton(name)
            rb.setStyleSheet(_rb_style)
            self._radio_group.addButton(rb, i)
            self._radio_row.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        self._radio_row.addStretch()
        if names:
            self.table_radio_selected.emit(names[0])

    def _on_radio_clicked(self, idx: int) -> None:
        btn = self._radio_group.button(idx)
        if btn:
            self.table_radio_selected.emit(btn.text())

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._model.setDataFrame(df)
        self._table_view.resizeColumnsToContents()
        for col in range(self._model.columnCount()):
            if self._table_view.columnWidth(col) > 120:
                self._table_view.setColumnWidth(col, 120)

    def clear_table(self) -> None:
        self._model.setDataFrame(pd.DataFrame())
        self._preview_hint.setVisible(False)

    def set_preview_hint(self, total_rows: int | None) -> None:
        """Show/hide the preview hint label."""
        if total_rows is not None and total_rows > 20:
            self._preview_hint.setText(f"Showing top 20 rows (of {total_rows} total)")
            self._preview_hint.setVisible(True)
        else:
            self._preview_hint.setVisible(False)

    def set_pygwalker_hint(self, row_count: int, sampled: bool = False) -> None:
        """Show a hint after sending data to PyGwalker."""
        if row_count == 0:
            self._preview_hint.setText("Loading data for PyGwalker...")
        elif sampled:
            self._preview_hint.setText(
                f"PyGwalker: sent {row_count} sampled rows (max 200 per group)"
            )
        else:
            self._preview_hint.setText(
                f"PyGwalker: sent all {row_count} rows"
            )
        self._preview_hint.setVisible(True)

    def set_pygwalker_buttons(self, has_tables: bool) -> None:
        self._btn_pgw_open.setEnabled(has_tables)

    def _on_write_to_db(self) -> None:
        reply = QMessageBox.question(
            self,
            "Write to Database",
            "This will overwrite the existing profiling database with the merged tables.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.write_to_db_clicked.emit()
