from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
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


class DataView(QWidget):
    """Data View tab: table selector + sortable/filterable QTableView."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Table"))
        self._table_sel = QComboBox()
        self._table_sel.setMinimumWidth(200)
        ctrl_row.addWidget(self._table_sel)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Table view
        self._model = _PandasTableModel(self)
        self._proxy = QSortFilterProxyModel(self)
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
        layout.addWidget(self._table_view)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._model.setDataFrame(df)
        self._table_view.resizeColumnsToContents()

    @property
    def table_selector(self) -> QComboBox:
        return self._table_sel
