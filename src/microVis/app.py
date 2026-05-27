"""QApplication bootstrap for the microVis desktop GUI."""
from pathlib import Path
import sys

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QSlider

from microVis.log_utils import setup_logging


class _GlobalScrollFilter(QObject):
    """Event filter that blocks scroll-wheel on all input widgets app-wide."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, (QAbstractSpinBox, QComboBox, QSlider)):
                return True
        return False


def run_app(dataset_dir: str | None = None) -> None:
    """Launch the microVis desktop application.

    Args:
        dataset_dir: Optional path to a measurement directory to load on startup.
    """
    setup_logging()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Block scroll-wheel on all input widgets app-wide
    _scroll_filter = _GlobalScrollFilter(app)
    app.installEventFilter(_scroll_filter)
    app.setApplicationName("microVis")
    app.setOrganizationName("microVis")
    from microVis import __version__
    app.setApplicationVersion(__version__)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(30, 30, 46))
    palette.setColor(QPalette.AlternateBase, QColor(37, 37, 54))
    palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 68))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(45, 45, 68))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, QColor(240, 71, 112))
    palette.setColor(QPalette.Highlight, QColor(76, 201, 240))
    palette.setColor(QPalette.HighlightedText, QColor(30, 30, 46))
    app.setPalette(palette)

    resources = Path(__file__).parent / "resources"

    icon_path = resources / "icon.png"
    if icon_path.exists():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

    qss_path = resources / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    from microVis.main_window import MainWindow

    window = MainWindow(dataset_dir=dataset_dir)
    if icon_path.exists():
        window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())
