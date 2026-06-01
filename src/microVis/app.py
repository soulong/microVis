"""QApplication bootstrap for the microVis desktop GUI."""
import os
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox

from microVis.log_utils import setup_logging


class _WheelBlocker(QObject):
    """Event filter that blocks scroll-wheel on unfocused input widgets.

    Uses hit-testing (widgetAt) + parent walk so the block works even when
    the event targets a child widget (e.g. the QLineEdit inside a spinbox).
    Only blocks when the widget does NOT already have keyboard focus — if
    the user explicitly focused a spinbox, scrolling should still work.
    """

    def eventFilter(self, watched, event):
        try:
            if event.type() != QEvent.Type.Wheel:
                return False
            pos = event.globalPosition().toPoint()
            w = QApplication.instance().widgetAt(pos)
            for _ in range(4):
                if w is None:
                    break
                if isinstance(w, (QAbstractSpinBox, QComboBox)) and not w.hasFocus():
                    return True
                w = w.parentWidget()
            return False
        except KeyboardInterrupt:
            # Ctrl+C during event processing — force-exit to avoid GIL crash
            os._exit(0)


def run_app(dataset_dir: str | None = None) -> None:
    """Launch the microVis desktop application.

    Args:
        dataset_dir: Optional path to a measurement directory to load on startup.
    """
    os.environ.setdefault("QT_LOGGING_RULES", "qt.gui.icc=false")

    setup_logging()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Block scroll-wheel on all input widgets app-wide
    app.installEventFilter(_WheelBlocker(app))
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
        app.setWindowIcon(QIcon(str(icon_path)))

    qss_path = resources / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    from microVis.main_window import MainWindow

    icon = QIcon(str(icon_path)) if icon_path.exists() else None

    window = MainWindow(dataset_dir=dataset_dir)
    if icon:
        window.setWindowIcon(icon)
    window.show()

    # Reinforce icon after window is fully rendered (fixes inconsistent taskbar icon on Windows)
    if icon:
        QTimer.singleShot(200, lambda: window.setWindowIcon(icon))

    sys.exit(app.exec())
