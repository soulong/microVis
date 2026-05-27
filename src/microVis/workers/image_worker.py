from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from microVis.processing.compositing import composite_image
from microVis.processing.contrast import apply_contrast, invert_image
from microVis.processing.overlay import extract_polygons
from microVis._settings import DTYPE_MAX


class ImageWorker(QObject):
    """Background worker for loading and compositing microscopy images.

    Moves itself to a QThread. Emits raw numpy arrays; the main thread
    creates QPixmap from them.
    """

    rgba_ready = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("ImageWorker")
        self.moveToThread(self._thread)
        self._thread.started.connect(self._execute)
        self._thread.finished.connect(self.deleteLater)

        self._dm = None
        self._rows_info: list = []
        self._channel_names: list[str] = []
        self._ch_config: dict = {}
        self._contrast_method: str = "none"
        self._contrast_gamma: float = 1.0
        self._invert: bool = False
        self._overlay_mask: str | None = None
        self._overlay_cmap: str = "Viridis"

    def load(self, dm, rows_info, channel_names, ch_config,
             contrast_method="none", contrast_gamma=1.0, invert=False,
             overlay_mask=None, overlay_col=None, overlay_cmap="Viridis") -> None:
        if self._thread.isRunning():
            return
        self._dm = dm
        self._rows_info = rows_info
        self._channel_names = channel_names
        self._ch_config = ch_config
        self._contrast_method = contrast_method
        self._contrast_gamma = contrast_gamma
        self._invert = invert
        self._overlay_mask = overlay_mask
        self._overlay_col = overlay_col
        self._overlay_cmap = overlay_cmap
        self._thread.start()

    def cancel(self) -> None:
        if self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait(10000)

    def _execute(self) -> None:
        results = []
        thread = self._thread
        try:
            for row_idx, well, field, stack, tp in self._rows_info:
                if thread.isInterruptionRequested():
                    break
                try:
                    img_data, mask_dict = self._dm.get_imageset(row_idx)
                    enhanced = self._apply_channel_contrast(img_data)
                    rgb = composite_image(
                        enhanced,
                        self._channel_names,
                        {ch: cfg for ch, cfg in self._ch_config.items()},
                        None, None,
                    )
                    polygons = None
                    if self._overlay_mask and self._overlay_mask != "None":
                        mask_key = f"mask_{self._overlay_mask}"
                        if mask_key in mask_dict:
                            polygons = extract_polygons(mask_dict[mask_key])
                    results.append({
                        "rgb": np.ascontiguousarray(rgb),
                        "well": well, "field": field, "stack": stack, "tp": tp,
                        "polygons": polygons,
                    })
                except Exception as inner_e:
                    if not thread.isInterruptionRequested():
                        self.error.emit(f"[{well} f{field} z{stack} t{tp}] {inner_e}")

            if not thread.isInterruptionRequested() and results:
                self.rgba_ready.emit(results)
        except Exception as e:
            if not thread.isInterruptionRequested():
                self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()
            thread.quit()
            thread.wait(2000)
