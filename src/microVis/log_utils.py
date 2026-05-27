"""Logger setup for microVis."""
from __future__ import annotations

import logging
import os
from pathlib import Path

_SETUP_DONE = False


def setup_logging() -> None:
    """Configure file logging to %TEMP%/microVis.log (idempotent)."""
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    log_dir = Path(
        os.environ.get("TEMP", os.environ.get("TMP", os.environ.get("TMPDIR", "/tmp")))
    )
    log_path = log_dir / "microVis.log"
    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger("microVis")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)


def get_logger(name: str = "microVis") -> logging.Logger:
    return logging.getLogger(name)
