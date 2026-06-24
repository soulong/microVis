"""CLI entry point for microVis.

Usage:
    microvis                          Launch with folder selector
    microvis /path/to/measurement     Launch directly with a dataset
    microvis install-shortcut         Create Windows Start Menu and Desktop shortcut
    microvis --help                   Show this help
    microvis --version                Show version
"""
from __future__ import annotations

import sys
from pathlib import Path

from microVis.log_utils import _ensure_std_streams


def main() -> int:
    """Dispatch to GUI or CLI based on command-line arguments."""
    _ensure_std_streams()

    if "--version" in sys.argv:
        from microVis import __version__
        print(f"microVis {__version__}")
        return 0

    if "--help" in sys.argv and "install-shortcut" not in sys.argv:
        from microVis.cli import build_parser
        build_parser().print_help()
        print()
        print("GUI mode (no subcommand):")
        print("  microvis                        Launch with folder selector")
        print("  microvis <dataset-dir>          Launch directly with a dataset")
        return 0

    if "install-shortcut" in sys.argv:
        from microVis.cli import main as cli_main
        return cli_main(sys.argv[1:])

    # Default: launch GUI
    dataset_dir: str | None = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        dataset_dir = str(Path(sys.argv[1]).resolve())

    from microVis.app import run_app

    run_app(dataset_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
