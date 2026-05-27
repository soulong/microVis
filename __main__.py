"""CLI entry point for the microVis desktop application.

Usage:
    microvis-gui                          # Launch with folder selector
    microvis-gui /path/to/measurement     # Launch directly with a dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="microvis-gui",
        description="microVis -- interactive visualization for microProfiler microscopy datasets",
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help="Path to a measurement directory to load on startup",
    )
    parser.add_argument(
        "--version", action="version", version="microVis 0.2.0",
    )
    args = parser.parse_args()

    dataset_dir: str | None = None
    if args.dataset_dir:
        dataset_dir = str(Path(args.dataset_dir).resolve())

    # Import after argument parsing for faster --version/--help
    from refactoring.app import run_app

    run_app(dataset_dir)


if __name__ == "__main__":
    main()
