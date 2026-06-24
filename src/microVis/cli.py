"""Command-line interface for microVis."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="microvis",
        description="microVis -- interactive visualization for microProfiler microscopy datasets",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "install-shortcut",
        help="Create Windows Start Menu and Desktop shortcut for the GUI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "install-shortcut":
        create_shortcut()
        return 0

    return 1


# ── Windows shortcut creation ────────────────────────────────────────────


def _find_conda_exe() -> str | None:
    """Return the path to ``conda.exe``, or *None* if not found."""
    conda = shutil.which("conda")
    if conda:
        return conda
    for base in (
        Path.home() / "miniconda3",
        Path.home() / "miniforge3",
        Path(r"C:\ProgramData\miniconda3"),
        Path(r"C:\ProgramData\miniforge3"),
    ):
        candidate = base / "condabin" / "conda.bat"
        if candidate.exists():
            return str(candidate)
    return None


def _find_env_pythonw(env_name: str = "micro") -> str | None:
    """Return the path to ``pythonw.exe`` in the given conda environment."""
    conda_exe = _find_conda_exe()
    if conda_exe is None:
        return None
    conda_prefix = Path(conda_exe).resolve().parent.parent
    env_prefix = conda_prefix / "envs" / env_name
    for candidate in (env_prefix / "pythonw.exe", env_prefix / "Scripts" / "pythonw.exe"):
        if candidate.exists():
            return str(candidate)
    try:
        result = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        import json
        data = json.loads(result.stdout)
        for env_path in data.get("envs", []):
            p = Path(env_path)
            if p.name == env_name:
                for candidate in (p / "pythonw.exe", p / "Scripts" / "pythonw.exe"):
                    if candidate.exists():
                        return str(candidate)
    except Exception:
        pass
    return None


def _find_pythonw() -> str | None:
    """Return path to ``pythonw.exe`` for the currently-running Python."""
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    if pyw.exists():
        return str(pyw)
    pyw = exe.parent / "Scripts" / "pythonw.exe"
    if pyw.exists():
        return str(pyw)
    return _find_env_pythonw("micro")


def _find_icon_ico() -> Path | None:
    """Return the path to the bundled ``icon.ico``."""
    try:
        from importlib.resources import files
        return Path(str(files("microVis.resources") / "icon.ico"))
    except Exception:
        return None


def _desktop() -> Path:
    return Path.home() / "Desktop"


def _start_menu_programs() -> Path:
    return Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs"


def _create_with_pywin32(
    link_path: Path, target: str, args: str, icon: str, work_dir: str
) -> bool:
    """Create shortcut using ``pywin32``.  Return *True* on success."""
    try:
        import win32com.client  # type: ignore[import-untyped]

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(link_path))
        shortcut.TargetPath = target
        shortcut.Arguments = args
        shortcut.WorkingDirectory = work_dir
        shortcut.WindowStyle = 1
        if icon:
            shortcut.IconLocation = icon
        shortcut.save()
        return True
    except Exception:
        return False


def _create_with_ps(
    link_path: Path, target: str, args: str, icon: str, work_dir: str
) -> bool:
    """Create shortcut using PowerShell COM.  Return *True* on success."""
    def _ps(s: str) -> str:
        return s.replace("'", "''")

    ps_script = (
        "$ws = New-Object -ComObject WScript.Shell\n"
        f"$sc = $ws.CreateShortcut('{_ps(str(link_path))}')\n"
        f"$sc.TargetPath = '{_ps(target)}'\n"
        f"$sc.Arguments = '{_ps(args)}'\n"
        f"$sc.WorkingDirectory = '{_ps(work_dir)}'\n"
        "$sc.WindowStyle = 1\n"
        f"$sc.IconLocation = '{_ps(icon)}'\n"
        "$sc.Save()\n"
    )
    ps1: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ps1", delete=False) as f:
            f.write(ps_script)
            ps1 = f.name
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
            check=True, capture_output=True,
        )
        return link_path.exists()
    except Exception:
        return False
    finally:
        if ps1:
            Path(ps1).unlink(missing_ok=True)


def create_shortcut() -> None:
    """Create shortcut that launches microVis without a console window.

    Uses ``pythonw.exe`` (windowless Python) to avoid a visible console
    window.  Falls back to ``conda run`` only if ``pythonw.exe`` cannot
    be found anywhere.
    """
    if sys.platform != "win32":
        print("Shortcut creation is only supported on Windows.")
        return

    icon_ico = _find_icon_ico()
    icon_str = str(icon_ico) if icon_ico else ""

    pythonw = _find_pythonw()
    if pythonw:
        target = pythonw
        args = "-m microVis"
    else:
        conda_exe = _find_conda_exe()
        if conda_exe is None:
            print("Could not locate pythonw.exe or conda. Shortcut not created.")
            return
        target = conda_exe
        args = "run -n micro --no-capture-output microvis"

    work_dir = str(Path.home())

    link_dirs = [_start_menu_programs(), _desktop()]
    for link_path in [d / "microVis.lnk" for d in link_dirs]:
        ok = _create_with_pywin32(link_path, target, args, icon_str, work_dir)
        if not ok:
            ok = _create_with_ps(link_path, target, args, icon_str, work_dir)

        if ok:
            print(f"Shortcut created: {link_path}")
        else:
            print("Failed to create shortcut. Install pywin32 (`pip install pywin32`) and retry.")


if __name__ == "__main__":
    sys.exit(main())
