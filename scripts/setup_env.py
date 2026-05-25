#!/usr/bin/env python3
"""Create a project-local virtual environment and install dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MIN_PYTHON = (3, 10)


def venv_executable(name: str) -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / f"{name}.exe"
    return VENV_DIR / "bin" / name


def activation_hint() -> str:
    if sys.platform == "win32":
        return f"{VENV_DIR}\\Scripts\\activate"
    return f"source {VENV_DIR}/bin/activate"


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        version = ".".join(str(part) for part in MIN_PYTHON)
        print(
            f"Python {version}+ is required (found {sys.version.split()[0]}).",
            file=sys.stderr,
        )
        return 1

    if not REQUIREMENTS.is_file():
        print(f"Missing {REQUIREMENTS}", file=sys.stderr)
        return 1

    if not VENV_DIR.is_dir():
        print(f"Creating virtual environment at {VENV_DIR} ...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"Using existing virtual environment at {VENV_DIR}")

    pip = venv_executable("pip")
    python = venv_executable("python")

    run([str(pip), "install", "--upgrade", "pip"])
    run([str(pip), "install", "-r", str(REQUIREMENTS)])

    jupyter = venv_executable("jupyter")
    if jupyter.is_file():
        run([str(jupyter), "--version"])

    print()
    print("Environment is ready.")
    print(f"  Activate: {activation_hint()}")
    print("  Run notebook: jupyter notebook gesture_recognition_eda.ipynb")
    print(f"  Or: {python} -m jupyter notebook gesture_recognition_eda.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
