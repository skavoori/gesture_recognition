#!/usr/bin/env python3
"""
Create isolated training and inference virtual environments.

Python 3.10–3.12 is required (PyTorch 2.x and MediaPipe 0.10.35). Python 3.11 is
preferred for Apple Silicon (MPS). Python 3.9 is not supported (PyTorch >= 3.10).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# PyTorch 2.12+ requires >=3.10; MediaPipe 0.10.35 supports 3.9–3.12 (we standardize on 3.10+)
MIN_PYTHON = (3, 10)
PREFERRED_PYTHON = (3, 11)
SUPPORTED_PYTHON = ((3, 12), (3, 11), (3, 10))

ENVIRONMENTS = {
    "training": {
        "dir": ROOT / ".venv_training",
        "req": ROOT / "requirements_training.txt",
        "description": "train.py, sweep.py, extract_landmarks.py, EDA notebooks",
    },
    "inference": {
        "dir": ROOT / ".venv_inference",
        "req": ROOT / "requirements_inference.txt",
        "description": "live_inference.py (real-time webcam)",
    },
}


def _parse_version(version_output: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_output)
    if not match:
        raise RuntimeError(f"Could not parse Python version from: {version_output!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _version_ok(major: int, minor: int) -> bool:
    return major == 3 and MIN_PYTHON[1] <= minor <= 12


def _candidate_executables() -> list[str]:
    names: list[str] = []
    for major, minor in (PREFERRED_PYTHON, *SUPPORTED_PYTHON):
        names.extend((f"python{major}.{minor}", f"python{major}{minor}"))
    names.append("python3")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        path = shutil.which(name)
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def find_python_executable() -> str:
    candidates = _candidate_executables()
    if not candidates:
        raise SystemExit(
            "No Python interpreter found. Install Python 3.11 (recommended) or 3.10/3.12."
        )

    best: str | None = None
    best_rank = 999

    for executable in candidates:
        try:
            out = subprocess.check_output(
                [executable, "--version"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue

        major, minor, _ = _parse_version(out)
        if not _version_ok(major, minor):
            print(f"Skipping {executable} ({major}.{minor}) — need Python >= 3.10")
            continue

        rank = 0 if (major, minor) == PREFERRED_PYTHON else (major, minor)[1]
        if rank < best_rank:
            best = executable
            best_rank = rank

    if best is None:
        raise SystemExit(
            "No suitable Python found. Install Python 3.11 (recommended) or 3.10/3.12. "
            "Python 3.9 cannot be used with current PyTorch releases."
        )

    version_line = subprocess.check_output(
        [best, "--version"], stderr=subprocess.STDOUT, text=True
    ).strip()
    print(f"Using interpreter: {best} ({version_line})")
    return best


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_pip(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print(f"+ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def smoke_test(name: str, py: Path) -> None:
    scripts_path = str(SCRIPTS)
    if name == "training":
        code = f"""
import sys
sys.path.insert(0, {scripts_path!r})
import torch, pandas, numpy, tqdm, optuna, cv2
import mediapipe as mp
assert not hasattr(mp, "solutions"), "mediapipe must use Tasks API (0.10+)"
from hand_landmarker_utils import create_hand_landmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from data_loader import JesterCoordinateDataset
from model import GestureLSTM, GestureGRU
hl = create_hand_landmarker(running_mode=VisionTaskRunningMode.IMAGE)
hl.close()
print("training smoke test OK")
"""
    else:
        code = f"""
import sys
sys.path.insert(0, {scripts_path!r})
import torch, cv2, numpy, pandas
import mediapipe as mp
assert not hasattr(mp, "solutions"), "mediapipe must use Tasks API (0.10+)"
from hand_landmarker_utils import create_hand_landmarker
from model import GestureLSTM
hl = create_hand_landmarker()
hl.close()
print("inference smoke test OK")
"""
    print(f"Running {name} smoke test...")
    run([str(py), "-c", code])


def setup_environment(name: str, config: dict, base_python: str) -> None:
    venv_dir: Path = config["dir"]
    req_file: Path = config["req"]

    print(f"\n=== {name.upper()} environment ({config['description']}) ===")

    if not venv_dir.is_dir():
        print(f"Creating venv: {venv_dir}")
        run([base_python, "-m", "venv", str(venv_dir)])

    pip = venv_pip(venv_dir)
    py = venv_python(venv_dir)

    run([str(pip), "install", "--upgrade", "pip"])

    if not req_file.is_file():
        raise SystemExit(f"Missing requirements file: {req_file}")

    print(f"Installing {req_file.name}...")
    run([str(pip), "install", "-r", str(req_file)])

    smoke_test(name, py)


def main() -> None:
    base_python = find_python_executable()

    for name, config in ENVIRONMENTS.items():
        setup_environment(name, config, base_python)

    print("\n=== Setup complete ===")
    print("Training (train, sweep, extract_landmarks, notebooks):")
    print("  source .venv_training/bin/activate")
    print("Inference (live webcam):")
    print("  source .venv_inference/bin/activate")


if __name__ == "__main__":
    main()
