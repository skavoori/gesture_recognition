#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Define our two environments
ENVIRONMENTS = {
    "training": {"dir": ROOT / ".venv", "req": ROOT / "requirements_training.txt"},
    "inference": {"dir": ROOT / ".venv_inference", "req": ROOT / "requirements_inference.txt"}
}

def venv_executable(venv_dir, name):
    return venv_dir / "bin" / name

def run(cmd, cwd):
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

def main():
    for name, config in ENVIRONMENTS.items():
        venv_dir = config["dir"]
        req_file = config["req"]
        
        print(f"\n=== Setting up {name.upper()} environment ===")
        
        if not venv_dir.is_dir():
            print(f"Creating venv: {venv_dir}")
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        
        pip = venv_executable(venv_dir, "pip")
        
        print("Upgrading pip...")
        run([str(pip), "install", "--upgrade", "pip"], cwd=ROOT)
        
        if req_file.is_file():
            print(f"Installing {req_file.name}...")
            run([str(pip), "install", "-r", str(req_file)], cwd=ROOT)
        else:
            print(f"Warning: {req_file.name} not found, skipping.")

if __name__ == "__main__":
    main()