from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# This function is used to resolve the path to the correct location
def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
