from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
