import yaml
from pathlib import Path
from typing import Any


_CONFIG: dict | None = None
_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    with open(cfg_path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get(key: str, default: Any = None) -> Any:
    """Dot-notation access: get('data.ticker')"""
    cfg = load_config()
    parts = key.split(".")
    node = cfg
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def root() -> Path:
    return _ROOT
