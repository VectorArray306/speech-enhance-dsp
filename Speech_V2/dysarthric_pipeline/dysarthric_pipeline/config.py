"""Load the pipeline YAML config."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"


def _expand(value: Any) -> Any:
    """Expand ~ and environment variables so the same config works on Windows and Linux."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    loaded = _expand(loaded)
    override = os.environ.get("DYSARTHRIC_DATA_ROOT")
    if override and isinstance(loaded.get("paths"), dict):
        loaded["paths"]["data_root"] = os.path.expandvars(os.path.expanduser(override))
    serial = loaded.get("serial")
    if isinstance(serial, dict):
        port = str(serial.get("port") or "")
        # A Linux device node cannot be opened on Windows. "auto" picks COM port or /dev/ttyACM0 later.
        if port.lower() == "auto" or (os.name == "nt" and port.startswith("/dev/")):
            serial["port"] = "auto"
    loaded["_config_path"] = str(config_path)
    return loaded


def resolve_under(config: dict[str, Any], key: str) -> Path:
    """Resolve a paths.* entry against the directory that contains pyproject.toml."""
    raw = Path(config["paths"][key])
    if raw.is_absolute():
        return raw
    return PACKAGE_ROOT.parent / raw
