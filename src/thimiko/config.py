"""Default data directory and database path, overridable via environment.

The default is a user-level data directory (not source-relative), so the index
lives in one stable place whether thimiko is run from a dev checkout or installed
globally with `uv tool install`. Override with `THIMIKO_DATA_DIR`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _default_data_dir() -> Path:
    override = os.environ.get("THIMIKO_DATA_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "thimiko"
    return Path.home() / ".local" / "share" / "thimiko"


DEFAULT_DATA_DIR = _default_data_dir()
DEFAULT_DB = DEFAULT_DATA_DIR / "thimiko.sqlite"
