"""Shared pytest helpers."""

from __future__ import annotations

import os
from pathlib import Path


def get_pytorch_path() -> Path | None:
    """Resolve PyTorch source from PYTORCH_SOURCE or PYTORCH_PATH env vars."""
    for var in ("PYTORCH_SOURCE", "PYTORCH_PATH"):
        if path := os.environ.get(var):
            p = Path(path)
            if p.exists() and (p / "torch").exists():
                return p
    return None
