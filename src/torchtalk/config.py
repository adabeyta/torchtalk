"""Persistent configuration for TorchTalk.

Manages user config at ~/.config/torchtalk/config.toml (XDG-compliant)
and cache at ~/.cache/torchtalk/.

Resolution order for pytorch_source:
  1. --pytorch-source CLI flag (highest priority)
  2. PYTORCH_SOURCE environment variable
  3. ~/.config/torchtalk/config.toml
"""

import logging
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore[assignment]

from platformdirs import user_cache_path, user_config_path

log = logging.getLogger(__name__)

CONFIG_DIR = user_config_path("torchtalk")
CONFIG_FILE = CONFIG_DIR / "config.toml"
CACHE_DIR = user_cache_path("torchtalk")


def load_config() -> dict:
    """Load config from ~/.config/torchtalk/config.toml.

    Returns empty dict if file doesn't exist or can't be parsed.
    """
    if not CONFIG_FILE.exists():
        return {}

    if tomllib is None:
        log.warning("Cannot read config: tomllib/tomli not available")
        return {}

    try:
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        log.warning("Failed to read %s: %s", CONFIG_FILE, e)
        return {}


def save_config(config: dict) -> Path:
    """Write config to ~/.config/torchtalk/config.toml.

    Returns the path written to.
    """
    if tomli_w is None:
        raise RuntimeError(
            "Cannot write config: tomli-w not installed. "
            "Install with: pip install tomli-w"
        )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(config, f)
    return CONFIG_FILE


def resolve_pytorch_source(cli_flag: str | None = None) -> str | None:
    """Resolve PyTorch source path using 3-level priority.

    1. cli_flag (--pytorch-source)
    2. PYTORCH_SOURCE env var
    3. config.toml [source] pytorch_source
    """
    # Level 1: CLI flag
    if cli_flag:
        return cli_flag

    # Level 2: Environment variable
    env_val = os.environ.get("PYTORCH_SOURCE") or os.environ.get("PYTORCH_PATH")
    if env_val and Path(env_val).exists():
        return env_val

    # Level 3: Config file
    config = load_config()
    config_val = config.get("source", {}).get("pytorch_source")
    if config_val and Path(config_val).exists():
        return config_val

    return None


def source_hash(source: str | Path) -> str:
    """Compute a stable hash for a PyTorch source directory.

    Used as a cache key suffix to distinguish indexes built from
    different source checkouts.
    """
    import hashlib

    return hashlib.md5(str(Path(source).resolve()).encode()).hexdigest()[:12]


def cache_paths(source: str | Path) -> dict[str, Path]:
    """Return the canonical cache file paths for a given source directory.

    Keys:
        bindings  - Binding index JSON
        callgraph - C++ call graph JSON
    """
    h = source_hash(source)
    return {
        "bindings": CACHE_DIR / f"bindings_{h}.json",
        "callgraph": CACHE_DIR / "call_graph" / f"pytorch_callgraph_parallel_{h}.json",
    }


def validate_pytorch_path(path: str | Path) -> tuple[bool, str]:
    """Validate that a path looks like a PyTorch source checkout.

    Returns (is_valid, message).
    """
    p = Path(path)
    if not p.exists():
        return False, f"Path does not exist: {p}"
    if not p.is_dir():
        return False, f"Path is not a directory: {p}"
    if not (p / "torch").exists():
        return False, f"No 'torch/' directory found in {p} (not a PyTorch checkout?)"
    return True, f"Valid PyTorch source: {p}"
