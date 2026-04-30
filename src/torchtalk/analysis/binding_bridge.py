"""Bridge: Python qualname → schema → C++ dispatch info."""

from __future__ import annotations

import ast
from pathlib import Path


def _parse_tensorbase_methods(path: Path) -> set[str]:
    """Names of methods declared in `class TensorBase` in `__init__.pyi`."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TensorBase":
            return {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("__")
            }
    return set()


def _parse_top_level_functions(path: Path) -> set[str]:
    """Names of top-level `def`s in `_VariableFunctions.pyi`."""
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def build_binding_bridge(
    pytorch_source: str,
    native_functions: dict[str, dict],
) -> dict[str, dict]:
    """Map Python qualname → schema + dispatch info from `.pyi` + yaml."""
    src = Path(pytorch_source)
    init_pyi = src / "torch/_C/__init__.pyi"
    var_pyi = src / "torch/_C/_VariableFunctions.pyi"

    methods = _parse_tensorbase_methods(init_pyi) if init_pyi.exists() else set()
    funcs = _parse_top_level_functions(var_pyi) if var_pyi.exists() else set()

    bridge: dict[str, dict] = {}

    for name in methods:
        entry = native_functions.get(name)
        if entry is None:
            continue
        bridge[f"Tensor.{name}"] = {
            "schema": name,
            "kind": "method",
            "dispatch": entry.get("dispatch", {}),
            "variants": entry.get("variants", ""),
        }

    for name in funcs:
        entry = native_functions.get(name)
        if entry is None:
            continue
        bridge[f"torch.{name}"] = {
            "schema": name,
            "kind": "function",
            "dispatch": entry.get("dispatch", {}),
            "variants": entry.get("variants", ""),
        }

    return bridge
