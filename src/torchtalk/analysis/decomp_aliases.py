"""Scrape `@register_decomposition(aten.X) def Y(...)` pairs.

PyTorch's decomp/refs registry is the only place that links internal aten
schema names (e.g. `convolution_overrideable`) to user-facing Python ops
(e.g. `conv2d`). Without this bridge, a binding-walk that lands on
`convolution_overrideable` finds no test class match because tests are named
after the user-facing API. The alias map is bidirectional so callers can
expand in either direction.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DECOMP_FILES = (
    "torch/_decomp/decompositions.py",
    "torch/_refs/__init__.py",
)


def _is_register_decomp(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "register_decomposition"
    if isinstance(node, ast.Attribute):
        return node.attr == "register_decomposition"
    return False


def _aten_op_name(attr: ast.expr) -> str | None:
    """`aten.X` → `X`; `aten.X.default` → `X`; anything else → None."""
    parts: list[str] = []
    node: ast.expr = attr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "aten" and parts:
        parts.reverse()
        return parts[0]
    return None


def _aten_ops_from_args(args: list[ast.expr]) -> list[str]:
    """Pull aten op names out of register_decomposition() args.

    Accepts `aten.X`, `aten.X.default`, `[aten.X, aten.Y]`, `(aten.X, aten.Y)`.
    """
    out: list[str] = []
    queue: list[ast.expr] = list(args)
    while queue:
        node = queue.pop()
        if isinstance(node, ast.List | ast.Tuple):
            queue.extend(node.elts)
        elif isinstance(node, ast.Attribute) and (name := _aten_op_name(node)):
            out.append(name)
    return out


def extract_decomp_aliases(source: Path) -> dict[str, list[str]]:
    """Walk decomp/refs files; return bidirectional aten ↔ python-fn alias map."""
    aliases: dict[str, set[str]] = {}

    def link(a: str, b: str) -> None:
        if a == b:
            return
        aliases.setdefault(a, set()).add(b)
        aliases.setdefault(b, set()).add(a)

    for rel in _DECOMP_FILES:
        path = source / rel
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, OSError) as e:
            log.debug(f"Skipping {path}: {e}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call):
                    continue
                if not _is_register_decomp(deco.func):
                    continue
                for op_name in _aten_ops_from_args(deco.args):
                    link(op_name, node.name)

    return {k: sorted(v) for k, v in aliases.items()}
