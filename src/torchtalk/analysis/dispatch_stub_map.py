"""Map kernel-impl C++ symbols to the ATen op they implement.

PyTorch's CPU/CUDA kernels live behind a stub-and-impl indirection:

    DECLARE_DISPATCH(fn_t, hardsigmoid_stub)        // header
    DEFINE_DISPATCH(hardsigmoid_stub)                // .cpp
    REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)  // <arch>/Kernel.cpp

When `hardsigmoid_kernel` changes, our walker has no binding to land on (the
only TORCH_LIBRARY_IMPL entry uses the stub via dispatch). This module
scrapes `REGISTER_*` macros to build `kernel_impl_name → ATen op name`, so
`_bindings_for` can resolve the kernel directly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Matches REGISTER_DISPATCH, REGISTER_AVX512, REGISTER_CUDA_DISPATCH,
# ALSO_REGISTER_AVX512_DISPATCH, REGISTER_NO_AVX2_DISPATCH, etc.
_REGISTER_RE = re.compile(r"\b(?:ALSO_)?REGISTER_\w+\s*\(\s*(\w+)\s*,\s*&?\s*(\w+)")

_SCAN_DIRS = ("cpu", "cuda", "quantized/cpu", "quantized/cuda")
_SCAN_EXTS = ("*.cpp", "*.cu", "*.h")
_STUB_SUFFIXES = ("_stub", "_kernel_impl", "_kernel")


def _stub_to_op(stub: str, nf_keys: set[str]) -> str | None:
    """Resolve a stub name to an ATen op by trying suffix-stripped candidates.

    Stubs like `softmax_lastdim_kernel` peel multiple `_`-segments before
    matching `softmax`. Tries longest→shortest until one is in
    native_functions.
    """
    candidates: list[str] = [stub]
    for suffix in _STUB_SUFFIXES:
        if stub.endswith(suffix):
            candidates.append(stub[: -len(suffix)])
    parts = stub.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidates.append("_".join(parts[:i]))
    for c in candidates:
        if c in nf_keys:
            return c
    return None


def extract_kernel_impl_to_op(
    source: Path, native_functions: dict[str, dict] | None
) -> dict[str, str]:
    """Build kernel-impl → ATen op map by scraping REGISTER_* macros."""
    if not native_functions:
        return {}
    nf_keys = set(native_functions)
    native_root = source / "aten" / "src" / "ATen" / "native"
    if not native_root.exists():
        return {}

    kernel_to_op: dict[str, str] = {}
    for d in _SCAN_DIRS:
        sub = native_root / d
        if not sub.exists():
            continue
        for ext in _SCAN_EXTS:
            for path in sub.rglob(ext):
                try:
                    content = path.read_text(errors="replace")
                except OSError as e:
                    log.debug(f"Skipping {path}: {e}")
                    continue
                for stub, kernel in _REGISTER_RE.findall(content):
                    if op := _stub_to_op(stub, nf_keys):
                        kernel_to_op[kernel] = op
    return kernel_to_op
