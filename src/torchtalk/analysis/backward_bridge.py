"""Map backward ATen functions to their forward ops.

PyTorch's tests for backward kernels live on the forward op's `TestCase` (via
`gradcheck`/`gradgradcheck`), not on a separate `TestSigmoidBackward`. So when
a backward kernel changes, test selection needs to expand to the forward op's
test class, OpInfo, etc.

Source of truth: `derivatives.yaml` — each forward entry's gradient formulas
reference backward functions by name (e.g. `self: sigmoid_backward(grad, result)`).
We invert that into `backward_fn → forward_op_name(s)`.
"""

from __future__ import annotations

import re

# Captures `\w+_backward` plus optional trailing identifier chars (e.g. `_symint`,
# `_overrideable`, `_input`). Stops at `(` so call-site refs are matched.
_BACKWARD_CALL_RE = re.compile(r"\b(\w+_backward\w*)\s*\(")


def extract_backward_to_forward(
    derivatives: dict[str, dict],
) -> dict[str, list[str]]:
    """Build `backward_fn → list of forward ops` from the parsed derivatives table.

    Multiple forwards may share a backward (e.g. `mm_mat1_backward` is referenced
    by both `addmm` and `mm`); the value is a sorted list. The map also includes
    a `_symint`-stripped variant of each key so both naming conventions resolve.
    """
    bridge: dict[str, set[str]] = {}
    for entry in derivatives.values():
        forward = entry.get("name", "").split(".")[0]
        if not forward or forward.endswith("_backward"):
            continue
        formulas = " ".join(entry.get("gradients", {}).values())
        for match in _BACKWARD_CALL_RE.finditer(formulas):
            backward = match.group(1)
            bridge.setdefault(backward, set()).add(forward)
            if backward.endswith("_symint"):
                bridge.setdefault(backward[: -len("_symint")], set()).add(forward)
    return {k: sorted(v) for k, v in bridge.items()}
