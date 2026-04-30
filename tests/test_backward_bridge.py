"""Unit tests for backward → forward op bridge construction."""

from __future__ import annotations

from torchtalk.analysis.backward_bridge import extract_backward_to_forward


def _entry(name: str, gradients: dict[str, str]) -> dict:
    return {"name": name, "gradients": gradients}


class TestExtractBackwardToForward:
    def test_simple_backward_call(self):
        derivs = {
            "sigmoid": _entry("sigmoid", {"self": "sigmoid_backward(grad, result)"}),
        }
        out = extract_backward_to_forward(derivs)
        assert out == {"sigmoid_backward": ["sigmoid"]}

    def test_multiple_forwards_share_a_backward(self):
        derivs = {
            "addmm": _entry("addmm", {"mat1": "mm_mat1_backward(grad)"}),
            "mm": _entry("mm", {"self": "mm_mat1_backward(grad)"}),
        }
        out = extract_backward_to_forward(derivs)
        assert out == {"mm_mat1_backward": ["addmm", "mm"]}

    def test_strips_overload_suffix_from_forward_name(self):
        # Forward `add.Tensor` → forward op is `add`.
        derivs = {
            "add.Tensor": _entry("add.Tensor", {"self": "add_backward(grad)"}),
        }
        out = extract_backward_to_forward(derivs)
        assert out == {"add_backward": ["add"]}

    def test_normalizes_symint_suffix(self):
        # Both `convolution_backward_symint` AND `convolution_backward` should
        # bridge to the forward op.
        derivs = {
            "convolution": _entry(
                "convolution",
                {"input": "convolution_backward_symint(grad, input, weight)"},
            ),
        }
        out = extract_backward_to_forward(derivs)
        assert out["convolution_backward_symint"] == ["convolution"]
        assert out["convolution_backward"] == ["convolution"]

    def test_skips_backward_entries_as_sources(self):
        # `*_backward` forward entries (yes, those exist) shouldn't seed the
        # bridge themselves — only the canonical forward ops do.
        derivs = {
            "convolution_backward": _entry(
                "convolution_backward", {"grad": "convolution_backward_x(grad)"}
            ),
            "convolution": _entry(
                "convolution", {"input": "convolution_backward(grad)"}
            ),
        }
        out = extract_backward_to_forward(derivs)
        assert "convolution_backward" in out
        assert out["convolution_backward"] == ["convolution"]
        # convolution_backward_x came from a `_backward` source — must be skipped
        assert "convolution_backward_x" not in out

    def test_empty_derivatives(self):
        assert extract_backward_to_forward({}) == {}

    def test_no_backward_call_in_formulas(self):
        derivs = {"foo": _entry("foo", {"self": "grad * 2"})}
        assert extract_backward_to_forward(derivs) == {}

    def test_compound_backward_names(self):
        # Names like `_softmax_backward_data`, `linear_backward` should match.
        derivs = {
            "softmax": _entry(
                "softmax", {"self": "_softmax_backward_data(grad, output, dim)"}
            ),
        }
        out = extract_backward_to_forward(derivs)
        assert out == {"_softmax_backward_data": ["softmax"]}
