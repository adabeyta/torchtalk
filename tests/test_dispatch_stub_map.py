"""Unit tests for kernel-impl → ATen op map construction."""

from __future__ import annotations

from torchtalk.analysis.dispatch_stub_map import (
    _stub_to_op,
    extract_kernel_impl_to_op,
)


def _write(tmp_path, rel: str, body: str) -> None:
    p = tmp_path / "aten/src/ATen/native" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


class TestStubToOp:
    def test_strips_stub_suffix(self):
        assert _stub_to_op("hardsigmoid_stub", {"hardsigmoid"}) == "hardsigmoid"

    def test_strips_kernel_suffix(self):
        assert _stub_to_op("hardsigmoid_kernel", {"hardsigmoid"}) == "hardsigmoid"

    def test_strips_kernel_impl_suffix(self):
        assert _stub_to_op("foo_kernel_impl", {"foo"}) == "foo"

    def test_walks_underscore_segments(self):
        # `softmax_lastdim_kernel` → strip `_kernel` → still no match → walk
        # back to `softmax`.
        assert _stub_to_op("softmax_lastdim_kernel", {"softmax"}) == "softmax"

    def test_returns_none_when_no_match(self):
        assert _stub_to_op("totally_unknown_name", {"abs", "neg"}) is None

    def test_prefers_longer_match(self):
        # If both `softmax_backward` and `softmax` exist, pick the longer one.
        assert (
            _stub_to_op("softmax_backward_kernel", {"softmax", "softmax_backward"})
            == "softmax_backward"
        )


class TestExtractKernelImplToOp:
    def test_basic_register_dispatch(self, tmp_path):
        _write(
            tmp_path,
            "cpu/Activation.cpp",
            "REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"hardsigmoid": {}})
        assert out == {"hardsigmoid_kernel": "hardsigmoid"}

    def test_register_avx_variants(self, tmp_path):
        _write(
            tmp_path,
            "cpu/SoftMaxKernel.cpp",
            "ALSO_REGISTER_AVX512_DISPATCH(softmax_lastdim_kernel, "
            "&softmax_lastdim_kernel_impl)\n"
            "REGISTER_AVX512(foo_stub, &foo_kernel)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"softmax": {}, "foo": {}})
        assert out["softmax_lastdim_kernel_impl"] == "softmax"
        assert out["foo_kernel"] == "foo"

    def test_register_cuda_dispatch(self, tmp_path):
        _write(
            tmp_path,
            "cuda/Activation.cu",
            "REGISTER_CUDA_DISPATCH(threshold_stub, &threshold_kernel_cuda)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"threshold": {}})
        assert out == {"threshold_kernel_cuda": "threshold"}

    def test_skips_when_stub_not_resolvable(self, tmp_path):
        _write(
            tmp_path,
            "cpu/Foo.cpp",
            "REGISTER_DISPATCH(some_unknown_stub, &some_kernel)\n",
        )
        # native_functions doesn't contain the op; entry skipped.
        out = extract_kernel_impl_to_op(tmp_path, {})
        assert out == {}

    def test_returns_empty_when_no_native_functions(self, tmp_path):
        assert extract_kernel_impl_to_op(tmp_path, None) == {}
        assert extract_kernel_impl_to_op(tmp_path, {}) == {}

    def test_handles_missing_native_dir(self, tmp_path):
        # No aten/src/ATen/native/ exists at all.
        assert extract_kernel_impl_to_op(tmp_path, {"foo": {}}) == {}
