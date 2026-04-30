"""Unit tests for binding_detector internals (no PyTorch source required)."""

from __future__ import annotations

from torchtalk.analysis.binding_detector import (
    BindingDetector,
    BindingType,
    _clean_impl_target,
)


class TestCleanImplTarget:
    def test_bare_name(self):
        assert _clean_impl_target("foo") == "foo"

    def test_strips_leading_ampersand(self):
        assert _clean_impl_target("&foo") == "foo"

    def test_strips_namespace(self):
        assert _clean_impl_target("at::native::foo") == "foo"
        assert _clean_impl_target("&at::native::foo") == "foo"

    def test_strips_torch_fn(self):
        assert _clean_impl_target("TORCH_FN(foo)") == "foo"
        assert _clean_impl_target("TORCH_FN(at::native::foo)") == "foo"

    def test_strips_torch_fn_boxed(self):
        assert _clean_impl_target("TORCH_FN_BOXED(foo)") == "foo"
        assert _clean_impl_target("TORCH_FN_BOXED(at::native::foo)") == "foo"

    def test_makefallthrough_falls_back_to_op_name(self):
        # `m.impl("abs", CppFunction::makeFallthrough())` has no real impl;
        # use op_name so by_cpp_name["abs"] still resolves.
        assert (
            _clean_impl_target("CppFunction::makeFallthrough(", op_name="abs") == "abs"
        )

    def test_makenamednotsupported_falls_back(self):
        assert (
            _clean_impl_target("CppFunction::makeNamedNotSupported(", op_name="foo")
            == "foo"
        )

    def test_makefromboxedfunction_extracts_template_arg(self):
        assert (
            _clean_impl_target(
                "CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>("
            )
            == "unsupportedDynamicOp"
        )
        assert (
            _clean_impl_target("CppFunction::makeFromBoxedFunction<at::native::foo>(")
            == "foo"
        )

    def test_static_cast_falls_back_to_op_name(self):
        # static_cast captures break the regex; use op_name fallback.
        assert _clean_impl_target("static_cast<int64_t (*", op_name="size") == "size"

    def test_lambda_falls_back_to_op_name(self):
        assert _clean_impl_target("[](Tensor", op_name="layer_norm") == "layer_norm"

    def test_empty_returns_op_name(self):
        assert _clean_impl_target("", op_name="foo") == "foo"


class TestImplRegex:
    """Verify cpp_name no longer leaks `TORCH_FN(` wrappers."""

    def _detect(self, src: str) -> list[tuple[str, str]]:
        detector = BindingDetector()
        graph = detector.detect_bindings("test.cpp", src)
        return [
            (b.python_name, b.cpp_name)
            for b in graph.bindings
            if b.binding_type == BindingType.TORCH_LIBRARY_IMPL.value
        ]

    def test_torch_fn_wrapper_extracts_inner_name(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, CPU, m) {
            m.impl("resize_", TORCH_FN(at::native::resize_));
            m.impl("add", TORCH_FN(add_kernel));
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "resize_" in cpp_names
        assert "add_kernel" in cpp_names
        assert not any("TORCH_FN" in cpp for cpp in cpp_names)

    def test_ampersand_and_namespace_stripped(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, CPU, m) {
            m.impl("foo", &at::native::foo);
            m.impl("bar", at::native::bar);
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "foo" in cpp_names
        assert "bar" in cpp_names

    def test_makefallthrough_keys_under_op_name(self):
        # The fallthrough has no real C++ impl, but we still want the binding
        # keyed under `abs` so a walk through `at::native::abs` finds it.
        src = """
        TORCH_LIBRARY_IMPL(aten, Named, m) {
            m.impl("abs", CppFunction::makeFallthrough());
            m.impl("abs.out", CppFunction::makeFallthrough());
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "abs" in cpp_names
        # Overload `abs.out` should also key under bare `abs`
        assert all(cpp == "abs" for _, cpp in bindings)

    def test_makefromboxedfunction_keys_under_template_arg(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
            m.impl("nonzero",
                torch::CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>());
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "unsupportedDynamicOp" in cpp_names
