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
