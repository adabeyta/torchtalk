"""Tests for binding_bridge: .pyi + native_functions.yaml join."""

from __future__ import annotations

import pytest

from torchtalk.analysis.binding_bridge import (
    _parse_tensorbase_methods,
    _parse_top_level_functions,
    build_binding_bridge,
)

from .conftest import get_pytorch_path

PYTORCH_PATH = get_pytorch_path()

INIT_PYI = """\
from typing import Any

class TensorBase:
    def copy_(self, other): ...
    def add(self, other): ...
    def __repr__(self): ...
    def _internal_helper(self): ...

class OtherClass:
    def not_a_binding(self): ...
"""

VAR_PYI = """\
from typing import overload

def add(input, other, *, alpha=1, out=None): ...

@overload
def add(input, alpha, other): ...

def matmul(input, other, *, out=None): ...

def _hidden(): ...
"""


@pytest.fixture
def pyi_tree(tmp_path):
    """Layout matches PyTorch source: torch/_C/{__init__,_VariableFunctions}.pyi."""
    c_dir = tmp_path / "torch" / "_C"
    c_dir.mkdir(parents=True)
    (c_dir / "__init__.pyi").write_text(INIT_PYI)
    (c_dir / "_VariableFunctions.pyi").write_text(VAR_PYI)
    return tmp_path


class TestParseTensorBaseMethods:
    def test_extracts_methods_skips_dunders(self, pyi_tree):
        methods = _parse_tensorbase_methods(pyi_tree / "torch/_C/__init__.pyi")
        assert methods == {"copy_", "add", "_internal_helper"}

    def test_ignores_other_classes(self, pyi_tree):
        methods = _parse_tensorbase_methods(pyi_tree / "torch/_C/__init__.pyi")
        assert "not_a_binding" not in methods

    def test_missing_class_returns_empty(self, tmp_path):
        path = tmp_path / "stub.pyi"
        path.write_text("class SomethingElse:\n    def foo(self): ...\n")
        assert _parse_tensorbase_methods(path) == set()


class TestParseTopLevelFunctions:
    def test_extracts_functions_dedupes_overloads(self, pyi_tree):
        funcs = _parse_top_level_functions(pyi_tree / "torch/_C/_VariableFunctions.pyi")
        assert funcs == {"add", "matmul"}

    def test_skips_underscore_prefixed(self, pyi_tree):
        funcs = _parse_top_level_functions(pyi_tree / "torch/_C/_VariableFunctions.pyi")
        assert "_hidden" not in funcs


class TestBuildBindingBridge:
    @pytest.fixture
    def native_functions(self):
        return {
            "copy_": {
                "dispatch": {"CPU": "copy_", "CUDA": "copy_"},
                "variants": "method",
            },
            "add": {
                "dispatch": {"CompositeImplicitAutograd": "add"},
                "variants": "function, method",
            },
            "matmul": {
                "dispatch": {"CompositeImplicitAutograd": "matmul"},
                "variants": "function",
            },
            "internal_only": {"dispatch": {}, "variants": ""},
        }

    def test_methods_become_tensor_qualnames(self, pyi_tree, native_functions):
        bridge = build_binding_bridge(str(pyi_tree), native_functions)
        assert "Tensor.copy_" in bridge
        assert bridge["Tensor.copy_"]["kind"] == "method"
        assert bridge["Tensor.copy_"]["schema"] == "copy_"
        assert bridge["Tensor.copy_"]["dispatch"] == {"CPU": "copy_", "CUDA": "copy_"}

    def test_functions_become_torch_qualnames(self, pyi_tree, native_functions):
        bridge = build_binding_bridge(str(pyi_tree), native_functions)
        assert "torch.add" in bridge
        assert bridge["torch.add"]["kind"] == "function"
        assert bridge["torch.matmul"]["kind"] == "function"

    def test_method_and_function_for_same_op(self, pyi_tree, native_functions):
        bridge = build_binding_bridge(str(pyi_tree), native_functions)
        assert "Tensor.add" in bridge
        assert "torch.add" in bridge
        assert bridge["Tensor.add"]["kind"] == "method"
        assert bridge["torch.add"]["kind"] == "function"

    def test_skips_yaml_entries_with_no_python_binding(
        self, pyi_tree, native_functions
    ):
        bridge = build_binding_bridge(str(pyi_tree), native_functions)
        assert "torch.internal_only" not in bridge
        assert "Tensor.internal_only" not in bridge

    def test_skips_pyi_entries_with_no_yaml(self, pyi_tree, native_functions):
        bridge = build_binding_bridge(str(pyi_tree), native_functions)
        assert "Tensor._internal_helper" not in bridge

    def test_missing_pyi_returns_empty_bridge(self, tmp_path, native_functions):
        bridge = build_binding_bridge(str(tmp_path), native_functions)
        assert bridge == {}

    def test_dunders_not_bridged(self, pyi_tree, native_functions):
        bridge = build_binding_bridge(str(pyi_tree), native_functions)
        assert not any(k.startswith("Tensor.__") for k in bridge)


@pytest.mark.skipif(
    PYTORCH_PATH is None,
    reason="PYTORCH_SOURCE or PYTORCH_PATH environment variable not set",
)
class TestRealPyTorch:
    """Smoke test against actual PyTorch source."""

    def test_real_pyi_yields_known_methods(self):
        methods = _parse_tensorbase_methods(PYTORCH_PATH / "torch/_C/__init__.pyi")
        for expected in ("copy_", "add", "mul", "view", "size", "matmul"):
            assert expected in methods, f"missing Tensor.{expected}"

    def test_real_pyi_yields_known_functions(self):
        funcs = _parse_top_level_functions(
            PYTORCH_PATH / "torch/_C/_VariableFunctions.pyi"
        )
        for expected in ("add", "matmul", "cat", "stack", "mean"):
            assert expected in funcs, f"missing torch.{expected}"
