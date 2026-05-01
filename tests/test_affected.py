"""Tests for affected_tests + symbols_in_file."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from torchtalk.analysis.affected import (
    _api_to_source_paths,
    _class_matches_api,
    _tests_mentioning_apis,
    _tests_via_profiling,
    affected_tests,
    api_attr_variants,
    normalize_api,
    symbols_in_file,
)
from torchtalk.tools.affected import _do_affected


class TestNormalizeApi:
    def test_bare_name(self):
        assert normalize_api("copy_") == "copy_"
        assert normalize_api("trace") == "trace"

    def test_strips_namespace(self):
        assert normalize_api("aten.zero_") == "zero_"
        assert normalize_api("aten.get_gradients") == "get_gradients"

    def test_drops_uppercase_overload_tag(self):
        assert normalize_api("aten.fill_.Scalar") == "fill_"
        assert normalize_api("aten.sum.dim_IntList") == "sum"

    def test_drops_lowercase_literal_overload_tag(self):
        assert normalize_api("aten.size.int") == "size"
        assert normalize_api("aten.foo.default") == "foo"

    def test_preserves_lowercase_subnamespace(self):
        # masked.sum is a sub-namespace, not an overload — keep it.
        assert normalize_api("aten.masked.sum") == "masked.sum"
        assert normalize_api("aten.special.zeta") == "special.zeta"


class TestClassMatchesApi:
    def test_exact_pascal_match(self):
        assert _class_matches_api("TestCopy", "copy_")
        assert _class_matches_api("TestAdd", "add")
        assert _class_matches_api("TestBinaryCrossEntropy", "binary_cross_entropy")

    def test_pascal_prefix_with_uppercase_boundary(self):
        assert _class_matches_api("TestCopyTo", "copy_")
        assert _class_matches_api("TestCopyExtended", "copy_")

    def test_rejects_lowercase_continuation(self):
        # TestCopying has lowercase 'i' after 'Copy', not a word boundary.
        assert not _class_matches_api("TestCopying", "copy_")

    def test_rejects_non_test_prefix(self):
        assert not _class_matches_api("MyCopy", "copy")
        assert not _class_matches_api("CopyHelper", "copy")

    def test_rejects_unrelated_class(self):
        assert not _class_matches_api("TestMul", "add")


class TestAffectedTests:
    @pytest.fixture
    def extractor(self):
        ext = MagicMock()
        # foo_kernel calls bar_kernel, both have bindings
        ext.get_callers.side_effect = lambda func, fuzzy: {
            "foo_kernel": [{"caller": "foo_dispatcher"}],
            "foo_dispatcher": [{"caller": "foo_entry"}],
            "foo_entry": [],
        }.get(func, [])
        return ext

    def test_walks_callers_collects_bindings_and_classes(self, extractor):
        by_cpp_name = {
            "foo_kernel": [{"python_name": "aten.foo", "cpp_name": "foo"}],
            "foo_entry": [{"python_name": "bar", "cpp_name": "bar_entry"}],
        }
        test_classes = {
            "TestFoo": [
                {"file": "test/test_ops.py", "is_test_class": True, "line": 10},
            ],
            "TestBar": [
                {"file": "test/test_bar.py", "is_test_class": True, "line": 20},
            ],
        }
        test_files = {"test/test_ops.py": {}, "test/test_bar.py": {}}

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            depth=3,
        )

        assert result["callers_walked"] == 3
        assert sorted(result["python_apis"]) == ["bar", "foo"]
        assert {tr["file"] for tr in result["test_runs"]} == {
            "test/test_ops.py",
            "test/test_bar.py",
        }

    def test_skips_non_test_helpers(self, extractor):
        by_cpp_name = {"foo_kernel": [{"python_name": "foo"}]}
        test_classes = {
            "TestFoo": [
                {"file": "test/helpers.py", "is_test_class": False, "line": 1},
            ],
        }
        test_files = {"test/helpers.py": {}}

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            depth=1,
        )
        assert result["test_runs"] == []

    def test_skips_files_outside_test_tree(self, extractor):
        by_cpp_name = {"foo_kernel": [{"python_name": "foo"}]}
        test_classes = {
            "TestFoo": [
                {"file": "torch/internal/foo.py", "is_test_class": True, "line": 1},
            ],
        }
        test_files = {"test/test_ops.py": {}}

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            depth=1,
        )
        assert result["test_runs"] == []

    def test_opinfo_match_adds_all_opinfo_files_with_no_class_filter(self, extractor):
        by_cpp_name = {"foo_kernel": [{"python_name": "aten.foo"}]}
        test_classes = {}
        test_files = {
            "test/test_ops.py": {},
            "test/test_meta.py": {},
            "test/test_other.py": {},
        }
        opinfo_registry = {"foo": {"file": "torch/.../opinfo/definitions/foo.py"}}
        opinfo_test_files = {"test/test_ops.py", "test/test_meta.py"}

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            opinfo_registry=opinfo_registry,
            opinfo_test_files=opinfo_test_files,
            depth=1,
        )
        runs_by_file = {
            tr["file"]: tr["included_classes"] for tr in result["test_runs"]
        }
        assert runs_by_file == {
            "test/test_ops.py": [],
            "test/test_meta.py": [],
        }

    def test_no_opinfo_match_falls_back_to_opinfo_files(self, extractor):
        # API resolved but not in OpInfo and no test class matches → catch-all
        # adds OpInfo whole-file runs (test_ops.py / test_meta.py exercise
        # ops via @ops(op_db) parametrization).
        by_cpp_name = {"foo_kernel": [{"python_name": "aten.foo"}]}
        opinfo_registry = {"bar": {}}  # foo not registered
        opinfo_test_files = {"test/test_ops.py"}

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes={},
            test_files={"test/test_ops.py": {}},
            opinfo_registry=opinfo_registry,
            opinfo_test_files=opinfo_test_files,
            depth=1,
        )
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_ops.py"}

    def test_opinfo_alias_match_expands_files(self, extractor):
        # API "conv2d" not in opinfo_registry directly, but is in alias_map
        # because OpInfo("nn.functional.conv2d", aliases=("conv2d",)).
        by_cpp_name = {"foo_kernel": [{"python_name": "aten.conv2d"}]}
        opinfo_registry = {"nn.functional.conv2d": {"name": "nn.functional.conv2d"}}
        opinfo_alias_map = {
            "conv2d": [{"name": "nn.functional.conv2d"}],
        }
        opinfo_test_files = {"test/test_ops.py", "test/test_meta.py"}

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes={},
            test_files={"test/test_ops.py": {}, "test/test_meta.py": {}},
            opinfo_registry=opinfo_registry,
            opinfo_alias_map=opinfo_alias_map,
            opinfo_test_files=opinfo_test_files,
            depth=1,
        )
        files = {tr["file"] for tr in result["test_runs"]}
        assert files == {"test/test_ops.py", "test/test_meta.py"}

    def test_decomp_alias_expansion_reaches_user_facing_test_class(self, extractor):
        # Binding lands on `convolution_overrideable` (no TestConvolutionOverrideable
        # in PyTorch). decomp_alias_map bridges to `conv2d`, which DOES have a
        # test class. Without the bridge, no test_runs would be produced.
        by_cpp_name = {
            "foo_kernel": [
                {"python_name": "aten.convolution_overrideable", "cpp_name": "foo"}
            ]
        }
        test_classes = {
            "TestConv2d": [
                {"file": "test/test_nn.py", "is_test_class": True, "line": 1},
            ],
        }
        test_files = {"test/test_nn.py": {}}
        decomp_alias_map = {
            "convolution_overrideable": ["conv2d"],
            "conv2d": ["convolution_overrideable"],
        }

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            decomp_alias_map=decomp_alias_map,
            depth=1,
        )
        assert "conv2d" in result["python_apis"]
        assert "convolution_overrideable" in result["python_apis"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_nn.py"}

    def test_backward_bridge_expands_to_forward_op(self, extractor):
        # Walked C++ func is a backward kernel; bridge maps it to the forward
        # op so the forward's TestCase gets pulled in.
        by_cpp_name = {
            "sigmoid_backward": [
                {"python_name": "aten.sigmoid_backward", "cpp_name": "sigmoid_backward"}
            ]
        }
        test_classes = {
            "TestSigmoid": [
                {"file": "test/test_unary.py", "is_test_class": True, "line": 1}
            ],
        }
        backward_to_forward = {"sigmoid_backward": ["sigmoid"]}

        result = affected_tests(
            funcs=["at::native::sigmoid_backward"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files={"test/test_unary.py": {}},
            backward_to_forward=backward_to_forward,
            depth=1,
        )
        assert "sigmoid" in result["python_apis"]
        assert "sigmoid_backward" in result["python_apis"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_unary.py"}

    def test_kernel_suffix_strip_falls_back_to_native_function(self, extractor):
        # binary_cross_entropy_kernel has no REGISTER_DISPATCH stub but the
        # base name (suffix-stripped) IS in native_functions.
        result = affected_tests(
            funcs=["at::native::binary_cross_entropy_kernel"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={
                "TestBinaryCrossEntropy": [
                    {"file": "test/test_loss.py", "is_test_class": True, "line": 1}
                ]
            },
            test_files={"test/test_loss.py": {}},
            native_functions={"binary_cross_entropy": {}},
            depth=1,
        )
        assert "binary_cross_entropy" in result["python_apis"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_loss.py"}

    def test_pascal_kernel_impl_resolves_to_snake_case_op(self, extractor):
        # GeluCUDAKernelImpl → strip CUDAKernelImpl → Gelu → gelu (in native_fns).
        result = affected_tests(
            funcs=["at::native::GeluCUDAKernelImpl"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={
                "TestGelu": [
                    {"file": "test/test_nn.py", "is_test_class": True, "line": 1}
                ]
            },
            test_files={"test/test_nn.py": {}},
            native_functions={"gelu": {}},
            depth=1,
        )
        assert "gelu" in result["python_apis"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_nn.py"}

    def test_pascal_kernel_impl_falls_back_to_native_prefix(self, extractor):
        # LayerNormBackwardKernelImpl → layer_norm_backward NOT in native_fns,
        # but native_layer_norm_backward IS — fallback should resolve.
        result = affected_tests(
            funcs=["at::native::LayerNormBackwardKernelImpl"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={},
            test_files={},
            native_functions={"native_layer_norm_backward": {}},
            depth=1,
        )
        assert "native_layer_norm_backward" in result["python_apis"]

    def test_pascal_kernel_impl_handles_acronym_runs(self, extractor):
        # RMSNormKernelImpl → acronym `RMS` correctly splits to `rms_norm`.
        result = affected_tests(
            funcs=["at::native::RMSNormKernelImpl"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={},
            test_files={},
            native_functions={"rms_norm": {}},
            depth=1,
        )
        assert "rms_norm" in result["python_apis"]

    def test_pascal_kernel_impl_no_match_when_op_unknown(self, extractor):
        # ChooseQuantizationParamsKernelImpl → snake `choose_quantization_params`
        # NOT in native_functions — no API is added.
        result = affected_tests(
            funcs=["at::native::ChooseQuantizationParamsKernelImpl"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={},
            test_files={},
            native_functions={"some_other_op": {}},
            depth=1,
        )
        assert result["python_apis"] == []

    def test_dispatch_reverse_index_resolves_vendor_backend(self, extractor):
        # cudnn_convolution_forward isn't its own native_function entry, but
        # the dispatch reverse index maps it to cudnn_convolution.
        dispatch_to_op = {"cudnn_convolution_forward": "cudnn_convolution"}
        result = affected_tests(
            funcs=["at::native::cudnn_convolution_forward"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={
                "TestCudnnConvolution": [
                    {"file": "test/test_conv.py", "is_test_class": True, "line": 1}
                ]
            },
            test_files={"test/test_conv.py": {}},
            dispatch_to_op=dispatch_to_op,
            depth=1,
        )
        assert "cudnn_convolution" in result["python_apis"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_conv.py"}

    def test_catchall_opinfo_when_no_test_class_matches(self, extractor):
        # API resolved (convolution_overrideable) but no test class by name and
        # not in OpInfo → fall back to OpInfo whole-file runs.
        by_cpp_name = {
            "convolution_overrideable": [
                {
                    "python_name": "aten.convolution_overrideable",
                    "cpp_name": "convolution_overrideable",
                }
            ]
        }
        result = affected_tests(
            funcs=["at::native::convolution_overrideable"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes={},  # no class matches
            test_files={"test/test_ops.py": {}, "test/test_meta.py": {}},
            opinfo_test_files={"test/test_ops.py", "test/test_meta.py"},
            depth=1,
        )
        files = {tr["file"] for tr in result["test_runs"]}
        assert files == {"test/test_ops.py", "test/test_meta.py"}

    def test_catchall_skipped_when_test_class_already_matched(self, extractor):
        # If class-name match already produced a hit, don't add catch-all noise.
        by_cpp_name = {
            "softmax": [{"python_name": "aten.softmax", "cpp_name": "softmax"}]
        }
        result = affected_tests(
            funcs=["at::native::softmax"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes={
                "TestSoftmax": [
                    {"file": "test/test_nn.py", "is_test_class": True, "line": 1}
                ]
            },
            test_files={
                "test/test_nn.py": {},
                "test/test_ops.py": {},
                "test/test_meta.py": {},
            },
            opinfo_test_files={"test/test_ops.py", "test/test_meta.py"},
            depth=1,
        )
        files = {tr["file"] for tr in result["test_runs"]}
        assert files == {"test/test_nn.py"}

    def test_kernel_impl_to_op_fallback_resolves_api(self, extractor):
        # Walked C++ func is a kernel impl name; no binding, no native_function
        # entry, but the dispatch stub map links it back to the ATen op.
        kernel_impl_to_op = {"softmax_lastdim_kernel_impl": "softmax"}
        result = affected_tests(
            funcs=["at::native::softmax_lastdim_kernel_impl"],
            cpp_extractor=extractor,
            by_cpp_name={},
            test_classes={
                "TestSoftmax": [
                    {"file": "test/test_nn.py", "is_test_class": True, "line": 1}
                ]
            },
            test_files={"test/test_nn.py": {}},
            kernel_impl_to_op=kernel_impl_to_op,
            depth=1,
        )
        assert "softmax" in result["python_apis"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_nn.py"}

    def test_native_implementations_fallback_resolves_api(self, extractor):
        # Walked C++ func has NO matching binding (e.g. fallthrough cases where
        # cpp_name is a no-impl marker), but native_implementations confirms
        # `abs` is an ATen op name. Resolution should still produce API "abs".
        by_cpp_name = {}  # empty — no binding cpp_name matches
        native_implementations = {"abs": [{"function_name": "abs", "file_path": "x"}]}
        test_classes = {
            "TestAbs": [
                {"file": "test/test_unary.py", "is_test_class": True, "line": 1}
            ],
        }
        test_files = {"test/test_unary.py": {}}

        result = affected_tests(
            funcs=["at::native::abs"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            native_implementations=native_implementations,
            depth=1,
        )
        assert result["python_apis"] == ["abs"]
        assert {tr["file"] for tr in result["test_runs"]} == {"test/test_unary.py"}

    def test_native_functions_fallback_strips_out_suffix(self):
        # `at::native::abs_out` → strip `_out` → look up `abs` in native_functions.
        ext = MagicMock()
        ext.get_callers.return_value = []
        result = affected_tests(
            funcs=["at::native::abs_out"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            native_functions={"abs": {"name": "abs"}},
            depth=1,
        )
        assert "abs" in result["python_apis"]

    def test_native_fallback_skipped_when_binding_matches(self, extractor):
        # When by_cpp_name resolves the walked name, native_functions fallback
        # must NOT also fire (don't double-count APIs).
        by_cpp_name = {"abs": [{"python_name": "aten.abs", "cpp_name": "abs"}]}
        result = affected_tests(
            funcs=["at::native::abs"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes={},
            test_files={},
            native_functions={"abs": {}},
            depth=1,
        )
        # Single API "abs" — derived from binding, fallback didn't add a duplicate.
        assert result["python_apis"] == ["abs"]

    def test_no_decomp_alias_map_leaves_apis_untouched(self, extractor):
        # Same setup, but without decomp_alias_map: no expansion, no test runs.
        by_cpp_name = {
            "foo_kernel": [
                {"python_name": "aten.convolution_overrideable", "cpp_name": "foo"}
            ]
        }
        test_classes = {
            "TestConv2d": [
                {"file": "test/test_nn.py", "is_test_class": True, "line": 1},
            ],
        }
        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files={"test/test_nn.py": {}},
            depth=1,
        )
        assert result["python_apis"] == ["convolution_overrideable"]
        assert result["test_runs"] == []

    def test_file_cohort_expands_apis_for_small_file(self, extractor):
        # Matched binding lives in a file with 3 sibling kernels (≤cohort_cap):
        # all sibling python_names should join the API set.
        binding = {
            "python_name": "aten.lift",
            "cpp_name": "lift_functionalize",
            "file_path": "aten/src/ATen/FunctionalizeFallbackKernel.cpp",
        }
        siblings = [
            binding,
            {
                "python_name": "aten.lift_fresh",
                "cpp_name": "lift_fresh_functionalize",
                "file_path": "aten/src/ATen/FunctionalizeFallbackKernel.cpp",
            },
            {
                "python_name": "aten._to_copy",
                "cpp_name": "_to_copy_functionalize",
                "file_path": "aten/src/ATen/FunctionalizeFallbackKernel.cpp",
            },
        ]
        result = affected_tests(
            funcs=["lift_functionalize"],
            cpp_extractor=extractor,
            by_cpp_name={"lift_functionalize": [binding]},
            test_classes={},
            test_files={},
            bindings_by_file={
                "aten/src/ATen/FunctionalizeFallbackKernel.cpp": siblings,
            },
            depth=1,
        )
        assert set(result["python_apis"]) == {"lift", "lift_fresh", "_to_copy"}

    def test_file_cohort_cap_suppresses_large_file(self, extractor):
        # Cohort over the cap (registry-style file) is skipped; only the matched
        # binding's API is included.
        binding = {
            "python_name": "aten.lift",
            "cpp_name": "lift_functionalize",
            "file_path": "aten/src/ATen/core/NamedRegistrations.cpp",
        }
        big_cohort = [binding] + [
            {
                "python_name": f"aten.op_{i}",
                "cpp_name": f"op_{i}",
                "file_path": "aten/src/ATen/core/NamedRegistrations.cpp",
            }
            for i in range(20)
        ]
        result = affected_tests(
            funcs=["lift_functionalize"],
            cpp_extractor=extractor,
            by_cpp_name={"lift_functionalize": [binding]},
            test_classes={},
            test_files={},
            bindings_by_file={
                "aten/src/ATen/core/NamedRegistrations.cpp": big_cohort,
            },
            cohort_cap=15,
            depth=1,
        )
        assert result["python_apis"] == ["lift"]

    def test_file_cohort_disabled_when_no_index(self, extractor):
        # Without bindings_by_file, only the directly matched binding contributes.
        binding = {
            "python_name": "aten.lift",
            "cpp_name": "lift_functionalize",
            "file_path": "aten/src/ATen/FunctionalizeFallbackKernel.cpp",
        }
        result = affected_tests(
            funcs=["lift_functionalize"],
            cpp_extractor=extractor,
            by_cpp_name={"lift_functionalize": [binding]},
            test_classes={},
            test_files={},
            depth=1,
        )
        assert result["python_apis"] == ["lift"]

    def test_seed_file_ops_bridges_inner_helper_to_parent(self):
        # raw_cudnn_convolution_forward_out has no callers in the call graph
        # but lives in ConvShared.cpp alongside cudnn_convolution. ops_by_file
        # bridges the helper to the parent op family.
        ext = MagicMock()
        ext.get_callers.return_value = []
        ext.function_locations = {
            "at::native::raw_cudnn_convolution_forward_out": (
                "aten/src/ATen/native/cudnn/ConvShared.cpp",
                208,
            ),
        }
        ops_by_file = {
            "aten/src/ATen/native/cudnn/ConvShared.cpp": {
                "cudnn_convolution",
                "cudnn_convolution_transpose",
                "cudnn_convolution_relu",
            },
        }
        result = affected_tests(
            funcs=["raw_cudnn_convolution_forward_out"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            ops_by_file=ops_by_file,
            depth=1,
        )
        assert "cudnn_convolution" in result["python_apis"]
        assert "cudnn_convolution_transpose" in result["python_apis"]

    def test_seed_file_ops_caps_large_files(self):
        # File with > cohort_cap ops is skipped; helper resolves to nothing.
        ext = MagicMock()
        ext.get_callers.return_value = []
        ext.function_locations = {
            "at::native::helper": ("aten/src/ATen/native/Big.cpp", 1),
        }
        ops_by_file = {
            "aten/src/ATen/native/Big.cpp": {f"op_{i}" for i in range(20)},
        }
        result = affected_tests(
            funcs=["helper"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            ops_by_file=ops_by_file,
            cohort_cap=15,
            depth=1,
        )
        assert result["python_apis"] == []

    def test_symbol_to_file_fallback_when_call_graph_empty(self):
        # libclang missed the symbol entirely (file body preprocessed out, e.g.
        # `#if AT_CUDNN_ENABLED()` with cuDNN disabled). The regex-derived
        # symbol_to_file gives us the file anyway.
        ext = MagicMock()
        ext.get_callers.return_value = []
        ext.function_locations = {}  # call graph empty for our target
        symbol_to_file = {
            "cudnn_convolution_forward_out": (
                "/p/aten/src/ATen/native/cudnn/ConvShared.cpp"
            ),
        }
        ops_by_file = {
            "/p/aten/src/ATen/native/cudnn/ConvShared.cpp": {
                "cudnn_convolution",
                "cudnn_convolution_transpose",
            },
        }
        result = affected_tests(
            funcs=["cudnn_convolution_forward_out"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            ops_by_file=ops_by_file,
            symbol_to_file=symbol_to_file,
            depth=1,
        )
        assert "cudnn_convolution" in result["python_apis"]
        assert "cudnn_convolution_transpose" in result["python_apis"]

    def test_dir_fallback_aggregates_when_file_has_no_ops(self):
        # Symbol's file (MHA.cpp) has no registered ops, but the cudnn dir
        # has bindings in sibling files — directory aggregation picks them up.
        ext = MagicMock()
        ext.get_callers.return_value = []
        ext.function_locations = {
            "at::native::run_cudnn_SDP_fprop": (
                "/p/aten/src/ATen/native/cudnn/MHA.cpp",
                15,
            ),
        }
        ops_by_file = {
            "/p/aten/src/ATen/native/cudnn/MHA.cpp": set(),  # helper-only
            "/p/aten/src/ATen/native/cudnn/ConvShared.cpp": {"cudnn_convolution"},
            "/p/aten/src/ATen/native/cudnn/BatchNorm.cpp": {"cudnn_batch_norm"},
        }
        result = affected_tests(
            funcs=["run_cudnn_SDP_fprop"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            ops_by_file=ops_by_file,
            depth=1,
        )
        apis = set(result["python_apis"])
        assert "cudnn_convolution" in apis
        assert "cudnn_batch_norm" in apis

    def test_dir_fallback_skips_non_vendor_paths(self):
        # File outside vendor backend dirs does NOT trigger directory fallback.
        ext = MagicMock()
        ext.get_callers.return_value = []
        ext.function_locations = {
            "at::native::helper": ("/p/aten/src/ATen/native/cpu/Helper.cpp", 1),
        }
        ops_by_file = {
            "/p/aten/src/ATen/native/cpu/Helper.cpp": set(),
            "/p/aten/src/ATen/native/cpu/Other.cpp": {"foo"},
        }
        result = affected_tests(
            funcs=["helper"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            ops_by_file=ops_by_file,
            depth=1,
        )
        assert result["python_apis"] == []

    def test_dir_fallback_respects_dir_cap(self):
        # Vendor dir aggregates over dir_cap → reject (no expansion).
        ext = MagicMock()
        ext.get_callers.return_value = []
        ext.function_locations = {
            "at::native::helper": (
                "/p/aten/src/ATen/native/cudnn/Helper.cpp",
                1,
            ),
        }
        ops_by_file = {
            "/p/aten/src/ATen/native/cudnn/Helper.cpp": set(),
        } | {
            f"/p/aten/src/ATen/native/cudnn/F{i}.cpp": {f"op_{i}"}
            for i in range(35)
        }
        result = affected_tests(
            funcs=["helper"],
            cpp_extractor=ext,
            by_cpp_name={},
            test_classes={},
            test_files={},
            ops_by_file=ops_by_file,
            dir_cap=30,
            depth=1,
        )
        assert result["python_apis"] == []


class TestApiAttrVariants:
    def test_includes_self_and_in_place_pair(self):
        assert api_attr_variants("copy") == {"copy", "copy_"}
        assert api_attr_variants("copy_") == {"copy", "copy_"}

    def test_dotted_api_includes_leaf(self):
        assert api_attr_variants("masked.sum") == {
            "masked.sum",
            "sum",
            "sum_",
        }

    def test_drops_empty_strings(self):
        # rstrip("_") on a bare "_" produces "" — must not pollute the set.
        assert "" not in api_attr_variants("_")


class TestTestsMentioningApis:
    def test_matches_attr_index_to_test_files(self):
        attr_index = {
            "size": [
                {
                    "file": "test/test_torch.py",
                    "class": "TestTorch",
                    "function": "test_sizes",
                },
                {
                    "file": "outside/foo.py",
                    "class": "TestFoo",
                    "function": "test_x",
                },
            ],
        }
        test_files = {"test/test_torch.py": {}}
        result = _tests_mentioning_apis({"size"}, attr_index, test_files)
        assert result == {"test/test_torch.py": {"TestTorch"}}

    def test_drops_api_when_over_cap(self):
        # Generic API matches 5 distinct (file, class) hits — cap of 3 drops it.
        attr_index = {
            "add": [
                {"file": f"test/f{i}.py", "class": f"TestF{i}", "function": "t"}
                for i in range(5)
            ],
        }
        test_files = {f"test/f{i}.py": {} for i in range(5)}
        result = _tests_mentioning_apis({"add"}, attr_index, test_files, per_api_cap=3)
        assert result == {}

    def test_keeps_api_at_or_under_cap(self):
        attr_index = {
            "rare_op": [
                {"file": f"test/f{i}.py", "class": f"TestF{i}", "function": "t"}
                for i in range(3)
            ],
        }
        test_files = {f"test/f{i}.py": {} for i in range(3)}
        result = _tests_mentioning_apis(
            {"rare_op"}, attr_index, test_files, per_api_cap=3
        )
        assert len(result) == 3

    def test_uses_api_variants(self):
        # An API named "copy_" should match attrs recorded as "copy" too.
        attr_index = {
            "copy": [
                {
                    "file": "test/test_torch.py",
                    "class": "TestTorch",
                    "function": "test_copy",
                },
            ],
        }
        result = _tests_mentioning_apis(
            {"copy_"}, attr_index, {"test/test_torch.py": {}}
        )
        assert result == {"test/test_torch.py": {"TestTorch"}}

    def test_skips_hits_without_class(self):
        attr_index = {
            "size": [
                {
                    "file": "test/test_torch.py",
                    "class": None,
                    "function": "test_top",
                },
            ],
        }
        # Class-less hits don't contribute included_classes — we only use
        # this index to refine class lists, not to add whole-file runs.
        result = _tests_mentioning_apis(
            {"size"}, attr_index, {"test/test_torch.py": {}}
        )
        assert result == {}

    def test_drops_non_torch_receiver(self):
        # `d.copy()` where `d` is a dict — drop the hit.
        attr_index = {
            "copy": [
                {
                    "file": "test/test_x.py",
                    "class": "TestX",
                    "function": "test_dict_copy",
                    "receiver_type": "dict",
                },
                {
                    "file": "test/test_x.py",
                    "class": "TestX",
                    "function": "test_tensor_copy",
                    "receiver_type": "tensor",
                },
            ],
        }
        result = _tests_mentioning_apis({"copy"}, attr_index, {"test/test_x.py": {}})
        assert result == {"test/test_x.py": {"TestX"}}

    def test_keeps_unknown_receiver(self):
        # No receiver_type recorded → pass through (conservative).
        attr_index = {
            "copy": [
                {
                    "file": "test/test_x.py",
                    "class": "TestX",
                    "function": "test_helper_result",
                    # No receiver_type key
                },
            ],
        }
        result = _tests_mentioning_apis({"copy"}, attr_index, {"test/test_x.py": {}})
        assert result == {"test/test_x.py": {"TestX"}}

    def test_drops_all_non_torch_filters_file(self):
        # When EVERY hit on a file is non-torch, the file should not appear.
        attr_index = {
            "copy": [
                {
                    "file": "test/test_dict_ops.py",
                    "class": "TestDictOps",
                    "function": "test_copy",
                    "receiver_type": "dict",
                },
                {
                    "file": "test/test_dict_ops.py",
                    "class": "TestDictOps",
                    "function": "test_copy_other",
                    "receiver_type": "list",
                },
            ],
        }
        result = _tests_mentioning_apis(
            {"copy"}, attr_index, {"test/test_dict_ops.py": {}}
        )
        assert result == {}


class TestAffectedTestsWithAttrIndex:
    @pytest.fixture
    def extractor(self):
        ext = MagicMock()
        ext.get_callers.side_effect = lambda func, fuzzy: []
        return ext

    def test_attr_index_adds_generic_class_match(self, extractor):
        # foo_kernel binds to aten.size; class-name match misses TestTorch
        # (generic), but attr-index catches `t.size(...)` inside it.
        by_cpp_name = {"foo_kernel": [{"python_name": "aten.size"}]}
        test_classes = {}  # No TestSize class — class-name match yields nothing.
        test_files = {"test/test_torch.py": {}}
        test_attr_index = {
            "size": [
                {
                    "file": "test/test_torch.py",
                    "class": "TestTorch",
                    "function": "test_sizes",
                }
            ],
        }

        result = affected_tests(
            funcs=["foo_kernel"],
            cpp_extractor=extractor,
            by_cpp_name=by_cpp_name,
            test_classes=test_classes,
            test_files=test_files,
            test_attr_index=test_attr_index,
            depth=1,
        )
        assert result["test_runs"] == [
            {"file": "test/test_torch.py", "included_classes": ["TestTorch"]}
        ]


class TestApiToSourcePaths:
    def test_dotted_api_yields_module_and_init(self):
        paths = _api_to_source_paths("nn.functional.conv2d")
        assert paths == ["torch/nn/functional.py", "torch/nn/functional/__init__.py"]

    def test_two_part_api(self):
        paths = _api_to_source_paths("optim.SGD")
        assert paths == ["torch/optim.py", "torch/optim/__init__.py"]

    def test_bare_api_yields_nothing(self):
        assert _api_to_source_paths("copy_") == []

    def test_underscore_namespacing_treated_as_dot(self):
        # ATen schemas use `linalg_cross` underscore form; profiling keys are
        # `torch/linalg.py` dot form. Bridge between them.
        assert _api_to_source_paths("linalg_cross") == [
            "torch/linalg.py",
            "torch/linalg/__init__.py",
        ]

    def test_leading_underscore_yields_nothing(self):
        assert _api_to_source_paths("_conj_physical") == []


class TestTestsViaProfiling:
    def test_resolves_test_via_source_file_match(self):
        profiling = {
            "torch/nn/functional.py": {
                "test_nn": 1.0,
                "test_torch": 1.0,
            },
        }
        test_files = {"test/test_nn.py": {}, "test/test_torch.py": {}}
        result = _tests_via_profiling({"nn.functional.conv2d"}, profiling, test_files)
        assert result == {"test/test_nn.py": set(), "test/test_torch.py": set()}

    def test_skips_unknown_source_file(self):
        result = _tests_via_profiling({"nn.functional.conv2d"}, {}, {})
        assert result == {}

    def test_skips_test_files_outside_tree(self):
        profiling = {"torch/nn/functional.py": {"test_external": 1.0}}
        test_files = {"test/test_nn.py": {}}
        result = _tests_via_profiling({"nn.functional.conv2d"}, profiling, test_files)
        assert result == {}


class TestSymbolsInFile:
    def test_returns_functions_by_suffix_match(self):
        ext = MagicMock()
        ext.function_locations = {
            "at::native::copy_impl": ("/abs/path/aten/src/ATen/native/Copy.cpp", 140),
            "at::native::add_impl": ("/abs/path/aten/src/ATen/native/Add.cpp", 50),
            "do_thing": ("/abs/path/other/Foo.cpp", 10),
        }

        result = symbols_in_file("aten/src/ATen/native/Copy.cpp", cpp_extractor=ext)
        assert len(result["functions"]) == 1
        assert result["functions"][0]["function"] == "at::native::copy_impl"

    def test_sorted_by_file_then_line(self):
        ext = MagicMock()
        ext.function_locations = {
            "g": ("/foo/Bar.cpp", 50),
            "f": ("/foo/Bar.cpp", 10),
        }
        result = symbols_in_file("foo/Bar.cpp", cpp_extractor=ext)
        assert [m["function"] for m in result["functions"]] == ["f", "g"]


class TestAffectedTool:
    @pytest.fixture
    def stubbed_state(self, monkeypatch):
        """Populate _state with a minimal stub so _do_affected can run."""
        from torchtalk import indexer

        ext = MagicMock()
        ext.get_callers.return_value = []

        monkeypatch.setattr(indexer._state, "bindings", [{"x": 1}])
        monkeypatch.setattr(indexer._state, "cpp_extractor", ext)
        monkeypatch.setattr(indexer._state, "by_cpp_name", {})
        monkeypatch.setattr(indexer._state, "test_classes", {})
        monkeypatch.setattr(indexer._state, "test_files", {})
        monkeypatch.setattr(indexer._state, "opinfo_registry", {})
        monkeypatch.setattr(indexer._state, "opinfo_alias_map", {})
        monkeypatch.setattr(indexer._state, "opinfo_test_files", set())
        monkeypatch.setattr(indexer._state, "test_attr_index", {})

    def test_empty_funcs_returns_message(self, stubbed_state):
        result = asyncio.run(_do_affected(""))
        assert "No functions provided" in result

    def test_runs_with_no_matches(self, stubbed_state):
        result = asyncio.run(_do_affected("nonexistent_kernel"))
        assert "No matching test runs" in result
        assert "nonexistent_kernel" in result

    def test_strips_whitespace_in_csv(self, stubbed_state):
        # `" a , b "` should parse to ["a", "b"] not [" a ", " b "].
        result = asyncio.run(_do_affected(" foo , bar "))
        assert "`foo, bar`" in result
