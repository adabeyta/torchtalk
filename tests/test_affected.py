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

    def test_no_opinfo_match_skips_opinfo_files(self, extractor):
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
        assert result["test_runs"] == []

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
