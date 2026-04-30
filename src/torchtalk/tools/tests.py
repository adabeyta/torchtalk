"""Test infrastructure tool implementations."""

from __future__ import annotations

from pathlib import Path

from ..formatting import create_formatter, relative_path
from ..indexer import _ensure_loaded, _state


def _rel_path(path: str) -> str:
    return relative_path(path, _state.pytorch_source)


def _word_match(query_lower: str, name_lower: str) -> bool:
    """Match on word boundaries to avoid 'add' matching 'padding'."""
    idx = name_lower.find(query_lower)
    if idx == -1:
        return False
    before = name_lower[idx - 1] if idx > 0 else "_"
    after = (
        name_lower[idx + len(query_lower)]
        if idx + len(query_lower) < len(name_lower)
        else "_"
    )
    return not before.isalpha() and not after.isalpha()


async def _do_find_similar_tests(
    query: str, limit: int = 10, focus: str = "all"
) -> str:
    _ensure_loaded("test")

    query_lower = query.lower()
    md = create_formatter()
    md.h2(f"Tests matching: `{query}`")

    matching_funcs = []
    if focus in ("all", "functions"):
        for func_name, locations in _state.test_functions.items():
            if _word_match(query_lower, func_name.lower()):
                for loc in locations:
                    matching_funcs.append(
                        {
                            "name": func_name,
                            "class": loc.get("class"),
                            "file": loc["file"],
                            "line": loc["line"],
                        }
                    )

    matching_classes = []
    if focus in ("all", "classes"):
        for class_name, locations in _state.test_classes.items():
            if _word_match(query_lower, class_name.lower()):
                for loc in locations:
                    matching_classes.append(
                        {
                            "name": class_name,
                            "file": loc["file"],
                            "line": loc["line"],
                            "bases": loc.get("bases", []),
                        }
                    )

    matching_files = []
    if focus in ("all", "files"):
        for file_path, info in _state.test_files.items():
            if query_lower in file_path.lower():
                matching_files.append(
                    {
                        "path": file_path,
                        "classes": len(info.get("classes", [])),
                        "functions": len(info.get("functions", [])),
                    }
                )

    matching_opinfo = []
    if focus in ("all", "functions"):
        for op_name, info in _state.opinfo_registry.items():
            if _word_match(query_lower, op_name.lower()):
                matching_opinfo.append(info)

    total = (
        len(matching_funcs)
        + len(matching_classes)
        + len(matching_files)
        + len(matching_opinfo)
    )

    if total == 0:
        return f"No tests found matching `{query}`."

    md.text(f"Found {total} matches\n")

    if matching_opinfo:
        md.h3(f"OpInfo Definitions ({len(matching_opinfo)})")
        md.text("*Operators with official test metadata:*\n")
        for info in matching_opinfo[:5]:
            md.item(f"`{info['name']}` → `{info['file']}:{info['line']}`")
        if len(matching_opinfo) > 5:
            md.item(f"*... and {len(matching_opinfo) - 5} more*")
        md.blank()

    if matching_funcs:
        md.h3(f"Test Functions ({len(matching_funcs)})")
        for func in matching_funcs[:limit]:
            class_prefix = f"{func['class']}." if func.get("class") else ""
            md.item(f"`{class_prefix}{func['name']}` → `{func['file']}:{func['line']}`")
        if len(matching_funcs) > limit:
            md.item(f"*... and {len(matching_funcs) - limit} more*")
        md.blank()

    if matching_classes:
        md.h3(f"Test Classes ({len(matching_classes)})")
        for cls in matching_classes[:5]:
            bases = f" ({', '.join(cls['bases'][:2])})" if cls.get("bases") else ""
            md.item(f"`{cls['name']}`{bases} → `{cls['file']}:{cls['line']}`")
        if len(matching_classes) > 5:
            md.item(f"*... and {len(matching_classes) - 5} more*")
        md.blank()

    if matching_files:
        md.h3(f"Test Files ({len(matching_files)})")
        for f in matching_files[:5]:
            md.item(f"`{f['path']}` ({f['classes']} classes, {f['functions']} tests)")
        if len(matching_files) > 5:
            md.item(f"*... and {len(matching_files) - 5} more*")

    return md.build()


async def _do_list_test_utils(category: str = "all") -> str:
    _ensure_loaded("test")

    md = create_formatter()
    md.h2("PyTorch Test Utilities")

    utility_info = {
        "torch/testing/_internal/common_utils.py": {
            "name": "common_utils",
            "description": "Core test utilities: TestCase, device/dtype helpers",
            "key_items": [
                "TestCase",
                "run_tests",
                "instantiate_parametrized_tests",
                "IS_CUDA",
            ],
        },
        "torch/testing/_internal/common_device_type.py": {
            "name": "common_device_type",
            "description": "Device-agnostic testing infrastructure",
            "key_items": [
                "instantiate_device_type_tests",
                "ops",
                "onlyCPU",
                "onlyCUDA",
            ],
        },
        "torch/testing/_internal/common_dtype.py": {
            "name": "common_dtype",
            "description": "Data type testing utilities",
            "key_items": ["floating_types", "integral_types", "all_types_and_complex"],
        },
        "torch/testing/_internal/common_cuda.py": {
            "name": "common_cuda",
            "description": "CUDA-specific test utilities",
            "key_items": ["TEST_CUDA", "TEST_MULTIGPU", "TEST_CUDNN"],
        },
        "torch/testing/_internal/opinfo/core.py": {
            "name": "opinfo",
            "description": "Operator test info registry (OpInfo)",
            "key_items": ["OpInfo", "SampleInput", "DecorateInfo"],
        },
        "torch/testing/_comparison.py": {
            "name": "comparison",
            "description": "Tensor comparison and assertions",
            "key_items": ["assert_close", "assert_equal"],
        },
        "torch/testing/_internal/hypothesis_utils.py": {
            "name": "hypothesis_utils",
            "description": "Property-based testing with Hypothesis",
            "key_items": ["tensor_strategy", "dtype_strategy"],
        },
    }

    md.h3("Core Utilities")
    for path, info in utility_info.items():
        if path in _state.test_utilities:
            exists = True
        elif _state.pytorch_source:
            exists = (Path(_state.pytorch_source) / path).exists()
        else:
            exists = False
        status = "✓" if exists else "?"
        md.item(f"**{info['name']}** {status}")
        md.item(f"*{info['description']}*", 1)
        md.item(f"Key: `{', '.join(info['key_items'][:4])}`", 1)
        md.item(f"Path: `{path}`", 1)
        md.blank()

    md.h3("Test Infrastructure Stats")
    if _state.test_files:
        md.item(f"Test files indexed: {len(_state.test_files)}")
        md.item(f"Test classes: {len(_state.test_classes)}")
        md.item(f"Test functions: {len(_state.test_functions)}")
        md.item(f"OpInfo definitions: {len(_state.opinfo_registry)}")
    else:
        md.text("*Test infrastructure not yet indexed*")

    md.h3("Common Test Patterns")
    patterns = [
        (
            "Device-type tests",
            "`@instantiate_device_type_tests`",
            "Run tests across CPU/CUDA",
        ),
        ("Parametrized tests", "`@parametrize`", "Run tests with multiple inputs"),
        ("OpInfo tests", "`@ops(op_db)`", "Test operators using OpInfo metadata"),
        ("Gradient check", "`gradcheck(fn, inputs)`", "Verify autograd correctness"),
        (
            "Assert close",
            "`torch.testing.assert_close(a, b)`",
            "Compare tensors with tolerance",
        ),
    ]
    for name, code, desc in patterns:
        md.item(f"**{name}**: {code}")
        md.item(f"*{desc}*", 1)

    return md.build()


async def _do_test_file_info(file_path: str) -> str:
    _ensure_loaded("test")

    query = file_path.lower()
    matches = []
    for path, info in _state.test_files.items():
        if query in path.lower():
            matches.append((path, info))

    if not matches:
        return f"No test file found matching `{file_path}`."

    md = create_formatter()

    for path, info in matches[:3]:
        md.h2(f"Test File: `{path}`")

        if info.get("classes"):
            md.h3(f"Test Classes ({len(info['classes'])})")
            for cls in info["classes"][:10]:
                bases = (
                    f" extends {', '.join(cls['bases'][:2])}"
                    if cls.get("bases")
                    else ""
                )
                md.item(f"`{cls['name']}`{bases} (line {cls['line']})")

        if info.get("functions"):
            md.h3(f"Test Functions ({len(info['functions'])})")
            for func in info["functions"][:20]:
                class_prefix = f"{func['class']}." if func.get("class") else ""
                md.item(f"`{class_prefix}{func['name']}` (line {func['line']})")
            if len(info["functions"]) > 20:
                md.item(f"*... and {len(info['functions']) - 20} more*")

        md.blank()

    if len(matches) > 3:
        md.text(f"*Showing 3 of {len(matches)} matching files*")

    return md.build()
