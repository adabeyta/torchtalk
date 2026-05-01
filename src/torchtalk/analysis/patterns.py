"""Shared configuration for TorchTalk analyzers."""

CPP_SEARCH_DIRS = [
    "aten/src/ATen/native",
    "torch/csrc",
    "c10",
    "aten/src/ATen/core",
    "aten/src/ATen/cuda",
]

# Vendor backend subdirs of `aten/src/ATen/native/` whose helper functions
# follow the "many helpers, few registered ops" pattern. Symbols here that
# aren't in `native_functions.yaml` still merit indexing for cohort fallback.
VENDOR_BACKEND_DIRS = ("cudnn", "miopen", "mkldnn", "mkl", "mps")
_VENDOR_DIR_MARKERS = tuple(f"/native/{v}/" for v in VENDOR_BACKEND_DIRS)


def is_vendor_path(path: str) -> bool:
    """True if `path` is inside an `aten/src/ATen/native/<vendor>/` subdir."""
    return any(m in path for m in _VENDOR_DIR_MARKERS)

PYTHON_SEARCH_DIRS = [
    "torch/nn",
    "torch/optim",
    "torch/autograd",
    "torch/jit",
    "torch/cuda",
    "torch/backends",
    "torch/fx",
    "torch/_dynamo",
    "torch/_inductor",
    "torch/_decomp",
    "torch/ao",
    "torch/distributed",
    "torch/export",
    "torch/_refs",
    "torch/_prims",
    "torch/nested",
    "torch/mps",
]

TEST_SEARCH_DIRS = [
    "test",
    "torch/testing",
    "torch/testing/_internal",
]

EXCLUDE_PATTERNS = [
    "/test/",
    "/tests/",
    "test_",
    "_test.",
    "/third_party/",
    "/generated/",
    "/build/",
    "__pycache__",
    "/benchmarks/",
    "/examples/",
]

# Patterns to detect C++ binding code
CPP_BINDING_PATTERNS = [
    "TORCH_LIBRARY",
    "PYBIND11_MODULE",
    "pybind11",
    "py::class_<",
    "m.def(",
    "__global__",
    "AT_DISPATCH",
    "c10::Dispatcher",
    "RegisterOperators",
]

# Patterns to detect test files
TEST_CONTENT_PATTERNS = [
    "TestCase",
    "pytest",
    "instantiate_device_type_tests",
    "OpInfo",
    "make_tensor",
    "assert_close",
    "dtype_test",
    "gradcheck",
]

TEST_UTILITY_MODULES = [
    "torch/testing/_internal/common_utils.py",
    "torch/testing/_internal/common_device_type.py",
    "torch/testing/_internal/common_dtype.py",
    "torch/testing/_internal/common_cuda.py",
    "torch/testing/_internal/opinfo/core.py",
    "torch/testing/_internal/opinfo/definitions.py",
    "torch/testing/_internal/hypothesis_utils.py",
    "torch/testing/_comparison.py",
]


def should_exclude(path: str) -> bool:
    return any(p in path.lower() for p in EXCLUDE_PATTERNS)


def should_include_dir(file_path: str, include_dirs: list[str]) -> bool:
    return any(d in file_path for d in include_dirs)


def has_binding_patterns(content: str) -> bool:
    return any(p in content for p in CPP_BINDING_PATTERNS)


def has_test_patterns(content: str) -> bool:
    return any(p in content for p in TEST_CONTENT_PATTERNS)
