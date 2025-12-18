"""
Test BindingDetector against the actual PyTorch repository.

Requires PYTORCH_SOURCE or PYTORCH_PATH environment variable to be set.

Usage:
    PYTORCH_SOURCE=/path/to/pytorch pytest tests/test_binding_detector_pytorch.py -v
"""

import os
from pathlib import Path

import pytest

from torchtalk.analysis.binding_detector import BindingDetector, BindingType


def get_pytorch_path() -> Path | None:
    """Get PyTorch path from environment variable."""
    for var in ("PYTORCH_SOURCE", "PYTORCH_PATH"):
        if path := os.environ.get(var):
            p = Path(path)
            if p.exists() and (p / "torch").exists():
                return p
    return None


PYTORCH_PATH = get_pytorch_path()

pytestmark = pytest.mark.skipif(
    PYTORCH_PATH is None,
    reason="PYTORCH_SOURCE or PYTORCH_PATH environment variable not set",
)


@pytest.fixture
def detector():
    """Create a BindingDetector instance."""
    return BindingDetector()


class TestPybind11Detection:
    """Tests for pybind11 pattern detection."""

    def test_detects_bindings_in_module_cpp(self, detector):
        """Should detect pybind11 bindings in torch/csrc/Module.cpp."""
        test_file = PYTORCH_PATH / "torch/csrc/Module.cpp"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        content = test_file.read_text(errors="replace")
        graph = detector.detect_bindings(str(test_file), content)

        assert len(graph.bindings) > 0, "Should find pybind11 bindings"


class TestTorchLibraryDetection:
    """Tests for TORCH_LIBRARY pattern detection."""

    def test_detects_torch_library_in_rnn(self, detector):
        """Should detect TORCH_LIBRARY bindings in RNN.cpp."""
        test_file = PYTORCH_PATH / "aten/src/ATen/native/RNN.cpp"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        content = test_file.read_text(errors="replace")
        graph = detector.detect_bindings(str(test_file), content)

        torch_lib_bindings = [
            b for b in graph.bindings if "torch" in b.binding_type.lower()
        ]
        assert len(torch_lib_bindings) > 0, "Should find TORCH_LIBRARY bindings"


class TestCudaKernelDetection:
    """Tests for CUDA kernel detection."""

    def test_detects_cuda_kernels(self, detector):
        """Should detect __global__ CUDA kernels in .cu files."""
        cuda_dir = PYTORCH_PATH / "aten/src/ATen/native/cuda"
        if not cuda_dir.exists():
            pytest.skip(f"CUDA directory not found: {cuda_dir}")

        found_kernels = False
        for cu_file in list(cuda_dir.glob("*.cu"))[:20]:
            content = cu_file.read_text(errors="replace")
            if "__global__" not in content:
                continue

            graph = detector.detect_bindings(str(cu_file), content)
            if graph.cuda_kernels:
                found_kernels = True
                break

        assert found_kernels, "Should find CUDA kernels in at least one .cu file"


class TestAtDispatchDetection:
    """Tests for AT_DISPATCH macro detection."""

    def test_detects_at_dispatch_macros(self, detector):
        """Should detect AT_DISPATCH macros in native ops."""
        native_dir = PYTORCH_PATH / "aten/src/ATen/native"
        if not native_dir.exists():
            pytest.skip(f"Native directory not found: {native_dir}")

        found_dispatch = False
        for cpp_file in list(native_dir.glob("*.cpp"))[:30]:
            content = cpp_file.read_text(errors="replace")
            if "AT_DISPATCH" not in content:
                continue

            graph = detector.detect_bindings(str(cpp_file), content)
            at_dispatch = [
                b
                for b in graph.bindings
                if b.binding_type == BindingType.AT_DISPATCH.value
            ]
            if at_dispatch:
                found_dispatch = True
                break

        assert found_dispatch, "Should find AT_DISPATCH macros"


class TestDirectoryScan:
    """Tests for full directory scanning."""

    def test_scans_autograd_directory(self, detector):
        """Should scan torch/csrc/autograd and find bindings."""
        scan_dir = PYTORCH_PATH / "torch/csrc/autograd"
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        graph = detector.detect_bindings_in_directory(str(scan_dir))

        assert len(graph.bindings) > 0, "Should find bindings in autograd directory"

    def test_categorizes_bindings_by_type(self, detector):
        """Should categorize bindings by their type."""
        scan_dir = PYTORCH_PATH / "torch/csrc/autograd"
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        graph = detector.detect_bindings_in_directory(str(scan_dir))

        by_type = {}
        for b in graph.bindings:
            by_type[b.binding_type] = by_type.get(b.binding_type, 0) + 1

        assert len(by_type) > 1, "Should find multiple binding types"
