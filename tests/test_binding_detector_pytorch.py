#!/usr/bin/env python3
"""
Test the enhanced BindingDetector against the actual PyTorch repository.
"""

import sys
from pathlib import Path

# Add torchtalk to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchtalk.analysis.binding_detector import BindingDetector, BindingType

PYTORCH_PATH = Path("/myworkspace/pytorch")


def test_pybind11_detection():
    """Test detection of pybind11 patterns in PyTorch"""
    print("\n" + "="*60)
    print("TEST: pybind11 Detection")
    print("="*60)

    detector = BindingDetector()

    # Find a file with PYBIND11_MODULE
    test_files = [
        PYTORCH_PATH / "torch/csrc/Module.cpp",
        PYTORCH_PATH / "torch/csrc/autograd/python_variable.cpp",
    ]

    for test_file in test_files:
        if test_file.exists():
            print(f"\nParsing: {test_file.name}")
            content = test_file.read_text(errors='replace')

            graph = detector.detect_bindings(str(test_file), content)

            print(f"  Found {len(graph.bindings)} bindings")

            # Show first few bindings
            for b in graph.bindings[:5]:
                print(f"    - [{b.binding_type}] {b.python_name} -> {b.cpp_name}")

            if len(graph.bindings) > 5:
                print(f"    ... and {len(graph.bindings) - 5} more")

            return len(graph.bindings) > 0

    print("  No test files found")
    return False


def test_torch_library_detection():
    """Test detection of TORCH_LIBRARY patterns"""
    print("\n" + "="*60)
    print("TEST: TORCH_LIBRARY Detection")
    print("="*60)

    detector = BindingDetector()

    # Use a known file that has TORCH_LIBRARY_FRAGMENT
    test_file = PYTORCH_PATH / "aten/src/ATen/native/RNN.cpp"

    if not test_file.exists():
        print(f"  Test file not found: {test_file}")
        return False

    print(f"Parsing: {test_file.relative_to(PYTORCH_PATH)}")
    content = test_file.read_text(errors='replace')
    graph = detector.detect_bindings(str(test_file), content)

    torch_lib_bindings = [b for b in graph.bindings
                           if 'torch' in b.binding_type.lower()]

    if torch_lib_bindings:
        print(f"  Found {len(torch_lib_bindings)} TORCH_LIBRARY bindings")

        for b in torch_lib_bindings[:5]:
            dispatch = f" [{b.dispatch_key}]" if b.dispatch_key else ""
            print(f"    - {b.python_name}{dispatch} -> {b.cpp_name}")

        if len(torch_lib_bindings) > 5:
            print(f"    ... and {len(torch_lib_bindings) - 5} more")
        return True

    print("  No TORCH_LIBRARY bindings found")
    return False


def test_cuda_kernel_detection():
    """Test detection of CUDA kernels"""
    print("\n" + "="*60)
    print("TEST: CUDA Kernel Detection")
    print("="*60)

    detector = BindingDetector()

    # Find CUDA files
    cuda_dir = PYTORCH_PATH / "aten/src/ATen/native/cuda"

    if not cuda_dir.exists():
        print(f"  Directory not found: {cuda_dir}")
        return False

    found_any = False
    for cu_file in list(cuda_dir.glob("*.cu"))[:20]:
        content = cu_file.read_text(errors='replace')

        if '__global__' in content:
            print(f"\nParsing: {cu_file.name}")
            graph = detector.detect_bindings(str(cu_file), content)

            if graph.cuda_kernels:
                found_any = True
                print(f"  Found {len(graph.cuda_kernels)} CUDA kernels")

                for k in graph.cuda_kernels[:5]:
                    callers = f" (called by: {', '.join(k.called_by[:2])})" if k.called_by else ""
                    print(f"    - __global__ {k.name}{callers}")

                if len(graph.cuda_kernels) > 5:
                    print(f"    ... and {len(graph.cuda_kernels) - 5} more")
                break

    return found_any


def test_at_dispatch_detection():
    """Test detection of AT_DISPATCH macros"""
    print("\n" + "="*60)
    print("TEST: AT_DISPATCH Detection")
    print("="*60)

    detector = BindingDetector()

    native_dir = PYTORCH_PATH / "aten/src/ATen/native"

    if not native_dir.exists():
        print(f"  Directory not found: {native_dir}")
        return False

    found_any = False
    for cpp_file in list(native_dir.glob("*.cpp"))[:30]:
        content = cpp_file.read_text(errors='replace')

        if 'AT_DISPATCH' in content:
            print(f"\nParsing: {cpp_file.name}")
            graph = detector.detect_bindings(str(cpp_file), content)

            at_dispatch_bindings = [b for b in graph.bindings
                                     if b.binding_type == BindingType.AT_DISPATCH.value]

            if at_dispatch_bindings:
                found_any = True
                print(f"  Found {len(at_dispatch_bindings)} AT_DISPATCH macros")

                for b in at_dispatch_bindings[:5]:
                    print(f"    - {b.python_name} in {b.cpp_name}")
                    if b.signature:
                        print(f"      {b.signature}")

                if len(at_dispatch_bindings) > 5:
                    print(f"    ... and {len(at_dispatch_bindings) - 5} more")
                break

    return found_any


def test_full_directory_scan():
    """Test scanning a portion of PyTorch"""
    print("\n" + "="*60)
    print("TEST: Directory Scan (torch/csrc subset)")
    print("="*60)

    detector = BindingDetector()

    # Scan a subset to keep it fast
    scan_dir = PYTORCH_PATH / "torch/csrc/autograd"

    if not scan_dir.exists():
        print(f"  Directory not found: {scan_dir}")
        return False

    print(f"Scanning: {scan_dir}")
    graph = detector.detect_bindings_in_directory(str(scan_dir))

    print(f"\nResults:")
    print(f"  Total bindings: {len(graph.bindings)}")
    print(f"  CUDA kernels: {len(graph.cuda_kernels)}")

    # Count by type
    by_type = {}
    for b in graph.bindings:
        by_type[b.binding_type] = by_type.get(b.binding_type, 0) + 1

    print(f"\n  By type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {t}: {count}")

    # Count by dispatch key
    by_dispatch = {}
    for b in graph.bindings:
        key = b.dispatch_key or "none"
        by_dispatch[key] = by_dispatch.get(key, 0) + 1

    if len(by_dispatch) > 1:
        print(f"\n  By dispatch key:")
        for k, count in sorted(by_dispatch.items(), key=lambda x: -x[1]):
            print(f"    {k}: {count}")

    return len(graph.bindings) > 0


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# Testing BindingDetector on PyTorch Repository")
    print("#"*60)

    if not PYTORCH_PATH.exists():
        print(f"\nERROR: PyTorch not found at {PYTORCH_PATH}")
        return 1

    results = {}

    results["pybind11"] = test_pybind11_detection()
    results["torch_library"] = test_torch_library_detection()
    results["cuda_kernels"] = test_cuda_kernel_detection()
    results["at_dispatch"] = test_at_dispatch_detection()
    results["directory_scan"] = test_full_directory_scan()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
