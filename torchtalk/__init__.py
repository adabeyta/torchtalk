"""
TorchTalk - Cross-language binding analysis for PyTorch codebases.

Provides MCP tools for Claude Code to understand PyTorch's
Python → C++ → CUDA dispatch architecture.
"""

__version__ = "3.0.0"
__author__ = "Adrian Abeyta"

from torchtalk.analysis.binding_detector import BindingDetector
from torchtalk.analysis.repo_analyzer import RepoAnalyzer

__all__ = ["BindingDetector", "RepoAnalyzer"]
