"""Tests for word boundary matching in test search."""

from torchtalk.tools.tests import _word_match


class TestWordMatch:
    def test_exact_word(self):
        assert _word_match("add", "test_add")
        assert _word_match("add", "add_tensor")

    def test_rejects_substring(self):
        assert not _word_match("add", "padding")
        assert not _word_match("add", "loading")
        assert not _word_match("add", "address")

    def test_underscore_boundary(self):
        assert _word_match("add", "test_add_relu")
        assert _word_match("norm", "batch_norm_cpu")

    def test_start_of_string(self):
        assert _word_match("add", "add_something")

    def test_end_of_string(self):
        assert _word_match("add", "test_add")

    def test_whole_string(self):
        assert _word_match("relu", "relu")

    def test_case_sensitivity(self):
        assert _word_match("add", "test_add")
        assert not _word_match("Add", "test_add")

    def test_digit_boundary(self):
        assert _word_match("conv", "conv2d")

    def test_no_match(self):
        assert not _word_match("matmul", "softmax")
