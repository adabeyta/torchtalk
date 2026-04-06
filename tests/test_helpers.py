"""Tests for analysis helper functions."""

from torchtalk.analysis.helpers import (
    dedupe_by_key,
    fuzzy_match,
    levenshtein_distance,
    safe_sort_key,
    truncate,
)


class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_edit(self):
        assert levenshtein_distance("cat", "bat") == 1
        assert levenshtein_distance("cat", "cats") == 1
        assert levenshtein_distance("cat", "at") == 1

    def test_symmetry(self):
        assert levenshtein_distance("abc", "xyz") == levenshtein_distance("xyz", "abc")

    def test_pytorch_function_names(self):
        assert levenshtein_distance("matmul", "matmull") == 1
        assert levenshtein_distance("softmax", "softmin") == 2
        assert levenshtein_distance("relu", "selu") == 1


class TestFuzzyMatch:
    def test_exact_match_first(self):
        candidates = ["relu", "relu6", "prelu", "leaky_relu"]
        result = fuzzy_match("relu", candidates)
        assert result[0] == "relu"

    def test_substring_matches(self):
        candidates = ["batch_norm", "layer_norm", "group_norm", "conv2d"]
        result = fuzzy_match("norm", candidates)
        assert "batch_norm" in result
        assert "layer_norm" in result
        assert "conv2d" not in result

    def test_levenshtein_for_typos(self):
        candidates = ["softmax", "softmin", "sigmoid", "relu"]
        result = fuzzy_match("sofmax", candidates)
        assert "softmax" in result

    def test_max_results(self):
        candidates = [f"func_{i}" for i in range(100)]
        result = fuzzy_match("func", candidates, max_results=5)
        assert len(result) == 5

    def test_empty_query_matches_all(self):
        result = fuzzy_match("", ["a", "b", "c"])
        assert len(result) == 3

    def test_case_insensitive(self):
        result = fuzzy_match("RELU", ["relu", "ReLU", "selu"])
        assert "relu" in result or "ReLU" in result


class TestTruncate:
    def test_short_string(self):
        assert truncate("hello", 10) == "hello"

    def test_exact_length(self):
        assert truncate("hello", 5) == "hello"

    def test_long_string(self):
        result = truncate("hello world this is long", 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_empty_string(self):
        assert truncate("", 10) == ""

    def test_none(self):
        assert truncate(None, 10) == ""


class TestSafeSortKey:
    def test_none_sorts_last(self):
        items = ["b", None, "a", "c"]
        result = sorted(items, key=safe_sort_key)
        assert result[-1] is None

    def test_strings_sort_normally(self):
        items = ["c", "a", "b"]
        result = sorted(items, key=safe_sort_key)
        assert result == ["a", "b", "c"]


class TestDedupeByKey:
    def test_removes_duplicates(self):
        items = [
            {"name": "add", "file": "a.cpp"},
            {"name": "add", "file": "b.cpp"},
            {"name": "mul", "file": "c.cpp"},
        ]
        result = dedupe_by_key(items, "name")
        assert len(result) == 2

    def test_preserves_order(self):
        items = [
            {"name": "c"},
            {"name": "a"},
            {"name": "b"},
        ]
        result = dedupe_by_key(items, "name")
        assert [r["name"] for r in result] == ["c", "a", "b"]

    def test_skips_missing_key(self):
        items = [
            {"name": "a"},
            {"other": "x"},
            {"name": "b"},
        ]
        result = dedupe_by_key(items, "name")
        assert len(result) == 2

    def test_empty_list(self):
        assert dedupe_by_key([], "name") == []
