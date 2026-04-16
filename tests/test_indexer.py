"""Tests for indexer data structures and fuzzy matching."""

from torchtalk.indexer import ServerState, _build_indexes, _fuzzy_find


class TestServerState:
    def test_default_empty(self):
        state = ServerState()
        assert state.bindings == []
        assert state.native_functions == {}
        assert state.pytorch_source is None

    def test_build_indexes(self):
        state = ServerState()
        state.bindings = [
            {"python_name": "add", "cpp_name": "at::add", "dispatch_key": "CPU"},
            {"python_name": "add", "cpp_name": "at::add", "dispatch_key": "CUDA"},
            {"python_name": "mul", "cpp_name": "at::mul", "dispatch_key": "CPU"},
        ]
        _build_indexes(state)

        assert "add" in state.by_python_name
        assert len(state.by_python_name["add"]) == 2
        assert "at::add" in state.by_cpp_name
        assert "CPU" in state.by_dispatch_key
        assert "CUDA" in state.by_dispatch_key


class TestFuzzyFind:
    def test_exact_match(self):
        data = {"relu": [{"name": "relu"}]}
        result = _fuzzy_find("relu", data)
        assert result is not None
        assert result[0]["name"] == "relu"

    def test_suffix_match(self):
        data = {"at::native::relu": [{"name": "relu"}]}
        result = _fuzzy_find("relu", data)
        assert result is not None

    def test_contains_match(self):
        data = {"mkldnn_relu": [{"name": "mkldnn_relu"}]}
        result = _fuzzy_find("relu", data)
        assert result is not None

    def test_no_match(self):
        data = {"softmax": [{"name": "softmax"}]}
        result = _fuzzy_find("nonexistent_function_xyz", data)
        assert result is None

    def test_returns_list(self):
        data = {"relu": {"name": "relu"}}
        result = _fuzzy_find("relu", data)
        assert isinstance(result, list)

    def test_levenshtein_match(self):
        data = {"softmax": [{"name": "softmax"}]}
        result = _fuzzy_find("sofmax", data)
        assert result is not None
