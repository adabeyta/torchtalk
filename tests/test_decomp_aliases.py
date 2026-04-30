"""Unit tests for decomp/refs alias scraping."""

from __future__ import annotations

from torchtalk.analysis.decomp_aliases import extract_decomp_aliases


def _write_decomp(tmp_path, body: str) -> None:
    (tmp_path / "torch/_decomp").mkdir(parents=True)
    (tmp_path / "torch/_decomp/decompositions.py").write_text(body)


def _write_refs(tmp_path, body: str) -> None:
    (tmp_path / "torch/_refs").mkdir(parents=True)
    (tmp_path / "torch/_refs/__init__.py").write_text(body)


class TestExtractDecompAliases:
    def test_single_op_decorator(self, tmp_path):
        _write_decomp(
            tmp_path,
            "@register_decomposition(aten.foo)\ndef foo_decomp(a, b):\n    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out == {"foo": ["foo_decomp"], "foo_decomp": ["foo"]}

    def test_strips_default_overload_tag(self, tmp_path):
        _write_decomp(
            tmp_path,
            "@register_decomposition(aten.bar.default)\ndef bar_decomp(x):\n    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out == {"bar": ["bar_decomp"], "bar_decomp": ["bar"]}

    def test_list_of_ops(self, tmp_path):
        _write_decomp(
            tmp_path,
            "@register_decomposition([aten.x, aten.y, aten.z])\n"
            "def common(*args):\n"
            "    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        # All three aten ops alias to `common` and to each other transitively.
        assert set(out["common"]) == {"x", "y", "z"}
        assert "common" in out["x"]
        assert "common" in out["y"]
        assert "common" in out["z"]

    def test_attribute_call(self, tmp_path):
        # `@torch._decomp.register_decomposition(aten.foo)` style
        _write_decomp(
            tmp_path,
            "@torch._decomp.register_decomposition(aten.foo)\n"
            "def foo_impl():\n"
            "    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out["foo"] == ["foo_impl"]

    def test_skips_unrelated_decorators(self, tmp_path):
        _write_decomp(
            tmp_path,
            "@some_other_decorator(aten.foo)\ndef unrelated():\n    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out == {}

    def test_ignores_non_aten_args(self, tmp_path):
        _write_decomp(
            tmp_path,
            "@register_decomposition(prim.foo)\ndef x():\n    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out == {}

    def test_self_alias_skipped(self, tmp_path):
        # A decomp function named the same as the aten op shouldn't self-link.
        _write_decomp(
            tmp_path,
            "@register_decomposition(aten.foo)\ndef foo():\n    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out == {}

    def test_merges_decomp_and_refs(self, tmp_path):
        _write_decomp(
            tmp_path,
            "@register_decomposition(aten.foo)\ndef foo_decomp():\n    pass\n",
        )
        _write_refs(
            tmp_path,
            "@register_decomposition(aten.bar)\ndef bar_ref():\n    pass\n",
        )
        out = extract_decomp_aliases(tmp_path)
        assert out["foo"] == ["foo_decomp"]
        assert out["bar"] == ["bar_ref"]

    def test_missing_files_returns_empty(self, tmp_path):
        out = extract_decomp_aliases(tmp_path)
        assert out == {}

    def test_syntax_error_skipped(self, tmp_path):
        _write_decomp(tmp_path, "this is not python {{{")
        out = extract_decomp_aliases(tmp_path)
        assert out == {}
