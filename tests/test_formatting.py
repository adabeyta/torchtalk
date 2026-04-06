"""Tests for response formatters."""

from torchtalk.formatting import (
    CompactText,
    Markdown,
    create_formatter,
    set_formatter_mode,
)


class TestCompactText:
    def test_h2_uses_brackets(self):
        md = CompactText()
        md.h2("Title")
        assert md.build() == "[Title]"

    def test_h3_plain_text(self):
        md = CompactText()
        md.h3("Subtitle")
        assert md.build() == "Subtitle"

    def test_blank_deduplicates(self):
        md = CompactText()
        md.text("a").blank().blank().blank().text("b")
        assert md.build() == "a\n\nb"

    def test_table_skips_headers(self):
        md = CompactText()
        md.table(["A", "B"], [["1", "2"], ["3", "4"]])
        result = md.build()
        assert "A" not in result
        assert "1  2" in result

    def test_item_indentation(self):
        md = CompactText()
        md.item("top").item("nested", indent=1)
        result = md.build()
        assert "- top" in result
        assert "  - nested" in result


class TestMarkdown:
    def test_h2_uses_hashes(self):
        md = Markdown()
        md.h2("Title")
        assert "## Title" in md.build()

    def test_table_has_pipes(self):
        md = Markdown()
        md.table(["A", "B"], [["1", "2"]])
        result = md.build()
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result

    def test_bold(self):
        md = Markdown()
        md.bold("Label", "value")
        assert "**Label:** value" in md.build()

    def test_codeblock(self):
        md = Markdown()
        md.codeblock("print('hi')", "python")
        result = md.build()
        assert "```python" in result
        assert "print('hi')" in result


class TestFormatterFactory:
    def test_default_is_compact(self):
        f = create_formatter()
        assert isinstance(f, CompactText)

    def test_switch_to_markdown(self):
        set_formatter_mode("markdown")
        f = create_formatter()
        assert isinstance(f, Markdown)
        set_formatter_mode("compact")

    def test_invalid_mode_raises(self):
        try:
            set_formatter_mode("invalid")
            assert False, "Should have raised"
        except ValueError:
            pass
