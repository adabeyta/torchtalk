"""Formatting utilities for MCP tool responses.

Provides a protocol-based formatter hierarchy:
  - Markdown: Full markdown output for human-facing contexts.
  - CompactText: Minimal plain-text output optimized for LLM token efficiency.

Use create_formatter() factory to get the active formatter.
"""

from typing import Protocol


class ResponseFormatter(Protocol):
    """Interface all formatters must implement."""

    def h2(self, text: str) -> "ResponseFormatter": ...
    def h3(self, text: str) -> "ResponseFormatter": ...
    def bold(self, label: str, value: str) -> "ResponseFormatter": ...
    def code(self, label: str, value: str) -> "ResponseFormatter": ...
    def item(self, text: str, indent: int = 0) -> "ResponseFormatter": ...
    def text(self, text: str) -> "ResponseFormatter": ...
    def blank(self) -> "ResponseFormatter": ...
    def table(
        self, headers: list[str], rows: list[list[str]]
    ) -> "ResponseFormatter": ...
    def codeblock(self, code: str, lang: str = "") -> "ResponseFormatter": ...
    def build(self) -> str: ...


class Markdown:
    """Full markdown builder for human-facing output."""

    def __init__(self):
        self._lines: list[str] = []

    def h2(self, text: str) -> "Markdown":
        self._lines.append(f"## {text}\n")
        return self

    def h3(self, text: str) -> "Markdown":
        self._lines.append(f"### {text}\n")
        return self

    def bold(self, label: str, value: str) -> "Markdown":
        self._lines.append(f"**{label}:** {value}")
        return self

    def code(self, label: str, value: str) -> "Markdown":
        self._lines.append(f"**{label}:** `{value}`")
        return self

    def item(self, text: str, indent: int = 0) -> "Markdown":
        prefix = "  " * indent
        self._lines.append(f"{prefix}- {text}")
        return self

    def text(self, text: str) -> "Markdown":
        self._lines.append(text)
        return self

    def blank(self) -> "Markdown":
        self._lines.append("")
        return self

    def table(self, headers: list[str], rows: list[list[str]]) -> "Markdown":
        self._lines.append("| " + " | ".join(headers) + " |")
        self._lines.append("|" + "|".join("-" * (len(h) + 2) for h in headers) + "|")
        for row in rows:
            self._lines.append("| " + " | ".join(row) + " |")
        return self

    def codeblock(self, code: str, lang: str = "") -> "Markdown":
        self._lines.append(f"```{lang}")
        self._lines.append(code)
        self._lines.append("```")
        return self

    def build(self) -> str:
        return "\n".join(self._lines)


class CompactText:
    """Minimal plain-text builder optimized for LLM token efficiency."""

    def __init__(self):
        self._lines: list[str] = []

    def h2(self, text: str) -> "CompactText":
        self._lines.append(f"[{text}]")
        return self

    def h3(self, text: str) -> "CompactText":
        self._lines.append(text)
        return self

    def bold(self, label: str, value: str) -> "CompactText":
        self._lines.append(f"{label}: {value}")
        return self

    def code(self, label: str, value: str) -> "CompactText":
        self._lines.append(f"{label}: {value}")
        return self

    def item(self, text: str, indent: int = 0) -> "CompactText":
        prefix = "  " * indent
        self._lines.append(f"{prefix}- {text}")
        return self

    def text(self, text: str) -> "CompactText":
        self._lines.append(text)
        return self

    def blank(self) -> "CompactText":
        if self._lines and self._lines[-1] != "":
            self._lines.append("")
        return self

    def table(self, headers: list[str], rows: list[list[str]]) -> "CompactText":
        for row in rows:
            self._lines.append("  ".join(row))
        return self

    def codeblock(self, code: str, lang: str = "") -> "CompactText":
        self._lines.append(code)
        return self

    def build(self) -> str:
        return "\n".join(self._lines)


_formatter_mode: str = "compact"


def set_formatter_mode(mode: str) -> None:
    """Set the active formatter mode ('compact' or 'markdown')."""
    global _formatter_mode
    if mode not in ("compact", "markdown"):
        raise ValueError(
            f"Unknown formatter mode: {mode!r}. Use 'compact' or 'markdown'."
        )
    _formatter_mode = mode


def create_formatter() -> ResponseFormatter:
    """Factory: return a formatter instance based on the active mode."""
    if _formatter_mode == "markdown":
        return Markdown()
    return CompactText()


def relative_path(full_path: str, base: str | None = None) -> str:
    """Convert absolute path to relative, stripping common prefixes."""
    if not full_path:
        return ""

    path = full_path
    prefixes = ["/myworkspace/pytorch/", "pytorch/"]
    if base:
        prefixes.insert(0, base.rstrip("/") + "/")

    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def coverage_note(extractor) -> str:
    """Return a one-line C++ TU coverage summary, or '' when empty."""
    cov = extractor.coverage_summary()
    if not cov:
        return ""
    ok = cov.get("ok", 0)
    unsupported = cov.get("unsupported_language", 0)
    parse_failed = cov.get("parse_failed", 0)
    filtered = cov.get("filtered", 0)
    total = ok + unsupported + parse_failed + filtered
    if total == 0:
        return ""
    bits = []
    if unsupported:
        bits.append(f"{unsupported:,} CUDA/.mm/.cc")
    if parse_failed:
        bits.append(f"{parse_failed:,} parse-failed")
    unindexed = f"; unindexed: {', '.join(bits)}" if bits else ""
    return f"Coverage: {ok:,} of {total:,} C++ TUs indexed{unindexed}."
