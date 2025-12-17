"""Markdown formatting utilities."""

from typing import List, Optional


class Markdown:
    """Simple markdown builder."""

    def __init__(self):
        self._lines: List[str] = []

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

    def table(self, headers: List[str], rows: List[List[str]]) -> "Markdown":
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


def relative_path(full_path: str, base: Optional[str] = None) -> str:
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


