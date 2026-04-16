"""Python module tool implementations."""

from __future__ import annotations

from ..analysis.helpers import fuzzy_match
from ..formatting import create_formatter, relative_path
from ..indexer import _ensure_loaded, _state


def _rel_path(path: str) -> str:
    return relative_path(path, _state.pytorch_source)


async def _do_trace_module(module_name: str, focus: str = "methods") -> str:
    _ensure_loaded()

    if not _state.py_classes:
        return "Python module analysis not available. Ensure PyTorch source is loaded."

    search_name = module_name.split(".")[-1]

    exact_matches = []
    substring_matches = []
    for name, classes in _state.py_classes.items():
        if search_name.lower() == name.lower():
            exact_matches.extend(classes)
        elif search_name.lower() in name.lower():
            substring_matches.extend(classes)
    matches = exact_matches if exact_matches else substring_matches

    if not matches:
        # Try fuzzy match
        similar = fuzzy_match(
            search_name, list(_state.py_classes.keys()), max_results=5
        )
        if similar:
            return f"Module `{module_name}` not found. Similar: {', '.join(similar)}"
        return f"Module `{module_name}` not found."

    md = create_formatter()
    md.h2(f"Module: `{module_name}`")

    for cls in matches[:1]:
        md.h3(f"{cls.qualified_name}")
        path = _rel_path(cls.file_path)
        md.text(f"**File:** `{path}:{cls.line_number}`")

        if focus == "full":
            if cls.bases:
                md.text(f"**Bases:** {', '.join(cls.bases)}")

            if cls.is_module:
                md.text("**Type:** torch.nn.Module")

            if cls.docstring:
                doc = (
                    cls.docstring[:150] + "..."
                    if len(cls.docstring) > 150
                    else cls.docstring
                )
                md.text(f"*{doc}*")

        if cls.methods:
            md.text("**Methods:**")
            for method in cls.methods[:10]:
                sig = method.signature or "()"
                md.item(f"`{method.name}{sig}`")
            if len(cls.methods) > 10:
                md.item(f"*... and {len(cls.methods) - 10} more*")

        md.blank()

    if len(matches) > 1:
        md.text(f"*Showing top match of {len(matches)} total.*")

    return md.build()


async def _do_list_modules(category: str = "nn") -> str:
    _ensure_loaded()

    if not _state.py_classes:
        return "Python module analysis not available."

    md = create_formatter()

    if category == "nn":
        md.h2("Neural Network Modules (torch.nn)")
        modules = sorted(_state.nn_modules, key=lambda x: x.name)
        md.text(f"Found {len(modules)} nn.Module subclasses\n")

        # Group by type
        layers = [
            m
            for m in modules
            if not any(x in m.name for x in ["Loss", "Container", "Sequential"])
        ]
        losses = [m for m in modules if "Loss" in m.name]
        containers = [
            m
            for m in modules
            if any(
                x in m.name
                for x in ["Container", "Sequential", "ModuleList", "ModuleDict"]
            )
        ]

        if layers:
            md.h3(f"Layers ({len(layers)})")
            for m in layers[:20]:
                md.item(f"`{m.name}` - {m.qualified_name}")
            if len(layers) > 20:
                md.item(f"*... and {len(layers) - 20} more.*")

        if losses:
            md.h3(f"Loss Functions ({len(losses)})")
            for m in losses:
                md.item(f"`{m.name}`")

        if containers:
            md.h3(f"Containers ({len(containers)})")
            for m in containers:
                md.item(f"`{m.name}`")

    elif category == "optim":
        md.h2("Optimizers (torch.optim)")
        optim_classes = [
            cls
            for name, classes in _state.py_classes.items()
            for cls in classes
            if "optim" in cls.qualified_name.lower() and not name.startswith("_")
        ]
        for cls in sorted(optim_classes, key=lambda x: x.name)[:20]:
            md.item(f"`{cls.name}` - {cls.qualified_name}")

    elif category == "all":
        md.h2("All Python Classes")
        md.text(f"Total: {sum(len(v) for v in _state.py_classes.values())} classes\n")
        for name in sorted(_state.py_classes.keys())[:50]:
            md.item(f"`{name}`")
        if len(_state.py_classes) > 50:
            md.text(f"\n*Showing 50 of {len(_state.py_classes)} classes*")

    else:
        # Search query
        md.h2(f"Search: '{category}'")
        matches = []
        query_lower = category.lower()
        for name, classes in _state.py_classes.items():
            if query_lower in name.lower():
                matches.extend(classes)

        if matches:
            for cls in matches[:20]:
                path = _rel_path(cls.file_path)
                md.item(f"`{cls.name}` - `{path}:{cls.line_number}`")
            if len(matches) > 20:
                md.text(f"\n*Showing 20 of {len(matches)} matches*")
        else:
            md.text("No matches found.")

    return md.build()
