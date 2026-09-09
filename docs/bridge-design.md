# Bridge design: cross-package references

TorchTalk indexes one package at a time (PyTorch, vLLM, torchvision, ...).
The *bridge* connects two indexed packages so questions like "which vLLM
kernels call this ATen op?" or "what breaks in vLLM if `torch.nn.Module`
changes?" can be answered from static analysis alone.

This document fixes the data model. Phase C (`Workspace`, resolvers, cross
tools) builds on it; PR-4 ships the primitive and the cheapest edge.

## One primitive: `ExternalRef`

```python
@dataclass(frozen=True)
class ExternalRef:
    from_symbol: str  # qualified symbol in the *referencing* package
    to_name: str  # name as written ("torch.nn.Module", "at::empty")
    kind: str  # import | op | cpp | base_class | provides | version_pin
    evidence: str  # "path:line" — always points at real source
    confidence: float  # 1.0 for syntactic facts, lower for heuristics
    to_package: str  # harness name this ref resolves against ("pytorch")
```

`to_package` is an addition to the five-field sketch from the design notes:
collection already knows which `depends_on` entry a name belongs to, and
recording it saves the resolver a second lookup.

A bridge is then simply: for every `ExternalRef` in package A whose
`to_package == B`, look up `to_name` in B's symbol table. Unresolved refs
are kept (they are the "vLLM uses a symbol PyTorch no longer exports"
signal), not dropped.

## Kinds and their resolvers

| kind          | collected from                                   | resolver list (manifest)      | status |
|---------------|--------------------------------------------------|-------------------------------|--------|
| `import`      | top-level and nested `import` / `from ... import` statements         | `python_package_roots` of dep | PR-4   |
| `op`          | `torch.ops.aten.X`, `torch.X` calls               | `[python] op_namespaces`      | C2     |
| `cpp`         | `at::X`, `c10::X` in C++ sources                  | `[bridge] cpp_namespaces`     | C2     |
| `base_class`  | `class Foo(torch.nn.Module)`                      | `[bridge] base_class_namespaces` | C2  |
| `provides`    | `TORCH_LIBRARY` / `register_op` registrations     | (direction flipped: this package *defines* `to_name`) | C2 |
| `version_pin` | `requirements*.txt`, `pyproject.toml`             | none — package-level edge     | C2     |

Everything is a manifest list, not code: adding a framework means listing
which namespaces belong to its dependency, never adding a new edge class.

### Why imports first

Import edges are pure syntax, already parsed by `PythonAnalyzer`, and exist
in every Python package. They give the bridge a smoke test on day one
(`external_refs > 0` for vLLM) and a coarse dependency map (which vLLM
modules touch `torch.distributed` vs `torch.nn`) before any resolver exists.

### Direction

Refs always point *out* of the package being indexed. A registration
(`kind="provides"`) is the same record with the meaning flipped: vLLM
*provides* `_C::rms_norm` into the `torch.ops` namespace. The Workspace
joins A's `provides` with B's `op` refs to answer "who implements this".

## Manifest fields

```toml
[package]
depends_on = ["pytorch"]          # bridge targets, in resolution order

[python.op_namespaces]
torch = "aten"                    # torch.X  -> aten::X

[bridge]
cpp_namespaces = ["at", "c10", "torch"]
base_class_namespaces = ["torch.nn", "torch.autograd", "torch.optim"]
```

`torch-extension.toml` carries these, so every extension profile inherits
them via `extends`.

## What is deliberately skipped

- **dynamo / `torch.compile` decorators** — runtime behaviour, no static
  target symbol.
- **Attribute chains through aliases** (`F = torch.nn.functional; F.relu`)
  — resolved later by the existing `alias_map`, not by the collector.
- **Third parties not in `depends_on`** (`numpy`, `triton`) — no symbol
  table to resolve against; listing them would only add noise.

## Storage

Python analysis is not cached in the JSON index (it runs at load), so
external refs live in `ServerState.external_refs` as plain dicts and are
recomputed per load. The count is reported as `external_refs` in the
stats returned by `build_index` / `update_index` and in `torchtalk index
build` output. The snapshot schema (v3) is untouched; resolved bridge
results get their own cache and a v4 schema in C2.

## Roadmap

- **PR-4 (this):** `ExternalRef`, `collect_import_refs`, `[bridge]` fields,
  stats + CLI count.
- **C1:** `Workspace` holding N packages; tools take `package=`.
- **C2:** op/cpp/base_class/provides resolvers, version-pin edge,
  `bridge(symbol)`, `trace`/`affected --across`, bridge cache + snapshot v4.
