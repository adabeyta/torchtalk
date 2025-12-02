import logging
import re
from collections import defaultdict
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PropertyGraphIndex
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import TextNode, BaseNode, TransformComponent
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from pathlib import Path
from typing import List, Dict, Union
import networkx as nx

from torchtalk.analysis.binding_detector import BindingDetector, Binding
from torchtalk.analysis.repo_analyzer import RepoAnalyzer

log = logging.getLogger(__name__)


def generate_chunk_context(
    rel_path: str,
    language: str,
    start_line: int,
    end_line: int,
    text: str,
    bindings: List[Dict] = None,
    file_importance: float = 0.0,
) -> str:
    """
    Generate contextual prefix for a code chunk following Anthropic's Contextual Retrieval.

    This prepends semantic context to each chunk BEFORE embedding, helping the embedding
    model understand what the chunk is about even when the code itself is ambiguous.

    For example, a chunk containing "return x" becomes:
    "[FILE: torch/nn/functional.py] [FUNCTION: layer_norm] [LINES: 2914-2920]
    This is the Python API for layer normalization in PyTorch's nn.functional module.
    return x"

    Args:
        rel_path: Relative file path (e.g., "torch/nn/functional.py")
        language: Programming language (python, cpp, cuda, yaml)
        start_line: Starting line number
        end_line: Ending line number
        text: The actual code text
        bindings: Cross-language binding information
        file_importance: PageRank-based importance score

    Returns:
        Context string to prepend to the chunk
    """
    context_parts = []

    # 1. File location context
    context_parts.append(f"[FILE: {rel_path}]")

    # 2. Line range
    if start_line and end_line:
        context_parts.append(f"[LINES: {start_line}-{end_line}]")

    # 3. Language/layer context - helps distinguish Python API from C++ impl from CUDA kernel
    layer_desc = {
        "python": "Python",
        "cpp": "C++",
        "cuda": "CUDA",
        "yaml": "YAML configuration",
    }.get(language, language)
    context_parts.append(f"[LANG: {layer_desc}]")

    # 4. PyTorch-specific path context - helps embeddings understand the role of this file
    path_context = _infer_path_context(rel_path)
    if path_context:
        context_parts.append(f"[COMPONENT: {path_context}]")

    # 5. Cross-language binding context - critical for finding related implementations
    if bindings:
        binding_names = []
        for b in bindings[:3]:  # Limit to avoid overly long context
            cpp_name = b.get("cpp_name", "")
            py_name = b.get("python_name", "")
            if cpp_name and py_name and cpp_name != py_name:
                binding_names.append(f"{py_name}/{cpp_name}")
            elif cpp_name:
                binding_names.append(cpp_name)
            elif py_name:
                binding_names.append(py_name)
        if binding_names:
            context_parts.append(f"[BINDINGS: {', '.join(binding_names)}]")

    # 6. Extract function/class names from code for additional context
    code_symbols = _extract_code_symbols(text, language)
    if code_symbols:
        context_parts.append(f"[DEFINES: {', '.join(code_symbols[:5])}]")

    # 7. High-importance file indicator
    if file_importance > 0.5:
        context_parts.append("[CORE_FILE]")

    return " ".join(context_parts)


def _infer_path_context(rel_path: str) -> str:
    """Infer the role of a file based on its path in the PyTorch codebase."""
    path_lower = rel_path.lower()

    # Python API layers
    if "torch/nn/functional" in path_lower:
        return "PyTorch nn.functional API"
    if "torch/nn/modules" in path_lower:
        return "PyTorch nn.Module classes"
    if "torch/autograd" in path_lower:
        return "PyTorch autograd system"
    if "torch/_refs" in path_lower:
        return "PyTorch reference implementations"
    if "torch/functional" in path_lower:
        return "PyTorch functional API"

    # ATen native implementations
    if "aten/src/aten/native/cuda" in path_lower:
        return "ATen CUDA kernel implementations"
    if "aten/src/aten/native/cpu" in path_lower:
        return "ATen CPU implementations"
    if "aten/src/aten/native" in path_lower:
        return "ATen native operator implementations"
    if "native_functions.yaml" in path_lower:
        return "ATen operator dispatch declarations"
    if "derivatives.yaml" in path_lower:
        return "Autograd derivative formulas"

    # C++ API
    if "torch/csrc/autograd" in path_lower:
        return "C++ autograd engine"
    if "torch/csrc/api" in path_lower:
        return "C++ frontend API"
    if "torch/csrc/jit" in path_lower:
        return "TorchScript JIT compiler"

    # Tools and codegen
    if "tools/autograd" in path_lower:
        return "Autograd code generation"
    if "torchgen" in path_lower:
        return "PyTorch code generation"

    return ""


def _extract_code_symbols(text: str, language: str) -> List[str]:
    """Extract function/class/kernel names defined in this chunk."""
    symbols = []

    if language == "python":
        # Python function and class definitions
        symbols.extend(re.findall(r"^def\s+(\w+)\s*\(", text, re.MULTILINE))
        symbols.extend(re.findall(r"^class\s+(\w+)", text, re.MULTILINE))
        # Also catch decorated functions
        symbols.extend(re.findall(r"@\w+\s*\ndef\s+(\w+)", text))
    elif language in ("cpp", "cuda"):
        # C++ function definitions (simplified)
        symbols.extend(
            re.findall(r"(?:void|int|float|bool|Tensor|auto)\s+(\w+)\s*\(", text)
        )
        # CUDA kernels
        symbols.extend(re.findall(r"__global__\s+void\s+(\w+)", text))
        # Template functions
        symbols.extend(re.findall(r"template.*?\n.*?(\w+)\s*\(", text))
    elif language == "yaml":
        # YAML function entries (native_functions.yaml style)
        symbols.extend(re.findall(r"^- func:\s*(\w+)", text, re.MULTILINE))
        symbols.extend(re.findall(r"^\s+(\w+):", text, re.MULTILINE))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in symbols:
        if s not in seen and len(s) > 2:  # Skip very short names
            seen.add(s)
            unique.append(s)

    return unique


class BindingGraphExtractor(TransformComponent):
    """
    LlamaIndex TransformComponent that creates graph entities and relations
    from cross-language bindings detected by BindingDetector.

    Creates edges for PropertyGraphIndex traversal:
    - PYTHON_API -> CPP_IMPL (pybind11 bindings)
    - ATEN_OP -> CUDA_KERNEL (native_functions.yaml dispatch)
    - ATEN_OP -> CPU_KERNEL
    """

    # Pydantic fields - use Any to avoid complex type validation
    repo_root: Path = None
    py_to_cpp: Dict = {}
    cpp_to_binding: Dict = {}

    def __init__(self, bindings: List[Binding], repo_root: Path, **kwargs):
        # Initialize Pydantic model first
        super().__init__(**kwargs)

        # Now set our fields
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "py_to_cpp", defaultdict(list))
        object.__setattr__(self, "cpp_to_binding", defaultdict(list))

        # Index bindings by python_name and cpp_name for O(1) lookup
        for b in bindings:
            # Normalize names (strip aten:: prefix, lowercase)
            py_name = b.python_name.replace("aten::", "").lower()
            cpp_name = b.cpp_name.lower()
            self.py_to_cpp[py_name].append(b)
            self.py_to_cpp[b.python_name].append(b)  # Also keep original
            self.cpp_to_binding[cpp_name].append(b)
            self.cpp_to_binding[b.cpp_name].append(b)

        log.info(f"BindingGraphExtractor: {len(bindings)} bindings indexed")

    def _extract_symbols(self, text: str, lang: str) -> List[str]:
        """Extract op/function names from code text."""
        symbols = []

        if lang == "python":
            symbols.extend(re.findall(r"def\s+(\w+)\s*\(", text))
            symbols.extend(re.findall(r"torch\.ops\.aten\.(\w+)", text))
            symbols.extend(re.findall(r"aten::(\w+)", text))
            symbols.extend(re.findall(r"F\.(\w+)\s*\(", text))  # F.softmax(
        elif lang in ("cpp", "cuda"):
            symbols.extend(re.findall(r"aten::(\w+)", text))
            symbols.extend(re.findall(r"DEFINE_DISPATCH\((\w+)_stub\)", text))
            symbols.extend(re.findall(r"__global__\s+void\s+(\w+)", text))
            symbols.extend(re.findall(r"(\w+)_kernel\s*\(", text))
            symbols.extend(re.findall(r"(\w+)_cuda\s*\(", text))

        return [s.lower() for s in set(symbols)]

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        """Add graph entities and relations to node metadata."""
        total_relations = 0

        for node in nodes:
            entities: List[EntityNode] = node.metadata.pop(KG_NODES_KEY, [])
            relations: List[Relation] = node.metadata.pop(KG_RELATIONS_KEY, [])

            lang = node.metadata.get("language", "unknown")
            text = node.get_content()
            symbols = self._extract_symbols(text, lang)

            # Find bindings matching symbols in this chunk
            for sym in symbols:
                matched = self.py_to_cpp.get(sym, []) + self.cpp_to_binding.get(sym, [])

                for binding in matched:
                    # Determine labels based on binding type
                    if "cuda" in binding.binding_type:
                        impl_label, rel_type = "CUDA_KERNEL", "DISPATCHES_TO"
                    elif "cpu" in binding.binding_type:
                        impl_label, rel_type = "CPU_KERNEL", "DISPATCHES_TO"
                    else:
                        impl_label, rel_type = "CPP_IMPL", "BINDS_TO"

                    # Create entities
                    py_entity = EntityNode(
                        name=binding.python_name,
                        label="PYTHON_API",
                        properties={"binding_type": binding.binding_type},
                    )
                    cpp_entity = EntityNode(
                        name=binding.cpp_name,
                        label=impl_label,
                        properties={"binding_type": binding.binding_type},
                    )
                    entities.extend([py_entity, cpp_entity])

                    # Create relation
                    relation = Relation(
                        source_id=binding.python_name,
                        target_id=binding.cpp_name,
                        label=rel_type,
                    )
                    relations.append(relation)
                    total_relations += 1

            node.metadata[KG_NODES_KEY] = entities
            node.metadata[KG_RELATIONS_KEY] = relations

        log.info(f"BindingGraphExtractor: created {total_relations} relations")
        return nodes


class GraphEnhancedIndexer:
    """
    Enhances LlamaIndex nodes with graph metadata for cross-language tracing.

    Uses your existing BindingDetector and RepoAnalyzer to inject:
    - Cross-language bindings (Python ↔ C++ ↔ CUDA)
    - Call graph relationships
    - Import dependencies
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo_root = self.repo_path.resolve()
        self.binding_detector = BindingDetector()
        self.repo_analyzer = RepoAnalyzer(str(repo_path))
        self._analysis_ready = False

    def _canon(self, p: str) -> str:
        """Canonicalize path to relative POSIX form.

        Handles both:
        - Absolute paths like /torchtalk/pytorch/torch/nn/functional.py
        - Relative paths like torch/nn/functional.py (from import graph)
        """
        path = Path(p)

        # If path is already relative to repo root, just return normalized form
        if not path.is_absolute():
            # Check if it's a valid path relative to repo root
            if (self.repo_root / path).exists():
                return path.as_posix()
            # Fallback: resolve and try to make relative
            path = path.resolve()

        # For absolute paths, make relative to repo root
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            # Path not under repo root - return as-is
            return path.as_posix()

    def _compute_file_importance(self) -> Dict[str, float]:
        """
        Compute file importance using PageRank on the import graph.

        This is a general, non-hardcoded approach to ranking files:
        - Files imported by many important files get higher scores
        - Based on Sourcegraph's approach to code search ranking

        The import graph has edges A→B where A imports B.
        PageRank gives high scores to nodes with many incoming edges (B is imported by A),
        so files that are imported by many other files get higher scores.

        Returns:
            Dict mapping canonical file paths to importance scores (0.0-1.0)
        """
        import_graph = self.repo_analyzer.import_graph

        if import_graph.number_of_nodes() == 0:
            return {}

        try:
            # Run PageRank with damping factor 0.85 (standard)
            # Files with many incoming edges (imported by many files) get high scores
            pagerank_scores = nx.pagerank(import_graph, alpha=0.85, max_iter=100)
        except Exception as e:
            log.warning(f"PageRank computation failed: {e}")
            return {}

        # Normalize to 0-1 range
        if not pagerank_scores:
            return {}

        max_score = max(pagerank_scores.values())
        min_score = min(pagerank_scores.values())
        score_range = max_score - min_score if max_score > min_score else 1.0

        normalized = {}
        for path, score in pagerank_scores.items():
            canon_path = self._canon(str(path))
            normalized[canon_path] = (score - min_score) / score_range

        # Also compute binding density scores (files with more bindings are more important)
        # This helps rank C++/CUDA implementation files that might not be in import graph
        binding_counts = defaultdict(int)
        for path, bindings in self._bindings_by_file.items():
            binding_counts[path] = len(bindings)

        if binding_counts:
            max_bindings = max(binding_counts.values())
            binding_normalized = {}
            for path, count in binding_counts.items():
                binding_normalized[path] = (
                    count / max_bindings if max_bindings > 0 else 0
                )

            # Combine PageRank and binding density:
            # - Files in both: max(PageRank, binding) to keep high-PageRank files important
            # - Files only in PageRank: keep their score (important Python modules)
            # - Files only in bindings: use binding score (C++/CUDA impl files)
            all_paths = set(normalized.keys()) | set(binding_normalized.keys())
            combined = {}
            for path in all_paths:
                pr_score = normalized.get(path, 0.0)
                bind_score = binding_normalized.get(path, 0.0)
                # Use max() so high PageRank files stay important even without bindings
                # and binding-heavy files get importance even if not in import graph
                combined[path] = max(pr_score, bind_score)
            normalized = combined

        # Log top important files for debugging
        top_files = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:10]
        log.info("Top 10 important files by PageRank+bindings:")
        for path, score in top_files:
            log.info(f"  {score:.3f}: {path[:70]}")

        return normalized

    def build_index(
        self,
        persist_dir: str,
        use_property_graph: bool = False,
        use_lancedb: bool = False,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
    ) -> Union[VectorStoreIndex, PropertyGraphIndex]:
        """
        Build LlamaIndex with graph metadata.

        Args:
            persist_dir: Directory to persist the index
            use_property_graph: If True, use PropertyGraphIndex with graph traversal.
                               If False (default), use VectorStoreIndex with metadata only.
            use_lancedb: If True, use LanceDB for vector storage with native hybrid search.
            neo4j_uri: Neo4j connection URI (e.g., "bolt://localhost:7687"). If provided,
                      stores property graph in Neo4j for advanced graph queries.
            neo4j_user: Neo4j username (default: "neo4j")
            neo4j_password: Neo4j password

        Returns:
            VectorStoreIndex or PropertyGraphIndex depending on use_property_graph
        """
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Run analysis once
        if not self._analysis_ready:
            log.info("Initializing graph analyzers...")
            log.info("Running repository analysis (this may take a few minutes)...")

            try:
                self.repo_analyzer.analyze_repository()
            except Exception as e:
                log.warning(
                    "Repo analysis failed, continuing with partial features: %s", e
                )

            log.info("Detecting cross-language bindings...")
            try:
                self.bindings = self.binding_detector.detect_bindings_in_directory(
                    str(self.repo_path)
                )
            except Exception as e:
                log.warning(
                    "Binding detection failed, continuing without bindings: %s", e
                )
                self.bindings = type("obj", (object,), {"bindings": []})()

            # Pre-index bindings by file with canonical paths
            self._bindings_by_file = {}
            # Also index by cpp_name AND python_name for content-based matching
            # (native_functions.yaml bindings have file_path pointing to YAML, not impl files)
            self._bindings_by_name = {}
            for b in self.bindings.bindings:
                binding_dict = {
                    "python_name": b.python_name,
                    "cpp_name": b.cpp_name,
                    "type": b.binding_type,
                    "line": b.line_number,
                }
                key = self._canon(b.file_path)
                self._bindings_by_file.setdefault(key, []).append(binding_dict)
                # Index by cpp_name (lowercase for case-insensitive matching)
                cpp_key = b.cpp_name.lower()
                self._bindings_by_name.setdefault(cpp_key, []).append(binding_dict)
                # Also index by python_name (for matching torch.ops.aten.xxx calls)
                py_key = b.python_name.lower()
                if py_key != cpp_key:  # Avoid duplicate if names are same
                    self._bindings_by_name.setdefault(py_key, []).append(binding_dict)

            # Pre-index graph nodes with canonical paths
            self._import_nodes_canon = {
                self._canon(str(n)): n for n in self.repo_analyzer.import_graph.nodes()
            }
            self._call_nodes_canon = {
                self._canon(str(n)): n for n in self.repo_analyzer.call_graph.nodes()
            }

            log.info("Analysis complete:")
            log.info(
                f"  - Found {sum(len(v) for v in self._bindings_by_file.values())} cross-language bindings"
            )
            log.info(
                f"  - Built call graph: {self.repo_analyzer.call_graph.number_of_nodes()} nodes"
            )
            log.info(
                f"  - Built import graph: {self.repo_analyzer.import_graph.number_of_nodes()} nodes"
            )

            # Compute file importance using PageRank on the import graph
            # Files that are imported by many other important files get higher scores
            self._file_importance = self._compute_file_importance()
            log.info(
                f"  - Computed importance scores for {len(self._file_importance)} files"
            )

            self._analysis_ready = True

        # Load documents with injected metadata
        log.info("\n" + "=" * 60)
        log.info("Loading documents")
        log.info("=" * 60)

        documents = SimpleDirectoryReader(
            str(self.repo_path),
            required_exts=[".py", ".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp", ".yaml"],
            recursive=True,
            exclude_hidden=True,
            exclude=["test", "tests", "build", "third_party"],
            file_metadata=lambda p: {
                "file_path": Path(p).resolve().as_posix(),
                "rel_path": Path(p).resolve().relative_to(self.repo_root).as_posix(),
            },
        ).load_data()

        log.info(f"Loaded {len(documents)} documents")

        # Parse into nodes with code-aware splitter
        log.info("\n" + "=" * 60)
        log.info("Parsing into nodes (code-aware)")
        log.info("=" * 60)

        from tree_sitter_language_pack import get_parser

        # Map file extensions to languages
        lang_map = {
            ".py": "python",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cu": "cpp",
            ".cuh": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".yaml": "yaml",
            ".yml": "yaml",
        }

        # Group documents by language
        docs_by_lang = {}
        for doc in documents:
            ext = Path(doc.metadata.get("file_path", "")).suffix
            lang = lang_map.get(ext)
            if lang:
                docs_by_lang.setdefault(lang, []).append(doc)

        # Parse each language separately
        # NOTE: Chunk size of 80 lines (~2000 chars) balances:
        # - Enough context to understand a function/class
        # - Small enough to avoid noise in retrieval
        # - Multiple chunks can be retrieved without overwhelming context
        # Research suggests 250-1000 tokens optimal for code RAG
        nodes = []
        for lang, docs in docs_by_lang.items():
            log.info(f"  - Parsing {len(docs)} {lang} files")
            parser = CodeSplitter.from_defaults(
                language=lang,
                chunk_lines=80,  # Reduced from 200 for better precision
                chunk_lines_overlap=15,  # ~20% overlap for context continuity
                parser=get_parser(lang),
            )

            # Parse documents with error handling
            failed = 0
            for doc in docs:
                try:
                    doc_nodes = parser.get_nodes_from_documents([doc])
                    nodes.extend(doc_nodes)
                except (ValueError, Exception):
                    failed += 1
                    continue

            if failed > 0:
                log.info(f"    (skipped {failed} unparseable files)")

        log.info(f"Created {len(nodes)} nodes")

        # Enhance with graph metadata
        log.info("\n" + "=" * 60)
        log.info("Enhancing with graph metadata")
        log.info("=" * 60)

        enhanced_nodes = self._enhance_nodes_with_graphs(nodes)

        log.info(f"Enhanced {len(enhanced_nodes)} nodes")

        # Build index
        log.info("\n" + "=" * 60)
        backend_info = []
        if use_property_graph:
            backend_info.append("PropertyGraphIndex")
        else:
            backend_info.append("VectorStoreIndex")
        if use_lancedb:
            backend_info.append("LanceDB")
        if neo4j_uri:
            backend_info.append("Neo4j")
        log.info(f"Building {' + '.join(backend_info)}")
        log.info("=" * 60)

        # Set up code-specific embedding model with GPU acceleration and large batches
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Settings, StorageContext
        import torch

        # Detect GPU and set optimal batch size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Very small batch size to avoid OOM when sharing GPU with vLLM
        # Jina embeddings with 8192 token context creates large attention matrices
        embed_batch_size = 4 if device == "cuda" else 2

        log.info(f"Embedding device: {device}, batch_size: {embed_batch_size}")

        embed_model = HuggingFaceEmbedding(
            model_name="jinaai/jina-embeddings-v2-base-code",
            trust_remote_code=True,
            device=device,
            embed_batch_size=embed_batch_size,
        )
        Settings.embed_model = embed_model

        # Set up vector store (LanceDB or default)
        vector_store = None
        if use_lancedb:
            from llama_index.vector_stores.lancedb import LanceDBVectorStore

            lancedb_path = persist_dir / "lancedb"
            log.info(f"Using LanceDB vector store at {lancedb_path}")
            vector_store = LanceDBVectorStore(
                uri=str(lancedb_path),
                mode="overwrite",
                query_type="hybrid",  # Enable native hybrid search
            )

        # Set up graph store (Neo4j or default)
        graph_store = None
        if neo4j_uri and use_property_graph:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

            log.info(f"Using Neo4j graph store at {neo4j_uri}")
            graph_store = Neo4jPropertyGraphStore(
                username=neo4j_user or "neo4j",
                password=neo4j_password or "",
                url=neo4j_uri,
            )

        if use_property_graph:
            # Create binding graph extractor for PropertyGraphIndex
            kg_extractor = BindingGraphExtractor(
                bindings=self.bindings.bindings,
                repo_root=self.repo_root,
            )

            # First, build a VectorStoreIndex to embed ALL text nodes
            # PropertyGraphIndex only embeds kg_nodes (graph entities), not source text
            log.info(
                f"Building vector embeddings for {len(enhanced_nodes)} text nodes..."
            )

            if vector_store:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                vector_index = VectorStoreIndex(
                    nodes=enhanced_nodes,
                    embed_model=embed_model,
                    storage_context=storage_context,
                    show_progress=True,
                    use_async=False,  # Disable async to avoid GPU memory fragmentation
                )
            else:
                vector_index = VectorStoreIndex(
                    nodes=enhanced_nodes,
                    embed_model=embed_model,
                    show_progress=True,
                    use_async=False,  # Disable async to avoid GPU memory fragmentation
                )

            # Now build PropertyGraphIndex with the same vector store
            # This gives us graph traversal + vector search over actual code
            # We pass embed_kg_nodes=False since text nodes are already embedded above
            log.info("Building property graph with cross-language bindings...")

            if graph_store:
                # Use Neo4j for graph storage
                index = PropertyGraphIndex(
                    nodes=enhanced_nodes,
                    kg_extractors=[kg_extractor],
                    embed_model=embed_model,
                    property_graph_store=graph_store,
                    vector_store=vector_index.vector_store,
                    embed_kg_nodes=False,
                    show_progress=True,
                )
            else:
                index = PropertyGraphIndex(
                    nodes=enhanced_nodes,
                    kg_extractors=[kg_extractor],
                    embed_model=embed_model,
                    vector_store=vector_index.vector_store,
                    embed_kg_nodes=False,
                    show_progress=True,
                )

            # Copy docstore from vector_index if it has nodes (non-LanceDB case).
            # When using LanceDB, the vector_index docstore is empty because LanceDB
            # handles storage externally. In that case, PropertyGraphIndex already
            # has the nodes in its own docstore from the constructor.
            if vector_index.storage_context.docstore.docs:
                index.storage_context.docstore = vector_index.storage_context.docstore
        else:
            if vector_store:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                index = VectorStoreIndex(
                    nodes=enhanced_nodes,
                    storage_context=storage_context,
                    show_progress=True,
                )
            else:
                index = VectorStoreIndex(nodes=enhanced_nodes, show_progress=True)

        # Persist (LlamaIndex built-in)
        # Note: Neo4j graph store persists to Neo4j directly, not to disk
        log.info(f"\nPersisting to {persist_dir}...")
        index.storage_context.persist(persist_dir=str(persist_dir))

        # Save index configuration for loading
        import json

        config = {
            "index_type": "property_graph" if use_property_graph else "vector",
            "use_lancedb": use_lancedb,
            "neo4j_uri": neo4j_uri,
        }
        config_file = persist_dir / ".index_config.json"
        config_file.write_text(json.dumps(config, indent=2))

        # Legacy marker for backward compatibility
        index_type_file = persist_dir / ".index_type"
        index_type_file.write_text("property_graph" if use_property_graph else "vector")

        log.info("\n" + "=" * 60)
        log.info("Index built successfully!")
        log.info("=" * 60)
        log.info(f"Location: {persist_dir}")
        log.info(
            f"Index type: {'PropertyGraphIndex' if use_property_graph else 'VectorStoreIndex'}"
        )
        if use_lancedb:
            log.info("Vector store: LanceDB (hybrid search enabled)")
        if neo4j_uri:
            log.info(f"Graph store: Neo4j ({neo4j_uri})")
        log.info(f"Documents: {len(documents)}")
        log.info(f"Nodes: {len(enhanced_nodes)}")
        if use_property_graph:
            log.info(f"Bindings indexed: {len(self.bindings.bindings)}")

        return index

    def _compute_line_numbers(
        self, node: TextNode, file_content_cache: Dict[str, str]
    ) -> None:
        """
        Compute start_line and end_line from character indices.

        LlamaIndex's CodeSplitter sets start_char_idx and end_char_idx on nodes.
        We convert these to line numbers for better citations.
        """
        start_idx = node.start_char_idx
        end_idx = node.end_char_idx

        if start_idx is None or end_idx is None:
            return

        # Get file content (cached)
        file_path = node.metadata.get("file_path", "")
        if not file_path:
            return

        if file_path not in file_content_cache:
            try:
                file_content_cache[file_path] = Path(file_path).read_text(
                    encoding="utf-8"
                )
            except Exception:
                return

        content = file_content_cache[file_path]

        # Count newlines up to start and end positions
        start_line = content[:start_idx].count("\n") + 1
        end_line = content[:end_idx].count("\n") + 1

        node.metadata["start_line"] = start_line
        node.metadata["end_line"] = end_line

    def _enhance_nodes_with_graphs(self, nodes: List[TextNode]) -> List[TextNode]:
        """Add graph metadata to each node.

        Note: Complex metadata (lists, dicts) are serialized to JSON strings for
        LanceDB compatibility which requires flat metadata (str, int, float, None).
        The postprocessors in graph_expanded_retriever.py handle deserialization.
        """
        import json

        enhanced_count = 0
        line_number_count = 0

        # Cache file contents for line number computation
        file_content_cache: Dict[str, str] = {}

        for node in nodes:
            # Get canonical relative path
            rel_path = node.metadata.get("rel_path")
            if not rel_path:
                # Fallback to file_path if rel_path not set
                file_path = (
                    node.metadata.get("file_path")
                    or node.metadata.get("file_name")
                    or node.metadata.get("filename")
                    or ""
                )
                if file_path:
                    rel_path = self._canon(file_path)
                else:
                    continue

            # Store canonical path on node
            node.metadata["rel_path"] = rel_path

            # Compute line numbers from character indices (P1: line numbers in index)
            self._compute_line_numbers(node, file_content_cache)
            if "start_line" in node.metadata:
                line_number_count += 1

            # Detect language (include .cuh)
            if rel_path.endswith(".py"):
                lang = "python"
            elif rel_path.endswith((".cpp", ".cc", ".h", ".hpp")):
                lang = "cpp"
            elif rel_path.endswith((".cu", ".cuh")):
                lang = "cuda"
            else:
                lang = "unknown"

            # Set consistent defaults for all nodes
            node.metadata.setdefault("schema", "tt_graph_meta_v1")
            node.metadata.setdefault("language", lang)
            node.metadata.setdefault("has_bindings", False)
            # Complex metadata serialized to JSON strings for LanceDB compatibility
            node.metadata.setdefault("cross_language_bindings", "[]")
            node.metadata.setdefault("function_calls", "[]")
            node.metadata.setdefault("imports", "[]")

            # Exclude verbose/redundant metadata from embedding
            # The contextual_header will contain the important semantic info
            node.excluded_embed_metadata_keys = [
                "file_path",
                "rel_path",
                "schema",
                "has_bindings",
                "cross_language_bindings",
                "function_calls",
                "imports",
                "file_importance",
                "start_line",
                "end_line",
                "language",
            ]
            # Keep all metadata available for LLM context
            node.excluded_llm_metadata_keys = [
                "schema",
                "file_importance",
                "has_bindings",
            ]

            # Add file importance score (PageRank-based)
            # This is a general, non-hardcoded ranking signal
            file_importance = self._file_importance.get(rel_path, 0.0)
            node.metadata["file_importance"] = file_importance

            # Add binding information - first try file-based lookup
            bindings = self._get_bindings_for_file(rel_path)

            # Also check for content-based binding matches (for native_functions.yaml bindings)
            # These bindings have file_path pointing to the YAML, not the implementation
            content_bindings = self._get_bindings_for_content(node.get_content(), lang)
            if content_bindings:
                # Merge, avoiding duplicates
                existing_cpp = {b.get("cpp_name") for b in bindings}
                for cb in content_bindings:
                    if cb.get("cpp_name") not in existing_cpp:
                        bindings.append(cb)

            if bindings:
                # Serialize to JSON for LanceDB compatibility (requires flat metadata)
                node.metadata["cross_language_bindings"] = json.dumps(bindings)
                node.metadata["has_bindings"] = True
                enhanced_count += 1

            # Add call graph information
            calls = self._get_calls_in_file(rel_path)
            if calls:
                # Serialize to JSON for LanceDB compatibility
                node.metadata["function_calls"] = json.dumps(calls)

            # Add import information
            imports = self._get_imports_for_file(rel_path)
            if imports:
                # Serialize to JSON for LanceDB compatibility
                node.metadata["imports"] = json.dumps(imports)

            # Generate contextual header for improved retrieval (Anthropic's Contextual Retrieval)
            # This prepends semantic context to the chunk text before embedding
            context_header = generate_chunk_context(
                rel_path=rel_path,
                language=lang,
                start_line=node.metadata.get("start_line"),
                end_line=node.metadata.get("end_line"),
                text=node.get_content(),
                bindings=bindings if bindings else None,
                file_importance=file_importance,
            )
            # Store as metadata - LlamaIndex will prepend this during embedding
            node.metadata["contextual_header"] = context_header

        log.info(f"  - Enhanced {enhanced_count} nodes with cross-language bindings")
        log.info(f"  - Added line numbers to {line_number_count} nodes")
        log.info(f"  - Added contextual headers to {len(nodes)} nodes")

        return nodes

    def _get_bindings_for_file(self, rel_path: str) -> List[Dict]:
        """Get cross-language bindings for a file (O(1) lookup)"""
        return self._bindings_by_file.get(rel_path, [])

    def _get_bindings_for_content(self, text: str, lang: str) -> List[Dict]:
        """
        Find bindings by matching dispatch function names in the node's content.

        This handles native_functions.yaml bindings which have file_path pointing
        to the YAML file, not the actual implementation files.
        """
        if not hasattr(self, "_bindings_by_name") or not self._bindings_by_name:
            return []

        matched_bindings = []
        seen_cpp_names = set()

        # Extract potential dispatch function names from the code
        symbols = []
        if lang == "python":
            # Python: torch.ops.aten.xxx, aten::xxx, F.xxx
            symbols.extend(re.findall(r"torch\.ops\.aten\.(\w+)", text))
            symbols.extend(re.findall(r"aten::(\w+)", text))
            symbols.extend(re.findall(r"F\.(\w+)\s*\(", text))
            symbols.extend(re.findall(r"def\s+(\w+)\s*\(", text))
        elif lang in ("cpp", "cuda"):
            # C++/CUDA: function definitions, DEFINE_DISPATCH, kernels
            symbols.extend(re.findall(r"aten::(\w+)", text))
            symbols.extend(re.findall(r"DEFINE_DISPATCH\((\w+)_stub\)", text))
            # Match function definitions: type funcname( - captures full name
            symbols.extend(
                re.findall(r"(?:^|\s)(\w+)\s*\([^)]*\)\s*\{", text, re.MULTILINE)
            )
            # Match xxx_kernel patterns (for kernel templates)
            symbols.extend(re.findall(r"(\w+_kernel)\s*[<\(]", text))
            # Match xxx_cuda and xxx_cpu function names (full name including suffix)
            symbols.extend(re.findall(r"\b(\w+_cuda)\s*\(", text))
            symbols.extend(re.findall(r"\b(\w+_cpu)\s*\(", text))
            # Match native_xxx patterns
            symbols.extend(re.findall(r"\b(native_\w+)\b", text))

        # Look up each symbol in the bindings index (indexed by both cpp_name and python_name)
        for sym in symbols:
            sym_lower = sym.lower()
            if sym_lower in self._bindings_by_name:
                for binding in self._bindings_by_name[sym_lower]:
                    cpp_name = binding.get("cpp_name", "")
                    if cpp_name not in seen_cpp_names:
                        matched_bindings.append(binding)
                        seen_cpp_names.add(cpp_name)

        return matched_bindings

    def _get_calls_in_file(self, file_path: str) -> List[Dict]:
        """Get function calls in this file"""
        fp = self._canon(file_path)

        # Use pre-indexed canonical nodes
        node = self._call_nodes_canon.get(fp)
        if node is not None:
            calls = []
            successors = list(self.repo_analyzer.call_graph.successors(node))
            if successors:
                calls.append(
                    {"function": str(node), "calls": [str(s) for s in successors[:10]]}
                )
            return calls[:10]

        return []

    def _get_imports_for_file(self, file_path: str) -> List[str]:
        """Get imports for this file"""
        fp = self._canon(file_path)

        # Use pre-indexed canonical nodes
        node = self._import_nodes_canon.get(fp)
        if node is not None:
            imports = list(self.repo_analyzer.import_graph.successors(node))
            return [self._canon(str(i)) for i in imports[:20]]

        return []
