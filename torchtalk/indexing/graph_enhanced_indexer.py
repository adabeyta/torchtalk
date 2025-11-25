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

from torchtalk.analysis.binding_detector import BindingDetector, Binding
from torchtalk.analysis.repo_analyzer import RepoAnalyzer

log = logging.getLogger(__name__)


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
        object.__setattr__(self, 'repo_root', repo_root)
        object.__setattr__(self, 'py_to_cpp', defaultdict(list))
        object.__setattr__(self, 'cpp_to_binding', defaultdict(list))

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
            symbols.extend(re.findall(r'def\s+(\w+)\s*\(', text))
            symbols.extend(re.findall(r'torch\.ops\.aten\.(\w+)', text))
            symbols.extend(re.findall(r'aten::(\w+)', text))
            symbols.extend(re.findall(r'F\.(\w+)\s*\(', text))  # F.softmax(
        elif lang in ("cpp", "cuda"):
            symbols.extend(re.findall(r'aten::(\w+)', text))
            symbols.extend(re.findall(r'DEFINE_DISPATCH\((\w+)_stub\)', text))
            symbols.extend(re.findall(r'__global__\s+void\s+(\w+)', text))
            symbols.extend(re.findall(r'(\w+)_kernel\s*\(', text))
            symbols.extend(re.findall(r'(\w+)_cuda\s*\(', text))

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
        """Canonicalize path to relative POSIX form"""
        try:
            return Path(p).resolve().relative_to(self.repo_root).as_posix()
        except Exception:
            return Path(p).resolve().as_posix()

    def build_index(
        self,
        persist_dir: str,
        use_property_graph: bool = False,
    ) -> Union[VectorStoreIndex, PropertyGraphIndex]:
        """
        Build LlamaIndex with graph metadata.

        Args:
            persist_dir: Directory to persist the index
            use_property_graph: If True, use PropertyGraphIndex with graph traversal.
                               If False (default), use VectorStoreIndex with metadata only.

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
                log.warning("Repo analysis failed, continuing with partial features: %s", e)

            log.info("Detecting cross-language bindings...")
            try:
                self.bindings = self.binding_detector.detect_bindings_in_directory(str(self.repo_path))
            except Exception as e:
                log.warning("Binding detection failed, continuing without bindings: %s", e)
                self.bindings = type('obj', (object,), {'bindings': []})()

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
                self._canon(str(n)): n
                for n in self.repo_analyzer.import_graph.nodes()
            }
            self._call_nodes_canon = {
                self._canon(str(n)): n
                for n in self.repo_analyzer.call_graph.nodes()
            }

            log.info("Analysis complete:")
            log.info(f"  - Found {sum(len(v) for v in self._bindings_by_file.values())} cross-language bindings")
            log.info(f"  - Built call graph: {self.repo_analyzer.call_graph.number_of_nodes()} nodes")
            log.info(f"  - Built import graph: {self.repo_analyzer.import_graph.number_of_nodes()} nodes")

            self._analysis_ready = True

        # Load documents with injected metadata
        log.info("\n" + "="*60)
        log.info("Loading documents")
        log.info("="*60)

        documents = SimpleDirectoryReader(
            str(self.repo_path),
            required_exts=[".py", ".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp"],
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
        log.info("\n" + "="*60)
        log.info("Parsing into nodes (code-aware)")
        log.info("="*60)

        from tree_sitter_language_pack import get_parser

        # Map file extensions to languages
        lang_map = {
            '.py': 'python',
            '.cpp': 'cpp', '.cc': 'cpp', '.cu': 'cpp',
            '.cuh': 'cpp', '.h': 'cpp', '.hpp': 'cpp'
        }

        # Group documents by language
        docs_by_lang = {}
        for doc in documents:
            ext = Path(doc.metadata.get('file_path', '')).suffix
            lang = lang_map.get(ext)
            if lang:
                docs_by_lang.setdefault(lang, []).append(doc)

        # Parse each language separately
        nodes = []
        for lang, docs in docs_by_lang.items():
            log.info(f"  - Parsing {len(docs)} {lang} files")
            parser = CodeSplitter.from_defaults(
                language=lang,
                chunk_lines=200,
                chunk_lines_overlap=20,
                parser=get_parser(lang)
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
        log.info("\n" + "="*60)
        log.info("Enhancing with graph metadata")
        log.info("="*60)

        enhanced_nodes = self._enhance_nodes_with_graphs(nodes)

        log.info(f"Enhanced {len(enhanced_nodes)} nodes")

        # Build index
        log.info("\n" + "="*60)
        if use_property_graph:
            log.info("Building PropertyGraphIndex (with graph traversal)")
        else:
            log.info("Building VectorStoreIndex")
        log.info("="*60)

        # Set up code-specific embedding model with GPU acceleration and large batches
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Settings
        import torch

        # Detect GPU and set optimal batch size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Very small batch size to avoid OOM when sharing GPU with vLLM
        # Jina embeddings with 8192 token context creates large attention matrices
        embed_batch_size = 16 if device == "cuda" else 8

        log.info(f"Embedding device: {device}, batch_size: {embed_batch_size}")

        embed_model = HuggingFaceEmbedding(
            model_name="jinaai/jina-embeddings-v2-base-code",
            trust_remote_code=True,
            device=device,
            embed_batch_size=embed_batch_size,
        )
        Settings.embed_model = embed_model

        if use_property_graph:
            # Create binding graph extractor for PropertyGraphIndex
            kg_extractor = BindingGraphExtractor(
                bindings=self.bindings.bindings,
                repo_root=self.repo_root,
            )

            # First, build a VectorStoreIndex to embed ALL text nodes
            # PropertyGraphIndex only embeds kg_nodes (graph entities), not source text
            log.info(f"Building vector embeddings for {len(enhanced_nodes)} text nodes...")
            vector_index = VectorStoreIndex(
                nodes=enhanced_nodes,
                embed_model=embed_model,
                show_progress=True,
                use_async=True,  # Async embedding for better throughput
            )

            # Now build PropertyGraphIndex with the same vector store
            # This gives us graph traversal + vector search over actual code
            # We pass embed_kg_nodes=False since text nodes are already embedded above
            log.info("Building property graph with cross-language bindings...")
            index = PropertyGraphIndex(
                nodes=enhanced_nodes,
                kg_extractors=[kg_extractor],
                embed_model=embed_model,
                vector_store=vector_index.vector_store,
                embed_kg_nodes=False,  # Skip re-embedding, text nodes already done
                show_progress=True,
            )

            # Use the docstore from vector_index which has all nodes
            index.storage_context.docstore = vector_index.storage_context.docstore
        else:
            index = VectorStoreIndex(
                nodes=enhanced_nodes,
                show_progress=True
            )

        # Persist (LlamaIndex built-in)
        log.info(f"\nPersisting to {persist_dir}...")
        index.storage_context.persist(persist_dir=str(persist_dir))

        # Save index type marker for loading
        index_type_file = persist_dir / ".index_type"
        index_type_file.write_text("property_graph" if use_property_graph else "vector")

        log.info("\n" + "="*60)
        log.info("Index built successfully!")
        log.info("="*60)
        log.info(f"Location: {persist_dir}")
        log.info(f"Index type: {'PropertyGraphIndex' if use_property_graph else 'VectorStoreIndex'}")
        log.info(f"Documents: {len(documents)}")
        log.info(f"Nodes: {len(enhanced_nodes)}")
        if use_property_graph:
            log.info(f"Bindings indexed: {len(self.bindings.bindings)}")

        return index

    def _enhance_nodes_with_graphs(self, nodes: List[TextNode]) -> List[TextNode]:
        """Add graph metadata to each node"""

        enhanced_count = 0

        for node in nodes:
            # Get canonical relative path
            rel_path = node.metadata.get("rel_path")
            if not rel_path:
                # Fallback to file_path if rel_path not set
                file_path = (node.metadata.get("file_path") or
                            node.metadata.get("file_name") or
                            node.metadata.get("filename") or "")
                if file_path:
                    rel_path = self._canon(file_path)
                else:
                    continue

            # Store canonical path on node
            node.metadata["rel_path"] = rel_path

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
            node.metadata.setdefault("cross_language_bindings", [])
            node.metadata.setdefault("function_calls", [])
            node.metadata.setdefault("imports", [])

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
                node.metadata["cross_language_bindings"] = bindings
                node.metadata["has_bindings"] = True
                enhanced_count += 1

            # Add call graph information
            calls = self._get_calls_in_file(rel_path)
            if calls:
                node.metadata["function_calls"] = calls

            # Add import information
            imports = self._get_imports_for_file(rel_path)
            if imports:
                node.metadata["imports"] = imports

        log.info(f"  - Enhanced {enhanced_count} nodes with cross-language bindings")

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
        if not hasattr(self, '_bindings_by_name') or not self._bindings_by_name:
            return []

        matched_bindings = []
        seen_cpp_names = set()

        # Extract potential dispatch function names from the code
        symbols = []
        if lang == "python":
            # Python: torch.ops.aten.xxx, aten::xxx, F.xxx
            symbols.extend(re.findall(r'torch\.ops\.aten\.(\w+)', text))
            symbols.extend(re.findall(r'aten::(\w+)', text))
            symbols.extend(re.findall(r'F\.(\w+)\s*\(', text))
            symbols.extend(re.findall(r'def\s+(\w+)\s*\(', text))
        elif lang in ("cpp", "cuda"):
            # C++/CUDA: function definitions, DEFINE_DISPATCH, kernels
            symbols.extend(re.findall(r'aten::(\w+)', text))
            symbols.extend(re.findall(r'DEFINE_DISPATCH\((\w+)_stub\)', text))
            # Match function definitions: type funcname( - captures full name
            symbols.extend(re.findall(r'(?:^|\s)(\w+)\s*\([^)]*\)\s*\{', text, re.MULTILINE))
            # Match xxx_kernel patterns (for kernel templates)
            symbols.extend(re.findall(r'(\w+_kernel)\s*[<\(]', text))
            # Match xxx_cuda and xxx_cpu function names (full name including suffix)
            symbols.extend(re.findall(r'\b(\w+_cuda)\s*\(', text))
            symbols.extend(re.findall(r'\b(\w+_cpu)\s*\(', text))
            # Match native_xxx patterns
            symbols.extend(re.findall(r'\b(native_\w+)\b', text))

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
                calls.append({
                    "function": str(node),
                    "calls": [str(s) for s in successors[:10]]
                })
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
