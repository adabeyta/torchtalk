import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import TextNode
from pathlib import Path
from typing import List, Dict

from torchtalk.analysis.binding_detector import BindingDetector
from torchtalk.analysis.repo_analyzer import RepoAnalyzer

log = logging.getLogger(__name__)


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

    def build_index(self, persist_dir: str) -> VectorStoreIndex:
        """Build LlamaIndex with graph metadata"""

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Run analysis once
        if not self._analysis_ready:
            log.info("Initializing graph analyzers...")
            log.info("Running repository analysis (this may take a few minutes)...")

            try:
                self.analysis = self.repo_analyzer.analyze_repository()
            except Exception as e:
                log.warning("Repo analysis failed, continuing with partial features: %s", e)
                self.analysis = {}

            log.info("Detecting cross-language bindings...")
            try:
                self.bindings = self.binding_detector.detect_bindings_in_directory(str(self.repo_path))
            except Exception as e:
                log.warning("Binding detection failed, continuing without bindings: %s", e)
                self.bindings = type('obj', (object,), {'bindings': []})()

            # Pre-index bindings by file with canonical paths
            self._bindings_by_file = {}
            for b in self.bindings.bindings:
                key = self._canon(b.file_path)
                self._bindings_by_file.setdefault(key, []).append({
                    "python_name": b.python_name,
                    "cpp_name": b.cpp_name,
                    "type": b.binding_type,
                    "line": b.line_number,
                })

            # Pre-index graph nodes with canonical paths
            self._import_nodes_canon = {
                self._canon(str(n)): n
                for n in self.repo_analyzer.import_graph.nodes()
            }
            self._call_nodes_canon = {
                self._canon(str(n)): n
                for n in self.repo_analyzer.call_graph.nodes()
            }

            log.info("✓ Analysis complete:")
            log.info(f"  - Found {sum(len(v) for v in self._bindings_by_file.values())} cross-language bindings")
            log.info(f"  - Built call graph: {self.repo_analyzer.call_graph.number_of_nodes()} nodes")
            log.info(f"  - Built import graph: {self.repo_analyzer.import_graph.number_of_nodes()} nodes")

            self._analysis_ready = True

        # Load documents with injected metadata
        log.info("\n" + "="*60)
        log.info("PHASE 1: Loading documents")
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

        log.info(f"✓ Loaded {len(documents)} documents")

        # Parse into nodes with code-aware splitter
        log.info("\n" + "="*60)
        log.info("PHASE 2: Parsing into nodes (code-aware)")
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
                except (ValueError, Exception) as e:
                    failed += 1
                    # Skip unparseable files silently
                    continue

            if failed > 0:
                log.info(f"    (skipped {failed} unparseable files)")

        log.info(f"✓ Created {len(nodes)} nodes")

        # Enhance with graph metadata
        log.info("\n" + "="*60)
        log.info("PHASE 3: Enhancing with graph metadata")
        log.info("="*60)

        enhanced_nodes = self._enhance_nodes_with_graphs(nodes)

        log.info(f"✓ Enhanced {len(enhanced_nodes)} nodes")

        # Build index (LlamaIndex built-in)
        log.info("\n" + "="*60)
        log.info("PHASE 4: Building vector index")
        log.info("="*60)

        # Set up HuggingFace embedding model
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Settings

        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        Settings.embed_model = embed_model

        index = VectorStoreIndex(
            nodes=enhanced_nodes,
            show_progress=True
        )

        # Persist (LlamaIndex built-in)
        log.info(f"\nPersisting to {persist_dir}...")
        index.storage_context.persist(persist_dir=str(persist_dir))

        log.info("\n" + "="*60)
        log.info("✓ Index built successfully!")
        log.info("="*60)
        log.info(f"Location: {persist_dir}")
        log.info(f"Documents: {len(documents)}")
        log.info(f"Nodes: {len(enhanced_nodes)}")

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

            # Add binding information
            bindings = self._get_bindings_for_file(rel_path)
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
