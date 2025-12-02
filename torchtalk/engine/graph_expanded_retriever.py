"""
Graph-expanded retriever for cross-language PyTorch code search.

Architecture:
    Query → [BM25 + Vector (RRF)] → Graph Expansion → Rerank → Format

The key insight: PyTorch code spans multiple layers:
    Python API (torch.nn.functional.dropout)
        ↓ BINDS_TO
    C++ Dispatch (native_functions.yaml → native_dropout)
        ↓ DISPATCHES_TO
    CUDA Kernel (aten/src/ATen/native/cuda/Dropout.cu)

Instead of forcing arbitrary file type slots (3 YAML, 5 CUDA, etc.),
we follow the ACTUAL binding relationships stored in chunk metadata.
If a chunk has bindings, we expand to find connected chunks.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Union

from llama_index.core import PropertyGraphIndex, QueryBundle, Settings, VectorStoreIndex
from llama_index.core.postprocessor import LongContextReorder, SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

log = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def _get_bindings_metadata(metadata: dict) -> List[dict]:
    """
    Get cross_language_bindings from metadata, handling both JSON strings and lists.

    LanceDB requires flat metadata (str, int, float, None), so bindings are stored
    as JSON strings during indexing. This function deserializes them.
    """
    bindings = metadata.get("cross_language_bindings", [])
    if isinstance(bindings, str):
        try:
            bindings = json.loads(bindings) if bindings else []
        except json.JSONDecodeError:
            bindings = []
    return bindings if isinstance(bindings, list) else []


def _normalize_binding_name(name: str) -> str:
    """Normalize a binding name for matching (lowercase, strip prefixes)."""
    if not name:
        return ""
    name = name.lower().replace("aten::", "").replace("torch.", "")
    return name


def _extract_base_op_name(name: str) -> Optional[str]:
    """
    Extract the base operation name from variants like native_dropout_cuda.

    Examples:
        native_dropout → dropout
        native_dropout_cuda → dropout
        layer_norm_backward → layer_norm
        cudnn_batch_norm → batch_norm
    """
    if not name or len(name) < 2:
        return None

    original = name

    # Strip common prefixes
    for prefix in ["native_", "fused_", "cudnn_", "mkldnn_", "at_", "_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Strip common suffixes (may need multiple passes)
    suffixes = [
        "_backward",
        "_forward",
        "_cuda",
        "_cpu",
        "_mps",
        "_nested",
        "_sparse",
        "_mkldnn",
        "_impl",
        "_kernel",
        "_out",
        "_stub",
    ]
    for _ in range(2):
        for suffix in suffixes:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                break

    return name if name and len(name) >= 2 and name != original else None


def _expand_pytorch_query(query: str) -> str:
    """
    Expand a query with PyTorch-specific terms to improve retrieval.

    This is a GENERAL approach that:
    1. Detects potential operation names (snake_case words like layer_norm, dropout)
    2. Adds the native_ prefix variant to capture C++ dispatch implementations
    3. Adds "dispatch" keyword to capture native_functions.yaml entries
    4. Adds autograd/derivatives context for gradient-related queries

    This avoids hardcoding specific operation names while still improving
    recall for PyTorch's dispatch-based architecture.

    Examples:
        "layer_norm" → adds "native_layer_norm dispatch"
        "dropout" → adds "native_dropout dispatch"
        "my_custom_op" → adds "native_my_custom_op dispatch"
        "autograd" → adds "derivatives.yaml backward"
    """
    import re

    query_lower = query.lower()
    additions = []

    # General pattern: find snake_case words that could be operation names
    # These are typically 2+ characters, may contain underscores, and aren't common English words
    common_english = {
        "the",
        "and",
        "for",
        "how",
        "does",
        "what",
        "where",
        "when",
        "why",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "with",
        "from",
        "into",
        "about",
        "which",
        "their",
        "there",
        "these",
        "those",
        "then",
        "than",
        "some",
        "such",
        "other",
        "each",
        "every",
        "pytorch",
        "torch",
        "python",
        "cuda",
        "kernel",
        "function",
        "implement",
        "implementation",
        "work",
        "works",
        "call",
        "called",
        "calling",
        "code",
    }

    # Find potential operation names (snake_case or single words that look like ops)
    potential_ops = re.findall(r"\b([a-z][a-z0-9_]+)\b", query_lower)

    for op in potential_ops:
        # Skip common English words
        if op in common_english:
            continue
        # Skip very short words (likely not op names)
        if len(op) < 3:
            continue
        # Skip words that already start with native_
        if op.startswith("native_"):
            continue

        # Add the native_ prefixed variant to capture dispatch implementations
        additions.append(f"native_{op}")

    # If we found any potential ops, add dispatch to improve yaml matching
    if additions:
        additions.append("dispatch")

    # If talking about implementation/internals, add key context terms
    if any(
        word in query_lower
        for word in [
            "implement",
            "cuda",
            "kernel",
            "dispatch",
            "native",
            "backward",
            "forward",
        ]
    ):
        additions.append("native_functions.yaml aten")

    # Autograd-specific expansion: add derivatives.yaml for gradient queries
    if any(
        word in query_lower
        for word in ["autograd", "gradient", "backward", "derivative", "grad"]
    ):
        additions.append("derivatives.yaml")
        additions.append("backward")
        additions.append("torch/autograd")

    # CUDA-specific: ensure we find actual CUDA kernels, not wrappers
    if "cuda" in query_lower:
        additions.append("aten/src/ATen/native/cuda")
        additions.append(".cu")

    if additions:
        # Deduplicate while preserving order
        seen = set()
        unique_additions = []
        for a in additions:
            if a not in seen:
                seen.add(a)
                unique_additions.append(a)

        expanded = query + " " + " ".join(unique_additions)
        log.info(
            f"[QueryExpand] '{query[:50]}...' → added {len(unique_additions)} expansions"
        )
        return expanded

    return query


def _get_layer_type(rel_path: str) -> str:
    """
    Determine which layer of the PyTorch stack a file belongs to.

    Returns: 'python', 'yaml', 'cpp', 'cuda', or 'other'
    """
    if not rel_path:
        return "other"

    # YAML config files (dispatch and derivatives)
    if "native_functions.yaml" in rel_path or "derivatives.yaml" in rel_path:
        return "yaml"
    elif rel_path.endswith(".cu") or "/cuda/" in rel_path:
        return "cuda"
    elif rel_path.endswith(".cpp") and "aten/src/ATen/native" in rel_path:
        return "cpp"
    elif rel_path.endswith(".py"):
        return "python"
    else:
        return "other"


# =============================================================================
# Binding Index - Maps operation names to chunk IDs
# =============================================================================


class BindingIndex:
    """
    Index that maps binding names to chunk node IDs.

    This enables O(1) lookup: given an operation name like "dropout",
    find all chunks that implement any part of the binding chain.
    """

    def __init__(self, docstore):
        self.docstore = docstore
        self._name_to_node_ids: Dict[str, Set[str]] = {}
        self._node_id_to_bindings: Dict[str, List[dict]] = {}
        self._indexed = False

    def _build_index(self):
        """Build the binding index from docstore."""
        if self._indexed or self.docstore is None:
            return

        try:
            all_docs = self.docstore.docs
            for doc_id, doc in all_docs.items():
                bindings = _get_bindings_metadata(doc.metadata)
                if not bindings:
                    continue

                self._node_id_to_bindings[doc_id] = bindings

                for b in bindings:
                    # Index by python_name, cpp_name, and base names
                    py_name = _normalize_binding_name(b.get("python_name", ""))
                    cpp_name = _normalize_binding_name(b.get("cpp_name", ""))

                    for name in [py_name, cpp_name]:
                        if name:
                            if name not in self._name_to_node_ids:
                                self._name_to_node_ids[name] = set()
                            self._name_to_node_ids[name].add(doc_id)

                            # Also index base name (e.g., "dropout" from "native_dropout_cuda")
                            base_name = _extract_base_op_name(name)
                            if base_name:
                                if base_name not in self._name_to_node_ids:
                                    self._name_to_node_ids[base_name] = set()
                                self._name_to_node_ids[base_name].add(doc_id)

            self._indexed = True
            log.info(
                f"[BindingIndex] Indexed {len(self._name_to_node_ids)} binding names "
                f"across {len(self._node_id_to_bindings)} chunks"
            )

        except Exception as e:
            log.warning(f"[BindingIndex] Failed to build index: {e}")

    def get_connected_node_ids(self, node_id: str) -> Set[str]:
        """
        Get all node IDs connected to this node via binding relationships.

        This is the core graph expansion: given a chunk, find all other chunks
        that share the same binding names.
        """
        self._build_index()

        if node_id not in self._node_id_to_bindings:
            return set()

        connected = set()
        bindings = self._node_id_to_bindings[node_id]

        for b in bindings:
            py_name = _normalize_binding_name(b.get("python_name", ""))
            cpp_name = _normalize_binding_name(b.get("cpp_name", ""))

            for name in [py_name, cpp_name]:
                if name and name in self._name_to_node_ids:
                    connected.update(self._name_to_node_ids[name])

                # Also check base name
                base_name = _extract_base_op_name(name)
                if base_name and base_name in self._name_to_node_ids:
                    connected.update(self._name_to_node_ids[base_name])

        # Don't include the original node
        connected.discard(node_id)
        return connected

    def lookup_by_name(self, op_name: str) -> Set[str]:
        """Look up all node IDs that have bindings matching the given operation name."""
        self._build_index()

        op_lower = op_name.lower()
        results = set()

        # Direct match
        if op_lower in self._name_to_node_ids:
            results.update(self._name_to_node_ids[op_lower])

        # Try base name extraction
        base_name = _extract_base_op_name(op_lower)
        if base_name and base_name in self._name_to_node_ids:
            results.update(self._name_to_node_ids[base_name])

        # Partial match fallback
        if not results:
            for indexed_name, node_ids in self._name_to_node_ids.items():
                if op_lower in indexed_name or indexed_name in op_lower:
                    results.update(node_ids)

        return results


# =============================================================================
# Simple Vector Retriever (for PropertyGraphIndex)
# =============================================================================


class _SimpleVectorRetriever(BaseRetriever):
    """
    Vector retriever that queries a SimpleVectorStore directly.

    Used for PropertyGraphIndex where text node embeddings are stored
    in the vector store but SimpleVectorStore doesn't store text itself.
    """

    def __init__(self, vector_store, docstore, embed_model, similarity_top_k: int = 10):
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        from llama_index.core.vector_stores import VectorStoreQuery

        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_bundle.query_str,
            similarity_top_k=self._similarity_top_k,
        )
        result = self._vector_store.query(query)

        nodes_with_scores = []
        if result.ids and result.similarities:
            for node_id, similarity in zip(result.ids, result.similarities):
                node = self._docstore.get_node(node_id)
                if node:
                    nodes_with_scores.append(
                        NodeWithScore(node=node, score=float(similarity))
                    )

        return nodes_with_scores


# =============================================================================
# File Importance Postprocessor - Uses computed PageRank scores
# =============================================================================


class FileImportancePostprocessor(BaseNodePostprocessor):
    """
    Boost files based on computed importance scores (PageRank + binding density).

    This is a GENERAL approach based on research from Sourcegraph and Deprank:
    - Files imported by many important files get higher scores (PageRank)
    - Files with cross-language bindings are more central to the codebase
    - No hardcoded path patterns - importance is computed from the graph

    The file_importance score is computed during indexing and stored in metadata.
    """

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        import re

        for node in nodes:
            original_score = node.score or 0.5

            # Get pre-computed file importance from metadata (0.0-1.0 range)
            file_importance = node.node.metadata.get("file_importance", 0.0)

            # Convert importance to a multiplier (0.5 to 2.5 range)
            # Low importance (0.0) → 0.5x multiplier
            # Mid importance (0.5) → 1.5x multiplier
            # High importance (1.0) → 2.5x multiplier
            importance_multiplier = 0.5 + 2.0 * file_importance

            # Boost files with cross-language bindings (they connect layers)
            bindings = _get_bindings_metadata(node.node.metadata)
            if bindings:
                importance_multiplier *= 1.3

            # Apply general penalty for test/experimental files
            # This is a minimal, general pattern that applies to any codebase
            path = node.node.metadata.get("rel_path", "")
            if re.search(r"(^|/)test[s_]?/", path) or path.endswith("_test.py"):
                importance_multiplier *= 0.3
            elif re.search(r"/experimental/", path):
                importance_multiplier *= 0.2

            # Apply multiplier
            node.score = original_score * importance_multiplier

        # Re-sort by adjusted score
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        return nodes


class PathBoostPostprocessor(BaseNodePostprocessor):
    """
    Fallback boost for files when PageRank scores aren't available.

    Used when the index was built without file_importance metadata.
    This is the legacy hardcoded approach - prefer FileImportancePostprocessor.
    """

    def _get_boost_patterns(self):
        """Multiplicative boost factors for implementation files."""
        return [
            # ATen native implementations - the actual kernels (highest value)
            (r"aten/src/ATen/native/cuda/[^/]+\.cu$", 3.0),  # CUDA kernels
            (r"aten/src/ATen/native/cpu/[^/]+\.cpp$", 2.8),  # CPU kernels
            (r"aten/src/ATen/native/[^/]+\.cpp$", 2.5),  # Native C++ impl
            (r"native_functions\.yaml$", 2.8),  # Dispatch definitions
            # Core Python API - these are what users call
            (r"torch/nn/functional\.py$", 2.5),  # Primary user API
            (r"torch/nn/modules/[^/]+\.py$", 2.0),
            (r"torch/autograd/[^/]+\.py$", 1.8),
            (r"torch/_refs/", 1.5),  # Reference implementations
            # Autograd internals
            (r"torch/csrc/autograd/", 1.8),
            (r"tools/autograd/derivatives\.yaml$", 2.0),
            (r"aten/src/ATen/functorch/", 1.6),  # Functorch internals
            # Core torch modules
            (r"torch/[^/]+\.py$", 1.3),  # Top-level torch/*.py
        ]

    def _get_penalty_patterns(self):
        """Multiplicative penalty factors for non-implementation files."""
        return [
            # Test and experimental code (strong penalty)
            (r"tools/experimental/", 0.2),
            (r"/test/", 0.25),
            (r"test_[^/]+\.py$", 0.3),
            (r"_test\.py$", 0.3),
            # Testing utilities
            (r"torch/testing/", 0.4),
            (r"common_methods_invocations\.py$", 0.35),
            # ONNX export (not core implementation)
            (r"torch/onnx/", 0.4),
            # Decompositions (derived implementations, not core)
            (r"torch/_decomp/", 0.5),
            # API wrappers and headers (not implementations)
            (r"torch/csrc/api/include/", 0.5),
            (r"torch/csrc/api/src/", 0.6),
            # Specialized backends (usually not what user wants)
            (r"/vulkan/", 0.4),
            (r"/metal/", 0.4),
            (r"/mps/", 0.5),
            # Quantization/distributed (specialized)
            (r"torch/ao/quantization/", 0.5),
            (r"torch/distributed/", 0.6),
            # Build/codegen tools (usually not what users want)
            (r"torchgen/", 0.4),
            (r"tools/codegen/", 0.5),
            (r"tools/autograd/gen", 0.5),  # Codegen scripts
        ]

    def _get_binding_boost(self):
        """Extra boost for files with cross-language bindings."""
        return 1.3

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        import re

        boost_patterns = self._get_boost_patterns()
        penalty_patterns = self._get_penalty_patterns()
        binding_boost = self._get_binding_boost()

        for node in nodes:
            path = node.node.metadata.get("rel_path", "")
            original_score = node.score or 0.5
            multiplier = 1.0

            # Apply boost patterns (first match wins)
            for pattern, boost in boost_patterns:
                if re.search(pattern, path):
                    multiplier *= boost
                    break

            # Apply penalty patterns (first match wins)
            for pattern, penalty in penalty_patterns:
                if re.search(pattern, path):
                    multiplier *= penalty
                    break

            # Boost files with cross-language bindings
            bindings = _get_bindings_metadata(node.node.metadata)
            if bindings:
                multiplier *= binding_boost

            # Apply multiplier
            node.score = original_score * multiplier

        # Re-sort by adjusted score (critical for reranker input quality)
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        return nodes


# =============================================================================
# Context Formatter - Adds metadata headers for LLM
# =============================================================================


class ContextFormatterPostprocessor(BaseNodePostprocessor):
    """
    Format nodes with file paths, line numbers, and relationship context.

    This is CRITICAL for response quality - the LLM needs to see:
    - Which file each chunk came from
    - Line numbers for citations
    - The cross-language binding chain (Python → C++ → CUDA)
    """

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for node in nodes:
            md = node.node.metadata or {}

            if md.get("_context_formatted"):
                continue

            header_lines = []

            # File path and line numbers
            file_path = md.get("rel_path") or md.get("file_path") or "unknown"
            start_line = md.get("start_line", "")
            end_line = md.get("end_line", "")
            if start_line and end_line:
                header_lines.append(f"[FILE: {file_path}:{start_line}-{end_line}]")
            else:
                header_lines.append(f"[FILE: {file_path}]")

            # Language/layer tag
            layer = _get_layer_type(file_path)
            lang = md.get("language", layer.upper())
            if lang:
                header_lines.append(f"[LAYER: {lang.upper()}]")

            # Cross-language binding chain
            bindings = _get_bindings_metadata(md)
            if bindings:
                binding_strs = []
                for b in bindings[:5]:
                    py_name = b.get("python_name", "?")
                    cpp_name = b.get("cpp_name", "?")
                    bind_type = b.get("type", "binding")
                    binding_strs.append(f"{py_name} → {cpp_name} ({bind_type})")
                header_lines.append(f"[BINDINGS: {'; '.join(binding_strs)}]")

            # Connection info (added during graph expansion)
            if md.get("_connected_from"):
                header_lines.append(f"[CONNECTED FROM: {md['_connected_from']}]")

            header = "\n".join(header_lines)
            original_text = node.node.get_content()
            node.node.text = f"{header}\n\n{original_text}"
            node.node.metadata["_context_formatted"] = True

        return nodes


# =============================================================================
# Graph-Expanded Retriever
# =============================================================================


class GraphExpandedRetriever(BaseRetriever):
    """
    Retriever that uses graph expansion to follow binding relationships.

    Architecture:
        1. Hybrid search (BM25 + Vector with RRF) → Entry points
        2. Graph expansion → Follow binding relationships to connected chunks
        3. Rerank → Cross-encoder for precision
        4. Format → Add metadata headers for LLM

    Key insight: Instead of forcing "3 YAML, 5 CUDA, 5 C++" slots, we follow
    the ACTUAL relationships. If dropout has Python → C++ → CUDA bindings,
    we retrieve all three. If an op is Python-only, we only get Python.

    The graph structure is implicit in the chunk metadata (cross_language_bindings),
    not a separate graph database query.
    """

    def __init__(
        self,
        index: Union[VectorStoreIndex, PropertyGraphIndex],
        similarity_top_k: int = 40,
        rerank_top_n: int = 15,
        expansion_depth: int = 1,
        max_expanded_per_entry: int = 5,
        max_final_nodes: int = 25,
        rerank_model: str = "BAAI/bge-reranker-base",
    ):
        """
        Args:
            index: LlamaIndex index (VectorStoreIndex or PropertyGraphIndex)
            similarity_top_k: Initial retrieval count for hybrid search
            rerank_top_n: Number of nodes to keep after reranking
            expansion_depth: How many hops to follow in graph expansion (1 = direct connections)
            max_expanded_per_entry: Max expanded nodes per entry point
            max_final_nodes: Final cap on total nodes
            rerank_model: Cross-encoder model for reranking
        """
        super().__init__()

        self.is_property_graph = isinstance(index, PropertyGraphIndex)

        # Get docstore
        if self.is_property_graph:
            self.docstore = index.storage_context.docstore
            vector_store = index.storage_context.vector_stores.get("default")
            vector_retriever = _SimpleVectorRetriever(
                vector_store=vector_store,
                docstore=self.docstore,
                embed_model=Settings.embed_model,
                similarity_top_k=similarity_top_k,
            )
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.docstore,
                similarity_top_k=similarity_top_k,
            )
        else:
            self.docstore = index.docstore
            vector_retriever = VectorIndexRetriever(
                index=index, similarity_top_k=similarity_top_k
            )
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.docstore,
                similarity_top_k=similarity_top_k,
            )

        # Hybrid retriever with RRF fusion
        from llama_index.core.llms import MockLLM

        self.hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            llm=MockLLM(),
            similarity_top_k=similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

        # Binding index for graph expansion
        self.binding_index = BindingIndex(self.docstore)

        # File importance booster (runs BEFORE reranker - critical for quality)
        # Check if index has file_importance metadata (new PageRank-based scoring)
        # by sampling a few documents. If not, fall back to hardcoded path patterns.
        self._use_file_importance = self._check_file_importance_available()
        if self._use_file_importance:
            log.info("Using FileImportancePostprocessor (PageRank-based scoring)")
            self.importance_booster = FileImportancePostprocessor()
        else:
            log.info("Using PathBoostPostprocessor (fallback hardcoded patterns)")
            self.importance_booster = PathBoostPostprocessor()

        # Reranker and formatter
        self.reranker = SentenceTransformerRerank(
            model=rerank_model, top_n=rerank_top_n
        )
        self.reorderer = LongContextReorder()
        self.context_formatter = ContextFormatterPostprocessor()

        # Config
        self.expansion_depth = expansion_depth
        self.max_expanded_per_entry = max_expanded_per_entry
        self.max_final_nodes = max_final_nodes

        log.info(
            f"GraphExpandedRetriever: hybrid(vector+bm25) → graph_expand(depth={expansion_depth}) "
            f"→ rerank(top_n={rerank_top_n}) → cap={max_final_nodes}"
        )

    def _check_file_importance_available(self) -> bool:
        """Check if the index has file_importance metadata (new PageRank-based scoring)."""
        try:
            if self.docstore is None:
                return False
            # Sample a few documents to check for file_importance metadata
            docs = list(self.docstore.docs.values())[:10]
            for doc in docs:
                if doc.metadata.get("file_importance") is not None:
                    return True
            return False
        except Exception:
            return False

    def _expand_via_bindings(
        self,
        entry_nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Expand entry points by following binding relationships.

        For each entry node that has bindings, find all connected chunks
        (chunks that share the same binding names).

        This is the "graph traversal" - but we do it via metadata lookup,
        not a graph database query.
        """
        seen_ids = {n.node.node_id for n in entry_nodes}
        expanded_nodes = []

        for entry_node in entry_nodes:
            entry_id = entry_node.node.node_id
            entry_path = entry_node.node.metadata.get("rel_path", "")

            # Get connected node IDs via binding relationships
            connected_ids = self.binding_index.get_connected_node_ids(entry_id)

            if not connected_ids:
                continue

            # Fetch connected nodes, tracking which layers we're adding
            added_for_this_entry = 0
            layers_added = set()

            for connected_id in connected_ids:
                if connected_id in seen_ids:
                    continue
                if added_for_this_entry >= self.max_expanded_per_entry:
                    break

                try:
                    connected_node = self.docstore.get_node(connected_id)
                    if not connected_node:
                        continue

                    connected_path = connected_node.metadata.get("rel_path", "")
                    connected_layer = _get_layer_type(connected_path)

                    # Prioritize getting different layers (Python, C++, CUDA, YAML)
                    # to ensure we have the complete chain
                    if connected_layer in layers_added and added_for_this_entry > 2:
                        continue

                    # Mark where this node came from (for context formatting)
                    connected_node.metadata["_connected_from"] = entry_path

                    expanded_nodes.append(
                        NodeWithScore(
                            node=connected_node,
                            score=(entry_node.score or 0.5)
                            * 0.8,  # Slightly lower score
                        )
                    )
                    seen_ids.add(connected_id)
                    layers_added.add(connected_layer)
                    added_for_this_entry += 1

                except Exception as e:
                    log.debug(f"Error fetching connected node {connected_id}: {e}")
                    continue

        if expanded_nodes:
            # Log what layers we expanded to
            layer_counts = {}
            for n in expanded_nodes:
                layer = _get_layer_type(n.node.metadata.get("rel_path", ""))
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            log.info(
                f"[GraphExpand] Added {len(expanded_nodes)} connected nodes: {layer_counts}"
            )

        return expanded_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Main retrieval pipeline:
        0. Dual-query retrieval (original + expanded)
        1. Hybrid search → Entry points
        2. Graph expansion → Follow bindings
        3. Deduplicate & merge
        4. Rerank
        5. Format
        """
        query_str = query_bundle.query_str

        # =================================================================
        # Step 0 & 1: Dual-query retrieval
        # Run BOTH original query and expanded query, then merge results.
        # This ensures we find both high-level Python API and low-level kernels.
        # =================================================================

        # First, get results from original query (finds Python API, user-facing code)
        entry_nodes = self.hybrid_retriever.retrieve(query_bundle)
        log.info(f"[Retrieval] Original query returned {len(entry_nodes)} nodes")

        # Then, check if query expansion adds value
        expanded_query = _expand_pytorch_query(query_str)
        if expanded_query != query_str:
            # Run expanded query to find kernels and dispatch config
            expanded_bundle = QueryBundle(query_str=expanded_query)
            expanded_nodes = self.hybrid_retriever.retrieve(expanded_bundle)
            log.info(f"[Retrieval] Expanded query returned {len(expanded_nodes)} nodes")

            # Merge results - expanded nodes get slightly lower base score
            seen_ids = {n.node.node_id for n in entry_nodes}
            for node in expanded_nodes:
                if node.node.node_id not in seen_ids:
                    # Give expanded results slightly lower score to prioritize original matches
                    node.score = (node.score or 0.5) * 0.9
                    entry_nodes.append(node)
                    seen_ids.add(node.node.node_id)

            log.info(f"[Retrieval] After merge: {len(entry_nodes)} unique nodes")

        # =================================================================
        # Step 2: Graph expansion - follow binding relationships
        # =================================================================
        expanded_nodes = self._expand_via_bindings(entry_nodes)

        # Merge entry nodes and expanded nodes
        all_nodes = entry_nodes + expanded_nodes

        # Deduplicate by node_id
        seen_ids = set()
        unique_nodes = []
        for node in all_nodes:
            if node.node.node_id not in seen_ids:
                seen_ids.add(node.node.node_id)
                unique_nodes.append(node)

        log.info(f"[Retrieval] After graph expansion: {len(unique_nodes)} unique nodes")

        # =================================================================
        # Step 3: File importance boosting (BEFORE reranker)
        # Uses PageRank-based scores when available, else hardcoded patterns
        # Research shows boosting before reranking improves quality
        # =================================================================
        pre_rerank_nodes = []
        if unique_nodes:
            unique_nodes = self.importance_booster.postprocess_nodes(
                unique_nodes, query_bundle=query_bundle
            )
            # Save a copy of boosted nodes before reranking for rescue
            pre_rerank_nodes = list(unique_nodes)
            log.info(
                "[Retrieval] After importance boosting: top candidates are implementation files"
            )

        # =================================================================
        # Step 4: Rerank with cross-encoder
        # =================================================================
        if unique_nodes:
            unique_nodes = self.reranker.postprocess_nodes(
                unique_nodes, query_str=query_str
            )
            log.info(f"[Retrieval] After reranking: {len(unique_nodes)} nodes")

        # =================================================================
        # Step 4b: Apply importance boosting AFTER reranking to correct for
        # cross-encoder bias towards simpler wrapper code
        # =================================================================
        if unique_nodes:
            unique_nodes = self.importance_booster.postprocess_nodes(
                unique_nodes, query_bundle=query_bundle
            )
            log.info(
                "[Retrieval] After post-rerank importance boosting: implementation files prioritized"
            )

        # =================================================================
        # Step 4c: Rescue high-value files that reranker underscored
        # Look at the boosted results BEFORE reranking to find
        # important files that got filtered by the cross-encoder
        # =================================================================
        if unique_nodes and pre_rerank_nodes:
            # Get IDs already in results
            reranked_ids = {n.node.node_id for n in unique_nodes}

            rescued = []
            # Look at top 20 boosted nodes from BEFORE reranking
            for node in pre_rerank_nodes[:20]:
                if node.node.node_id in reranked_ids:
                    continue

                # Rescue files with high file_importance (general approach)
                file_importance = node.node.metadata.get("file_importance", 0.0)
                has_bindings = bool(_get_bindings_metadata(node.node.metadata))

                # Rescue if: high importance score OR has cross-language bindings
                if file_importance > 0.7 or has_bindings:
                    # Use importance-based score
                    node.score = 1.0 + file_importance
                    rescued.append(node)
                    reranked_ids.add(node.node.node_id)

                if len(rescued) >= 5:  # Limit rescued
                    break

            if rescued:
                log.info(f"[Retrieval] Rescued {len(rescued)} high-importance files")
                unique_nodes.extend(rescued)
                unique_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        # =================================================================
        # Step 4d: Layer diversity rescue - ensure we have coverage across
        # Python, YAML, C++, and CUDA layers for implementation queries
        # =================================================================
        if unique_nodes and pre_rerank_nodes:
            # Check which layers we already have
            present_layers = set()
            for node in unique_nodes:
                path = node.node.metadata.get("rel_path", "")
                layer = _get_layer_type(path)
                present_layers.add(layer)

            # Target layers for implementation queries
            target_layers = {"python", "yaml", "cpp", "cuda"}
            missing_layers = target_layers - present_layers

            if missing_layers:
                reranked_ids = {n.node.node_id for n in unique_nodes}
                layer_rescued = []

                # Search through ALL pre-rerank nodes (not just top 20)
                for node in pre_rerank_nodes:
                    if node.node.node_id in reranked_ids:
                        continue

                    path = node.node.metadata.get("rel_path", "")
                    layer = _get_layer_type(path)

                    if layer in missing_layers:
                        # Prioritize certain implementation files
                        is_impl_file = (
                            "torch/nn/functional.py" in path
                            or "torch/autograd/" in path
                            or "derivatives.yaml" in path
                            or ("/cuda/" in path and path.endswith(".cu"))
                            or (
                                "aten/src/ATen/native/" in path
                                and path.endswith(".cpp")
                            )
                        )

                        if is_impl_file:
                            # Give good score to rescued layer-diverse files
                            node.score = 1.5
                            layer_rescued.append(node)
                            reranked_ids.add(node.node.node_id)
                            missing_layers.discard(layer)

                        if not missing_layers:
                            break

                if layer_rescued:
                    log.info(
                        f"[Retrieval] Layer diversity rescue: added {len(layer_rescued)} files from missing layers"
                    )
                    unique_nodes.extend(layer_rescued)
                    unique_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        # =================================================================
        # Step 4e: Query-aware key file rescue - ensure critical files
        # appear based on query keywords
        # =================================================================
        if unique_nodes and pre_rerank_nodes:
            query_lower = query_str.lower()
            reranked_ids = {n.node.node_id for n in unique_nodes}
            key_file_rescued = []

            # Define query-to-key-file mappings
            key_file_rules = []

            # Autograd queries need derivatives.yaml
            if any(
                kw in query_lower
                for kw in ["autograd", "gradient", "backward", "derivative"]
            ):
                key_file_rules.append(("derivatives.yaml", "yaml"))

            # Operation queries need functional.py
            # Check if any operation name pattern is present
            import re

            op_pattern = re.search(
                r"\b(layer_norm|dropout|batch_norm|conv|relu|softmax|matmul)\b",
                query_lower,
            )
            if op_pattern or any(
                kw in query_lower
                for kw in ["function", "nn.functional", "operation", "op"]
            ):
                key_file_rules.append(("torch/nn/functional.py", "python"))

            # Apply rules
            for key_pattern, layer in key_file_rules:
                # Check if already present
                already_present = any(
                    key_pattern in n.node.metadata.get("rel_path", "")
                    for n in unique_nodes
                )
                if already_present:
                    continue

                # Find in pre-rerank results
                for node in pre_rerank_nodes:
                    if node.node.node_id in reranked_ids:
                        continue

                    path = node.node.metadata.get("rel_path", "")
                    if key_pattern in path:
                        node.score = 1.8  # High score for key files
                        key_file_rescued.append(node)
                        reranked_ids.add(node.node.node_id)
                        break  # Only rescue one per pattern

            if key_file_rescued:
                log.info(
                    f"[Retrieval] Key file rescue: added {len(key_file_rescued)} query-relevant files"
                )
                unique_nodes.extend(key_file_rescued)
                unique_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        # =================================================================
        # Step 5: Cap to max_final_nodes
        # =================================================================
        if len(unique_nodes) > self.max_final_nodes:
            unique_nodes = unique_nodes[: self.max_final_nodes]
            log.info(f"[Retrieval] Capped to {len(unique_nodes)} nodes")

        # =================================================================
        # Step 6: Long context reorder
        # =================================================================
        if unique_nodes:
            unique_nodes = self.reorderer.postprocess_nodes(
                unique_nodes, query_str=query_str
            )

        # =================================================================
        # Step 7: Format with metadata headers
        # =================================================================
        if unique_nodes:
            unique_nodes = self.context_formatter.postprocess_nodes(
                unique_nodes, query_bundle=query_bundle
            )

        # Handle empty results
        if not unique_nodes:
            log.info("No relevant code context found - returning fallback")
            from llama_index.core.schema import TextNode

            fallback = TextNode(
                text="No relevant PyTorch code was found for this query. "
                "Please try rephrasing or ask about a specific operation.",
                metadata={"file_path": "system", "is_fallback": True},
            )
            return [NodeWithScore(node=fallback, score=1.0)]

        # Log final results
        log.info(f"[Retrieval] Final {len(unique_nodes)} nodes:")
        for i, node in enumerate(unique_nodes[:8]):
            md = node.node.metadata or {}
            path = md.get("rel_path") or md.get("file_path", "?")
            layer = _get_layer_type(path)
            connected = (
                " ← " + md.get("_connected_from", "")[:30]
                if md.get("_connected_from")
                else ""
            )
            log.info(f"  {i + 1}. [{layer:6}] {path[:60]}{connected}")

        return unique_nodes
