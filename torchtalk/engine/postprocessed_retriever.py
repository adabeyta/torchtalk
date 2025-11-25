"""
Hybrid retriever with BM25 + vector search and postprocessing pipeline.

Uses LlamaIndex QueryFusionRetriever for hybrid retrieval (dense + sparse),
followed by reranking, filtering, and context optimization.

Supports both VectorStoreIndex and PropertyGraphIndex:
- VectorStoreIndex: BM25 + vector fusion with metadata-only graph info
- PropertyGraphIndex: BM25 + vector fusion (text embeddings stored in vector store)

Key feature: Query-aware binding lookup
- Extracts operation names from queries (e.g., "dropout" from "where is dropout implemented")
- Directly retrieves the binding chain: Python API → native_functions.yaml → C++/CUDA kernels
- Merges binding nodes with main retrieval results for complete cross-language traces
"""

import json
import logging
import re
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Union
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, QueryBundle, Settings
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever, QueryFusionRetriever
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
    LongContextReorder,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

log = logging.getLogger(__name__)


def _get_bindings_metadata(metadata: dict) -> List[dict]:
    """
    Get cross_language_bindings from metadata, handling both JSON strings and lists.

    LanceDB requires flat metadata (str, int, float, None), so bindings are stored
    as JSON strings during indexing. This function deserializes them for use in
    the retrieval pipeline.
    """
    bindings = metadata.get("cross_language_bindings", [])
    if isinstance(bindings, str):
        try:
            bindings = json.loads(bindings) if bindings else []
        except json.JSONDecodeError:
            bindings = []
    return bindings if isinstance(bindings, list) else []


# =============================================================================
# Query-Aware Binding Lookup
# =============================================================================

# Common PyTorch operation names to look for in queries
# These are operations that have cross-language implementations (Python → C++ → CUDA)
# NOTE: Avoid generic English words like "to", "where", "sum", "mean" etc. that
# commonly appear in natural language queries but pollute the binding lookup.
PYTORCH_OPS = {
    # Neural network operations (high precision - rarely used as regular words)
    "dropout", "softmax", "layer_norm", "batch_norm", "conv2d", "conv1d", "conv3d",
    "linear", "relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu",
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "embedding",
    "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "bce_loss",
    "scaled_dot_product_attention", "multi_head_attention",
    # Tensor operations (compound names - high precision)
    "matmul", "addmm", "baddbmm",
    "normalize", "index_select", "masked_fill",
    # Reduction operations
    "argmax", "argmin", "topk", "argsort",
    # Advanced operations
    "einsum", "tensordot", "rfft", "irfft",
    "conv_transpose2d", "unfold",
}

# Patterns to extract operation names from natural language queries
OP_EXTRACTION_PATTERNS = [
    # Direct mentions: "the dropout function", "torch.dropout"
    r"(?:torch\.(?:nn\.functional\.|nn\.)?|F\.|aten::)?(\w+?)(?:\s+function|\s+op(?:eration)?|\s+kernel|\s+implementation)?",
    # "how does X work", "where is X implemented"
    r"(?:how\s+(?:does|is)|where\s+is|what\s+is|explain)\s+(?:the\s+)?(?:torch\.(?:nn\.)?)?(\w+)",
    # "X layer", "X module"
    r"(\w+)\s+(?:layer|module|operator|kernel)",
    # Underscore variants: layer_norm, batch_norm
    r"\b(\w+_\w+)\b",
]


def extract_ops_from_query(query: str) -> Set[str]:
    """
    Extract PyTorch operation names from a natural language query.

    Examples:
        "Where is dropout implemented?" → {"dropout"}
        "How does layer_norm work from Python to CUDA?" → {"layer_norm"}
        "Explain torch.nn.functional.softmax" → {"softmax"}

    Returns:
        Set of operation names found in the query (lowercase)
    """
    query_lower = query.lower()
    found_ops = set()

    # First, check for exact matches of known ops
    for op in PYTORCH_OPS:
        # Match as whole word (not part of another word)
        if re.search(rf'\b{re.escape(op)}\b', query_lower):
            found_ops.add(op)

    # Also extract potential op names using patterns
    for pattern in OP_EXTRACTION_PATTERNS:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            # Clean up the match
            op_name = match.strip().lower()
            # Filter out common non-op words and ambiguous terms
            # These either appear too often in natural language or match too many files
            if op_name and len(op_name) > 2 and op_name not in {
                # Common English words
                "the", "how", "what", "where", "does", "from", "this", "that",
                "with", "for", "and", "are", "can", "you", "down", "into",
                # Technical but non-specific terms
                "pytorch", "torch", "python", "cuda", "gpu", "cpu", "mps",
                "implementation", "implemented", "function", "functional",
                "kernel", "native", "aten", "file", "path", "module", "class",
                "show", "explain", "tell", "code", "source", "work", "works",
                # Ambiguous PyTorch ops (common English words)
                "to", "sum", "mean", "var", "std", "norm", "cat", "stack",
                "split", "chunk", "squeeze", "view", "clone", "copy", "sort",
                "add", "sub", "mul", "div", "pow", "exp", "log", "sqrt",
                "sin", "cos", "abs", "clamp", "where", "fold", "gather",
                "scatter", "transpose", "permute", "reshape", "contiguous",
                "attention", "mm", "bmm", "fft",
            }:
                found_ops.add(op_name)

    return found_ops


def is_implementation_query(query: str) -> bool:
    """
    Check if the query is asking about implementation details.

    These queries benefit most from binding chain lookup because they need
    to trace from Python API → dispatch → kernel implementations.
    """
    query_lower = query.lower()
    implementation_keywords = [
        "implement", "implementation", "implemented",
        "kernel", "cuda", "gpu",
        "native", "aten", "dispatch",
        "c++", "cpp", "source",
        "how does", "how is", "where is",
        "trace", "path", "chain",
        "from python to", "python to cuda", "python to c++",
    ]
    return any(kw in query_lower for kw in implementation_keywords)


def _extract_base_op_name(name: str) -> Optional[str]:
    """
    Extract the base operation name from variants like native_dropout_cuda.

    This normalizes operation names so that:
    - native_dropout → dropout
    - native_dropout_cuda → dropout
    - fused_dropout_cuda → dropout
    - layer_norm_backward → layer_norm
    - cudnn_batch_norm → batch_norm

    Returns None if the name is too short or only prefixes/suffixes.
    """
    if not name or len(name) < 2:
        return None

    original = name

    # Strip common prefixes (order matters - longer prefixes first)
    prefixes = ["native_", "fused_", "cudnn_", "mkldnn_", "at_", "_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Strip common suffixes (order matters - longer suffixes first)
    suffixes = [
        "_backward", "_forward",
        "_cuda", "_cpu", "_mps", "_nested", "_sparse", "_mkldnn",
        "_impl", "_kernel", "_out", "_stub",
    ]
    # May need multiple passes for compound suffixes like "_backward_cuda"
    for _ in range(2):
        for suffix in suffixes:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]
                break

    # If we stripped everything or it's the same, return None
    if not name or len(name) < 2 or name == original:
        return None

    return name


class BindingChainLookup:
    """
    Direct lookup of cross-language binding chains from the docstore.

    This is used for query-aware retrieval: when a user asks about a specific
    operation (e.g., "where is dropout implemented"), we directly look up nodes
    that have bindings for that operation, ensuring we get the complete chain:
    - Python API (torch/nn/functional.py)
    - Dispatch definition (native_functions.yaml)
    - C++ implementation (aten/src/ATen/native/)
    - CUDA kernel (aten/src/ATen/native/cuda/)

    This solves the problem where semantic search might miss the YAML file or
    kernel implementations because they don't have high textual similarity to
    the query.
    """

    def __init__(self, docstore):
        self.docstore = docstore
        self._binding_index: Dict[str, List[str]] = {}  # op_name → [node_ids]
        self._indexed = False

    def _add_to_index(self, name: str, doc_id: str):
        """Helper to add a name→doc_id mapping to the index."""
        if name:
            if name not in self._binding_index:
                self._binding_index[name] = []
            if doc_id not in self._binding_index[name]:
                self._binding_index[name].append(doc_id)

    def _build_index(self):
        """Build an index from operation names to node IDs."""
        if self._indexed or self.docstore is None:
            return

        try:
            all_docs = self.docstore.docs
            for doc_id, doc in all_docs.items():
                bindings = _get_bindings_metadata(doc.metadata)
                for b in bindings:
                    # Index by both python_name and cpp_name
                    py_name = b.get("python_name", "").lower()
                    cpp_name = b.get("cpp_name", "").lower()

                    # Normalize names (strip aten:: prefix, etc.)
                    py_name = py_name.replace("aten::", "").replace("torch.", "")
                    cpp_name = cpp_name.replace("aten::", "")

                    for name in [py_name, cpp_name]:
                        if name:
                            # Index under the full name (e.g., "native_dropout_cuda")
                            self._add_to_index(name, doc_id)

                            # Also index under the base name (e.g., "dropout")
                            # This allows users to query "dropout" and find all variants
                            base_name = _extract_base_op_name(name)
                            if base_name:
                                self._add_to_index(base_name, doc_id)

            self._indexed = True
            log.info(f"[BindingChainLookup] Indexed {len(self._binding_index)} operation names")
        except Exception as e:
            log.warning(f"[BindingChainLookup] Failed to build index: {e}")

    def _is_python_api_file(self, rel_path: str) -> bool:
        """Check if a file is part of the Python API layer."""
        python_api_patterns = [
            "torch/nn/functional",      # F.dropout, F.softmax, etc.
            "torch/nn/modules/",        # nn.Dropout, nn.Linear, etc.
            "torch/_C",                 # C extension bindings
            "torch/functional.py",      # torch.functional
            "torch/_ops.py",            # Operator definitions
            "torch/_decomp/",           # Decompositions
        ]
        return any(p in rel_path for p in python_api_patterns)

    def _get_file_priority(self, rel_path: str) -> tuple:
        """
        Get priority tuple for sorting (lower = higher priority).
        Returns (priority_tier, path) for stable sorting.
        """
        if "native_functions.yaml" in rel_path:
            return (0, rel_path)  # Dispatch definitions - highest priority
        elif "aten/src/ATen/native/cuda" in rel_path and rel_path.endswith(".cu"):
            return (1, rel_path)  # CUDA kernels
        elif "aten/src/ATen/native" in rel_path and rel_path.endswith(".cpp"):
            return (2, rel_path)  # C++ implementations
        elif self._is_python_api_file(rel_path):
            return (3, rel_path)  # Python API
        else:
            return (4, rel_path)  # Other files

    def _get_score_for_path(self, rel_path: str) -> float:
        """Get retrieval score based on file path."""
        if "native_functions.yaml" in rel_path:
            return 0.95  # Dispatch definitions are critical
        elif "aten/src/ATen/native/cuda" in rel_path:
            return 0.92  # CUDA kernels
        elif "aten/src/ATen/native" in rel_path:
            return 0.90  # C++ implementations
        elif self._is_python_api_file(rel_path):
            return 0.88  # Python API
        else:
            return 0.80  # Other related files

    def lookup(self, op_names: Set[str], max_per_op: int = 10) -> List[NodeWithScore]:
        """
        Look up nodes that have bindings for the given operation names.

        Returns a diverse mix of file types to ensure we get:
        - native_functions.yaml (dispatch definitions)
        - CUDA kernels (.cu)
        - C++ implementations (.cpp)
        - Python API

        Args:
            op_names: Set of operation names to look up
            max_per_op: Maximum nodes to return per operation

        Returns:
            List of NodeWithScore with binding chain nodes
        """
        self._build_index()

        if not op_names or not self._binding_index:
            return []

        nodes = []
        seen_ids = set()

        for op_name in op_names:
            op_lower = op_name.lower()

            # Get all node_ids for this op from the index
            node_ids = self._binding_index.get(op_lower, [])

            if not node_ids:
                # Fallback: try partial matches
                for indexed_name, ids in self._binding_index.items():
                    if op_lower in indexed_name or indexed_name in op_lower:
                        node_ids.extend(ids)

            # Deduplicate
            unique_ids = []
            for nid in node_ids:
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    unique_ids.append(nid)

            # Group by file type tier to ensure diversity
            # tier 0: native_functions.yaml, tier 1: CUDA, tier 2: C++, tier 3: Python, tier 4: other
            tiers = {0: [], 1: [], 2: [], 3: [], 4: []}
            for nid in unique_ids:
                try:
                    node = self.docstore.get_node(nid)
                    rel_path = node.metadata.get("rel_path", "") if node else ""
                    tier = self._get_file_priority(rel_path)[0]
                    tiers[tier].append(nid)
                except Exception:
                    pass

            # Select nodes ensuring diversity: at least some from each tier that has content
            # Allocate slots: 3 for YAML, 3 for CUDA, 3 for C++, 2 for Python, rest for other
            slot_allocation = {0: 3, 1: 3, 2: 3, 3: 2, 4: max_per_op}
            selected = []

            for tier in [0, 1, 2, 3, 4]:
                tier_nodes = tiers[tier]
                slots = slot_allocation[tier]
                selected.extend(tier_nodes[:slots])
                if len(selected) >= max_per_op:
                    break

            # If we have fewer than max_per_op, fill with remaining from any tier
            if len(selected) < max_per_op:
                remaining = max_per_op - len(selected)
                all_remaining = []
                for tier in [0, 1, 2, 3, 4]:
                    slots = slot_allocation[tier]
                    all_remaining.extend(tiers[tier][slots:])
                selected.extend(all_remaining[:remaining])

            # Convert to NodeWithScore
            for node_id in selected[:max_per_op]:
                try:
                    node = self.docstore.get_node(node_id)
                    if node:
                        rel_path = node.metadata.get("rel_path", "")
                        score = self._get_score_for_path(rel_path)
                        nodes.append(NodeWithScore(node=node, score=score))
                except Exception:
                    continue

        if nodes:
            log.info(f"[BindingChainLookup] Found {len(nodes)} nodes for ops: {op_names}")

        return nodes


class BindingExpansionPostprocessor(BaseNodePostprocessor):
    """
    Expand retrieval results by finding related cross-language implementations.

    When a node has cross-language bindings (e.g., layer_norm → layer_norm_cuda),
    this postprocessor searches the docstore for nodes that implement those bindings.

    This enables complete cross-language traces:
    - If we retrieve a Python API, also get the CUDA kernel
    - If we retrieve a CUDA kernel, also get the Python API
    - If we retrieve dispatch info, get both ends

    This is CRITICAL for response quality because the LLM needs to see all layers
    of the implementation chain to provide accurate answers.
    """

    # Pydantic model config
    model_config = {"arbitrary_types_allowed": True}

    docstore: object = None
    max_expansion: int = 5  # Max additional nodes to add per binding

    def __init__(self, docstore=None, max_expansion: int = 5, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'docstore', docstore)
        object.__setattr__(self, 'max_expansion', max_expansion)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Expand nodes by finding related binding implementations."""
        if self.docstore is None:
            return nodes

        # Collect all binding names we need to search for
        binding_names = set()
        seen_node_ids = {n.node.node_id for n in nodes}

        for node in nodes:
            bindings = _get_bindings_metadata(node.node.metadata)
            for b in bindings:
                # Add both Python and C++ names
                binding_names.add(b.get("python_name", "").lower())
                binding_names.add(b.get("cpp_name", "").lower())

        if not binding_names:
            return nodes

        # Search docstore for nodes containing these binding names
        expanded_nodes = []
        try:
            all_docs = self.docstore.docs
            for doc_id, doc in all_docs.items():
                if doc_id in seen_node_ids:
                    continue

                # Check if this node has matching bindings
                doc_bindings = _get_bindings_metadata(doc.metadata)
                for b in doc_bindings:
                    py_name = b.get("python_name", "").lower()
                    cpp_name = b.get("cpp_name", "").lower()

                    if py_name in binding_names or cpp_name in binding_names:
                        # Found a related node!
                        expanded_nodes.append(NodeWithScore(
                            node=doc,
                            score=0.5  # Lower score than original results
                        ))
                        seen_node_ids.add(doc_id)
                        break

                if len(expanded_nodes) >= self.max_expansion * len(nodes):
                    break  # Don't expand too much

        except Exception as e:
            log.warning(f"BindingExpansionPostprocessor error: {e}")

        if expanded_nodes:
            log.info(f"[Retrieval] Binding expansion added {len(expanded_nodes)} related nodes")
            nodes = nodes + expanded_nodes

        return nodes


class ContextFormatterPostprocessor(BaseNodePostprocessor):
    """
    Format nodes with file paths, line numbers, and binding info for the LLM.

    This is CRITICAL for response quality - without this, the LLM only sees
    raw code without knowing:
    - Which file it came from
    - What line numbers it spans
    - What cross-language bindings exist (Python ↔ C++ ↔ CUDA)

    The formatted context enables the LLM to:
    - Cite specific file:line locations
    - Trace cross-language implementation chains
    - Provide accurate, grounded responses
    """

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Prepend metadata header to each node's text content."""
        for node in nodes:
            md = node.node.metadata or {}

            # Skip if already formatted (avoid double-formatting)
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

            # Language tag
            lang = md.get("language", "")
            if lang:
                header_lines.append(f"[LANGUAGE: {lang.upper()}]")

            # Cross-language bindings - THE KEY FOR RESPONSE QUALITY
            # Note: This is computed metadata (not from file content) to help trace implementations
            bindings = _get_bindings_metadata(md)
            if bindings:
                binding_strs = []
                for b in bindings[:5]:  # Show up to 5 bindings
                    py_name = b.get("python_name", "?")
                    cpp_name = b.get("cpp_name", "?")
                    bind_type = b.get("type", "binding")
                    # Format: python_name → cpp_name (type)
                    binding_strs.append(f"{py_name} → {cpp_name} ({bind_type})")
                header_lines.append(f"[CROSS-LANGUAGE BINDINGS (computed, not file content): {'; '.join(binding_strs)}]")

            # Build final header
            header = "\n".join(header_lines)

            # Get original text and prepend header
            original_text = node.node.get_content()
            formatted_text = f"{header}\n\n{original_text}"

            # Update node text (create new TextNode to avoid modifying original)
            node.node.text = formatted_text
            node.node.metadata["_context_formatted"] = True

        return nodes


class PathBoostPostprocessor(BaseNodePostprocessor):
    """
    Boost scores for nodes from high-value implementation paths.

    PyTorch's real implementations live in aten/src/ATen/native/, not in
    the API wrappers (torch/csrc/api/). This postprocessor boosts scores
    for ATen native files to ensure they rank higher than wrappers.
    """

    # Paths to boost (real implementations)
    BOOST_PATTERNS: ClassVar[List[Tuple[str, float]]] = [
        (r"aten/src/ATen/native/cuda/", 0.15),   # CUDA kernels
        (r"aten/src/ATen/native/cpu/", 0.12),    # CPU kernels
        (r"aten/src/ATen/native/.+\.(cpp|h)$", 0.10),  # ATen native implementations (any depth)
        (r"native_functions\.yaml", 0.08),       # Dispatch definitions
    ]

    # Paths to slightly penalize (wrappers, not implementations)
    PENALIZE_PATTERNS: ClassVar[List[Tuple[str, float]]] = [
        (r"torch/csrc/api/", -0.05),  # C++ API wrappers
        (r"torch/_refs/", -0.02),     # Reference implementations (not primary)
    ]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Apply path-based score adjustments."""
        for node in nodes:
            # Robust path extraction - handle different metadata key names
            md = node.node.metadata or {}
            rel_path = md.get("rel_path") or md.get("file_path") or md.get("path") or ""
            adjustment = 0.0

            # Check boost patterns
            for pattern, boost in self.BOOST_PATTERNS:
                if re.search(pattern, rel_path, re.IGNORECASE):
                    adjustment = max(adjustment, boost)
                    break

            # Check penalize patterns (only if no boost applied)
            if adjustment == 0:
                for pattern, penalty in self.PENALIZE_PATTERNS:
                    if re.search(pattern, rel_path, re.IGNORECASE):
                        adjustment = penalty
                        break

            if adjustment != 0:
                node.score = (node.score or 0) + adjustment

        # Re-sort by adjusted score
        nodes.sort(key=lambda n: n.score or 0, reverse=True)
        return nodes


class _SimpleVectorRetriever(BaseRetriever):
    """
    Simple vector retriever that queries a SimpleVectorStore directly and fetches
    nodes from a docstore. Used for PropertyGraphIndex where we store text node
    embeddings in the vector store but SimpleVectorStore doesn't store text itself.
    """

    def __init__(
        self,
        vector_store,
        docstore,
        embed_model,
        similarity_top_k: int = 10,
    ):
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Query vector store and fetch nodes from docstore."""
        from llama_index.core.vector_stores import VectorStoreQuery

        # Get query embedding
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)

        # Query vector store
        # Note: query_str is required for LanceDB hybrid search mode
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query_bundle.query_str,
            similarity_top_k=self._similarity_top_k,
        )
        result = self._vector_store.query(query)

        # Fetch nodes from docstore and build results
        nodes_with_scores = []
        if result.ids and result.similarities:
            for node_id, similarity in zip(result.ids, result.similarities):
                node = self._docstore.get_node(node_id)
                if node:
                    nodes_with_scores.append(NodeWithScore(node=node, score=float(similarity)))

        return nodes_with_scores


class PostprocessedRetriever(BaseRetriever):
    """
    Hybrid retriever with postprocessing: BM25+vector fusion → path boost → rerank → filter → reorder.

    Pipeline (same for both VectorStoreIndex and PropertyGraphIndex):
    0. Query-aware binding lookup: Extract op names, directly fetch binding chain nodes
    1. Hybrid search: BM25 (keyword) + vector (semantic) with reciprocal rank fusion
    2. Merge binding nodes with hybrid results
    3. Path boost: Lift ATen native files, penalize API wrappers
    4. Cross-encoder rerank (high precision)
    5. Similarity filter (quality threshold) - skipped for PropertyGraphIndex
    6. Long context reorder (LLM optimization)
    7. Final cap: Limit total nodes to max_final_nodes to prevent context overflow

    Key feature: Query-aware binding lookup ensures that when a user asks about a specific
    operation (e.g., "where is dropout implemented"), we directly retrieve the complete
    implementation chain (Python API → native_functions.yaml → CUDA kernel) even if
    semantic search would miss some files.

    Note: PropertyGraphIndex stores text node embeddings in the vector store (not graph entity
    embeddings), so we use the same hybrid retrieval approach. Cross-language binding metadata
    is stored in node.metadata for the LLM to use when tracing code paths.
    """

    def __init__(
        self,
        index: Union[VectorStoreIndex, PropertyGraphIndex],
        similarity_top_k: int = 60,
        rerank_top_n: int = 10,
        similarity_cutoff: float = 0.5,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        graph_path_depth: int = 2,
        max_final_nodes: int = 40,
    ):
        self.is_property_graph = isinstance(index, PropertyGraphIndex)

        if self.is_property_graph:
            # PropertyGraphIndex: The vector store contains TEXT NODE embeddings (not graph entity
            # embeddings), so we use hybrid retrieval. The graph metadata (cross-language bindings)
            # is stored in node.metadata for the LLM to use when tracing code paths.
            #
            # We use a custom vector retriever that queries the SimpleVectorStore directly
            # and fetches nodes from the docstore.
            vector_retriever = _SimpleVectorRetriever(
                vector_store=index.storage_context.vector_stores.get("default"),
                docstore=index.storage_context.docstore,
                embed_model=Settings.embed_model,
                similarity_top_k=similarity_top_k,
            )

            bm25_retriever = BM25Retriever.from_defaults(
                docstore=index.storage_context.docstore,
                similarity_top_k=similarity_top_k,
            )

            from llama_index.core.llms import MockLLM
            self.retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=MockLLM(),
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
            log.info(f"PostprocessedRetriever: PropertyGraph (hybrid vector+bm25) "
                     f"retrieve={similarity_top_k}")
        else:
            # VectorStoreIndex: use hybrid BM25 + vector fusion
            vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)

            bm25_retriever = BM25Retriever.from_defaults(
                docstore=index.docstore,
                similarity_top_k=similarity_top_k,
            )

            # Use a mock LLM to prevent QueryFusionRetriever from trying to resolve Settings.llm
            from llama_index.core.llms import MockLLM
            self.retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=MockLLM(),
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
            log.info(f"PostprocessedRetriever: hybrid(vector+bm25) retrieve={similarity_top_k}")

        self.path_booster = PathBoostPostprocessor()
        self.reranker = SentenceTransformerRerank(model=rerank_model, top_n=rerank_top_n)
        self.similarity_filter = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        self.reorderer = LongContextReorder()
        self.context_formatter = ContextFormatterPostprocessor()

        # Binding expansion: find related cross-language implementations
        docstore = index.storage_context.docstore if self.is_property_graph else index.docstore
        self.binding_expander = BindingExpansionPostprocessor(docstore=docstore, max_expansion=5)

        # Query-aware binding chain lookup: directly fetch binding chain for detected ops
        self.binding_chain_lookup = BindingChainLookup(docstore=docstore)

        # Final cap to prevent context overflow
        self.max_final_nodes = max_final_nodes

        log.info(f"  → binding_chain_lookup → rerank={rerank_top_n} → path_boost → binding_expand → filter(cutoff={similarity_cutoff}) → cap={max_final_nodes} → context_format")

    def _smart_cap_nodes(self, nodes: List[NodeWithScore], max_nodes: int) -> List[NodeWithScore]:
        """
        Smart capping that preserves critical file types.

        When we have more nodes than max_nodes, we need to cap, but we want to ensure
        critical files (YAML, CUDA, C++) are preserved because they contain the actual
        implementation details the user is asking about.

        Strategy:
        1. Separate nodes by file type (YAML, CUDA, C++, Python, other)
        2. Reserve slots for critical types (YAML: 3, CUDA: 5, C++: 5)
        3. Fill remaining slots with other nodes sorted by score
        """
        if len(nodes) <= max_nodes:
            return nodes

        log.info(f"[Retrieval] Smart capping from {len(nodes)} to {max_nodes} nodes")

        # Categorize nodes by file type
        yaml_nodes = []
        cuda_nodes = []
        cpp_nodes = []
        other_nodes = []

        for node in nodes:
            rel_path = node.node.metadata.get("rel_path", "") or node.node.metadata.get("file_path", "")

            if "native_functions.yaml" in rel_path:
                yaml_nodes.append(node)
            elif rel_path.endswith(".cu"):
                cuda_nodes.append(node)
            elif rel_path.endswith(".cpp") and "aten/src/ATen/native" in rel_path:
                cpp_nodes.append(node)
            else:
                other_nodes.append(node)

        # Sort each category by score (descending)
        yaml_nodes.sort(key=lambda n: n.score or 0, reverse=True)
        cuda_nodes.sort(key=lambda n: n.score or 0, reverse=True)
        cpp_nodes.sort(key=lambda n: n.score or 0, reverse=True)
        other_nodes.sort(key=lambda n: n.score or 0, reverse=True)

        # Reserve slots for critical file types
        # These are the actual implementations, not wrappers
        yaml_slots = min(3, len(yaml_nodes))
        cuda_slots = min(5, len(cuda_nodes))
        cpp_slots = min(5, len(cpp_nodes))
        reserved_slots = yaml_slots + cuda_slots + cpp_slots

        # Remaining slots for other files
        other_slots = max(0, max_nodes - reserved_slots)

        # Build final list
        capped_nodes = []
        capped_nodes.extend(yaml_nodes[:yaml_slots])
        capped_nodes.extend(cuda_nodes[:cuda_slots])
        capped_nodes.extend(cpp_nodes[:cpp_slots])
        capped_nodes.extend(other_nodes[:other_slots])

        # If we still have room (e.g., not enough critical files), fill with more from each category
        remaining = max_nodes - len(capped_nodes)
        if remaining > 0:
            # Collect unused nodes from all categories
            unused = (
                yaml_nodes[yaml_slots:] +
                cuda_nodes[cuda_slots:] +
                cpp_nodes[cpp_slots:] +
                other_nodes[other_slots:]
            )
            unused.sort(key=lambda n: n.score or 0, reverse=True)
            capped_nodes.extend(unused[:remaining])

        # Sort final list by score for consistent ordering before LongContextReorder
        capped_nodes.sort(key=lambda n: n.score or 0, reverse=True)

        log.info(f"[Retrieval] Smart cap preserved: {yaml_slots} YAML, {cuda_slots} CUDA, {cpp_slots} C++, {len(capped_nodes) - reserved_slots} other")

        return capped_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve and postprocess nodes through the pipeline."""
        query_str = query_bundle.query_str

        # =================================================================
        # Step 0: Query-aware binding chain lookup
        # =================================================================
        # Extract operation names from the query and directly fetch binding chain nodes.
        # This ensures we get native_functions.yaml and kernel implementations even if
        # semantic search would miss them.
        binding_chain_nodes = []
        detected_ops = extract_ops_from_query(query_str)
        is_impl_query = is_implementation_query(query_str)

        if detected_ops and is_impl_query:
            log.info(f"[Retrieval] Detected ops in query: {detected_ops}")
            binding_chain_nodes = self.binding_chain_lookup.lookup(detected_ops, max_per_op=12)
            if binding_chain_nodes:
                log.info(f"[Retrieval] Binding chain lookup found {len(binding_chain_nodes)} nodes")

        # =================================================================
        # Step 1: Hybrid retrieval (BM25 + vector)
        # =================================================================
        nodes = self.retriever.retrieve(query_bundle)
        log.info(f"[Retrieval] Initial hybrid retrieval returned {len(nodes)} nodes")

        # =================================================================
        # Step 2: Rerank with cross-encoder
        # =================================================================
        # Rerank hybrid results first (without binding chain nodes)
        nodes = self.reranker.postprocess_nodes(nodes, query_str=query_str)
        log.info(f"[Retrieval] After reranking: {len(nodes)} nodes")

        # =================================================================
        # Step 3: Merge binding chain nodes AFTER reranking
        # =================================================================
        # For implementation queries, we need to ensure critical files (YAML, C++, CUDA)
        # make it into the final results. We merge them after reranking so they're not
        # filtered out by the cross-encoder (which favors high textual similarity).
        if binding_chain_nodes and is_impl_query:
            # Separate binding chain nodes by type
            yaml_nodes = []
            cuda_nodes = []
            cpp_nodes = []
            other_binding_nodes = []

            for n in binding_chain_nodes:
                rel_path = n.node.metadata.get("rel_path", "")
                if "native_functions.yaml" in rel_path:
                    yaml_nodes.append(n)
                elif rel_path.endswith(".cu"):
                    cuda_nodes.append(n)
                elif rel_path.endswith(".cpp") and "aten/src/ATen/native" in rel_path:
                    cpp_nodes.append(n)
                else:
                    other_binding_nodes.append(n)

            # Check what we already have from reranking
            seen_ids = {n.node.node_id for n in nodes}
            reranked_paths = [n.node.metadata.get("rel_path", "") for n in nodes]

            has_yaml = any("native_functions.yaml" in p for p in reranked_paths)
            has_cuda = any(p.endswith(".cu") for p in reranked_paths)
            has_cpp = any(p.endswith(".cpp") and "aten/src/ATen/native" in p for p in reranked_paths)

            # Add missing critical file types (ensuring diversity)
            nodes_to_add = []

            if not has_yaml and yaml_nodes:
                for n in yaml_nodes[:2]:  # Add up to 2 YAML chunks
                    if n.node.node_id not in seen_ids:
                        nodes_to_add.append(n)
                        seen_ids.add(n.node.node_id)

            if not has_cuda and cuda_nodes:
                for n in cuda_nodes[:2]:  # Add up to 2 CUDA chunks
                    if n.node.node_id not in seen_ids:
                        nodes_to_add.append(n)
                        seen_ids.add(n.node.node_id)

            if not has_cpp and cpp_nodes:
                for n in cpp_nodes[:2]:  # Add up to 2 C++ chunks
                    if n.node.node_id not in seen_ids:
                        nodes_to_add.append(n)
                        seen_ids.add(n.node.node_id)

            if nodes_to_add:
                log.info(f"[Retrieval] Adding {len(nodes_to_add)} critical binding chain nodes (YAML/CUDA/C++)")
                nodes = nodes + nodes_to_add

        # =================================================================
        # Step 4: Path boost - lift ATen native/CUDA files
        # =================================================================
        nodes = self.path_booster.postprocess_nodes(nodes, query_bundle=query_bundle)

        # =================================================================
        # Step 5: Binding expansion - find related implementations
        # =================================================================
        # This is CRITICAL for response quality - when we find layer_norm_cuda,
        # also retrieve the Python API and dispatch definition
        nodes = self.binding_expander.postprocess_nodes(nodes, query_bundle=query_bundle)

        # NOTE: Skip similarity filter for PropertyGraphIndex because:
        # 1. PGRetriever returns scores in [0,1] range (cosine similarity)
        # 2. SentenceTransformerRerank returns cross-encoder logits (can be negative)
        # The reranker already selects top_n so further filtering is unnecessary
        if not self.is_property_graph:
            pre_filter_count = len(nodes)
            nodes = self.similarity_filter.postprocess_nodes(nodes, query_str=query_str)
            log.info(f"[Retrieval] After similarity filter: {len(nodes)} nodes (filtered {pre_filter_count - len(nodes)})")

        if not nodes:
            log.info("No relevant code context found - returning fallback node")
            from llama_index.core.schema import TextNode
            fallback_node = TextNode(
                text="No relevant PyTorch code was found for this query. "
                     "Respond naturally as a helpful PyTorch assistant. "
                     "If the user is greeting you or asking off-topic questions, "
                     "respond politely and offer to help with PyTorch codebase questions.",
                metadata={"file_path": "system", "is_fallback": True}
            )
            return [NodeWithScore(node=fallback_node, score=1.0)]

        # =================================================================
        # Step 6: Smart cap to prevent context overflow
        # =================================================================
        # After binding expansion, we may have way more nodes than max_final_nodes.
        # IMPORTANT: Cap BEFORE LongContextReorder so we don't lose critical files.
        # Use smart capping that preserves critical file types (YAML, CUDA, C++).
        if len(nodes) > self.max_final_nodes:
            nodes = self._smart_cap_nodes(nodes, self.max_final_nodes)

        # =================================================================
        # Step 7: Long context reorder for LLM attention optimization
        # =================================================================
        nodes = self.reorderer.postprocess_nodes(nodes, query_str=query_str)

        # Format context with file paths, line numbers, and binding metadata
        # This is CRITICAL - makes metadata visible to the LLM for accurate responses
        nodes = self.context_formatter.postprocess_nodes(nodes, query_bundle=query_bundle)

        # Log final retrieved files for debugging
        log.info(f"[Retrieval] Final {len(nodes)} nodes from files:")
        for i, node in enumerate(nodes[:10]):  # Log top 10
            md = node.node.metadata or {}
            file_path = md.get("rel_path") or md.get("file_path", "unknown")
            has_bindings = bool(_get_bindings_metadata(md))
            score = node.score or 0
            binding_marker = " [HAS BINDINGS]" if has_bindings else ""
            log.info(f"  {i+1}. [{score:.3f}] {file_path}{binding_marker}")
        if len(nodes) > 10:
            log.info(f"  ... and {len(nodes) - 10} more")

        return nodes
