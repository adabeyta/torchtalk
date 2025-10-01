#!/usr/bin/env python3
"""
Vector-based code retrieval using semantic similarity search.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from torchtalk.indexing.vector_store import TorchTalkVectorStore
from torchtalk.indexing.embedder import CodeEmbedder


class VectorRetriever:
    """
    Retrieve relevant code chunks using vector similarity search.

    Features:
    - Semantic search using embeddings
    - Metadata filtering (by file, type, etc.)
    - Score normalization
    - Result ranking by relevance
    """

    def __init__(
        self,
        index_dir: str,
        collection_name: str = "torchtalk_code",
        embedding_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the vector retriever.

        Args:
            index_dir: Directory containing the index
            collection_name: Name of the vector store collection
            embedding_model: HuggingFace model name (loads from metadata if None)
            device: Device for embeddings ('cpu' or 'cuda', auto-detect if None)
        """
        self.index_dir = Path(index_dir)

        # Load index metadata
        metadata_path = self.index_dir / "index_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Index not found at {index_dir}. Run indexing first.")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Use embedding model from metadata if not specified
        if embedding_model is None:
            embedding_model = self.metadata.get('embedding_model')

        # Initialize components
        self.embedder = CodeEmbedder(model_name=embedding_model, device=device)
        self.vector_store = TorchTalkVectorStore(
            persist_dir=str(self.index_dir / "chroma_db"),
            collection_name=collection_name
        )

        print(f" VectorRetriever initialized")
        print(f"  Index: {index_dir}")
        print(f"  Documents: {self.vector_store.count()}")
        print(f"  Model: {embedding_model}")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code chunks for a query.

        Args:
            query: Natural language or code query
            top_k: Number of results to return
            filters: Metadata filters (e.g., {'type': 'function', 'file': 'foo.py'})
            score_threshold: Minimum similarity score (0-1, None for no threshold)

        Returns:
            List of results, each with:
                - content: Code chunk content
                - metadata: Chunk metadata (file, type, name, etc.)
                - score: Similarity score (0-1, higher is better)
                - distance: Raw distance from vector store
        """
        # Generate query embedding
        query_embedding = self.embedder.get_query_embedding(query)

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters
        )

        # Format results with scores
        formatted_results = []
        for doc, meta, dist in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            # Convert distance to similarity score (cosine distance -> similarity)
            # Chroma returns distances in [0, 2], convert to similarity [0, 1]
            score = 1.0 - (dist / 2.0)

            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue

            formatted_results.append({
                'content': doc,
                'metadata': meta,
                'score': score,
                'distance': dist
            })

        return formatted_results

    def retrieve_by_file(
        self,
        query: str,
        file_path: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve code chunks from a specific file.

        Args:
            query: Query text
            file_path: Path to the file (relative to repo root)
            top_k: Number of results

        Returns:
            List of results from the specified file
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={'file': file_path}
        )

    def retrieve_by_type(
        self,
        query: str,
        code_type: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve code chunks of a specific type.

        Args:
            query: Query text
            code_type: Type of code ('function', 'class', 'method', 'module_level')
            top_k: Number of results

        Returns:
            List of results of the specified type
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={'type': code_type}
        )

    def retrieve_functions(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve functions matching the query"""
        return self.retrieve_by_type(query, 'function', top_k)

    def retrieve_classes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve classes matching the query"""
        return self.retrieve_by_type(query, 'class', top_k)

    def retrieve_methods(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve methods matching the query"""
        return self.retrieve_by_type(query, 'method', top_k)

    def get_context_for_symbol(
        self,
        symbol_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get code context for a specific symbol (function/class name).

        Args:
            symbol_name: Name of the symbol to find
            top_k: Number of results

        Returns:
            List of code chunks related to the symbol
        """
        # Search by symbol name directly
        return self.retrieve(
            query=f"definition of {symbol_name}",
            top_k=top_k
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'index_dir': str(self.index_dir),
            'total_documents': self.vector_store.count(),
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.get_embedding_dimension(),
            **self.metadata
        }


