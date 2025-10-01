#!/usr/bin/env python3
"""
Code embedder using UniXcoder for semantic code understanding.
"""

from typing import List, Union
import numpy as np


class CodeEmbedder:
    """
    Generate embeddings for code using sentence-transformers.

    Default model: all-mpnet-base-v2 (general-purpose, works well for code)
    Note: For best code-specific results, can use microsoft/codebert-base
    after upgrading to torch 2.6+
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        """
        Initialize the code embedder.

        Args:
            model_name: HuggingFace model name (default: UniXcoder)
            device: Device to run on ('cpu' or 'cuda', auto-detects if None)
        """
        self.model_name = model_name
        # Auto-detect device if not specified
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.tokenizer = None
        self._lazy_load()

    def _lazy_load(self):
        """Lazy load the model (only when first needed)"""
        if self.model is not None:
            return

        print(f"Loading embedding model: {self.model_name}")
        print("(This will download ~500MB on first run)")

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Use sentence-transformers which handles safetensors better
            self.model = SentenceTransformer(self.model_name, device=self.device)

            print(f" Model loaded: {self.model_name}")
            if self.device == "cuda":
                print(" Using GPU acceleration (H200)")
            else:
                print(" Using CPU")

        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Code or text to embed

        Returns:
            Embedding vector as numpy array (shape: [embedding_dim])
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of code/text strings
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        self._lazy_load()

        # sentence-transformers handles batching internally
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return list(embeddings)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Embed a query (same as embed_text, but semantically clearer for search).

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.embed_text(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        self._lazy_load()
        return self.model.get_sentence_embedding_dimension()  # 768 for CodeBERT-base

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clip to [0, 1] range
        return float(np.clip(similarity, 0, 1))


