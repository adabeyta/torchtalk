#!/usr/bin/env python3
"""
Vector store using Chroma for storing and retrieving code embeddings.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path


class TorchTalkVectorStore:
    """
    Wrapper around ChromaDB for storing and retrieving code embeddings.

    Features:
    - Persistent storage on disk
    - Metadata filtering
    - Similarity search
    - Batch operations
    """

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "torchtalk_code"):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to store the database
            collection_name: Name of the collection to use
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f" Vector store initialized: {self.persist_dir}")
        print(f"  Collection: {self.collection_name}")
        print(f"  Documents: {self.collection.count()}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of text content
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of unique IDs (auto-generated if not provided)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Convert numpy arrays to lists for Chroma
        embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embs = embeddings_list[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size] if metadatas else None

            self.collection.add(
                documents=batch_docs,
                embeddings=batch_embs,
                ids=batch_ids,
                metadatas=batch_meta
            )

        print(f" Added {len(documents)} documents to vector store")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "function"})
            where_document: Document content filter

        Returns:
            Dictionary with 'ids', 'documents', 'metadatas', 'distances'
        """
        # Convert to list if numpy array
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        # Flatten results (Chroma returns lists of lists)
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            Dictionary with 'ids', 'documents', 'metadatas', 'embeddings'
        """
        results = self.collection.get(ids=ids)
        return results

    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[np.ndarray]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update existing documents.

        Args:
            ids: IDs of documents to update
            documents: New document content (optional)
            embeddings: New embeddings (optional)
            metadatas: New metadata (optional)
        """
        update_dict = {'ids': ids}

        if documents:
            update_dict['documents'] = documents
        if embeddings:
            update_dict['embeddings'] = [
                emb.tolist() if isinstance(emb, np.ndarray) else emb
                for emb in embeddings
            ]
        if metadatas:
            update_dict['metadatas'] = metadatas

        self.collection.update(**update_dict)
        print(f" Updated {len(ids)} documents")

    def delete_documents(self, ids: List[str]):
        """Delete documents by their IDs"""
        self.collection.delete(ids=ids)
        print(f" Deleted {len(ids)} documents")

    def count(self) -> int:
        """Get total number of documents in the store"""
        return self.collection.count()

    def reset(self):
        """Delete all documents in the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(" Vector store reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'collection_name': self.collection_name,
            'persist_dir': str(self.persist_dir),
            'total_documents': self.count(),
            'metadata': self.collection.metadata
        }


