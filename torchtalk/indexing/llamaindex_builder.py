#!/usr/bin/env python3
"""
LlamaIndex-based code indexing pipeline for TorchTalk.

Orchestrates:
- Code analysis (RepoAnalyzer)
- Code chunking (CodeChunker)
- Embedding generation (CodeEmbedder)
- Vector storage (TorchTalkVectorStore)
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm

from torchtalk.analysis.repo_analyzer import RepoAnalyzer
from torchtalk.analysis.chunker import CodeChunker, CodeChunk
from torchtalk.analysis.binding_detector import BindingDetector  # Phase 6
from torchtalk.indexing.embedder import CodeEmbedder
from torchtalk.indexing.vector_store import TorchTalkVectorStore


class LlamaIndexBuilder:
    """
    Build a searchable index of a codebase.

    Pipeline:
    1. Analyze repository structure and relationships (RepoAnalyzer)
    2. Chunk code files into meaningful pieces (CodeChunker)
    3. Generate embeddings for each chunk (CodeEmbedder)
    4. Store in vector database (TorchTalkVectorStore)
    """

    def __init__(
        self,
        repo_path: str,
        persist_dir: str = "./torchtalk_index",
        collection_name: str = "torchtalk_code",
        max_chunk_chars: int = 2000,
        min_chunk_chars: int = 100,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None
    ):
        """
        Initialize the index builder.

        Args:
            repo_path: Path to the repository to index
            persist_dir: Directory to store the index
            collection_name: Name for the vector store collection
            max_chunk_chars: Maximum characters per chunk
            min_chunk_chars: Minimum characters per chunk
            embedding_model: HuggingFace model name for embeddings
            device: Device for embeddings ('cpu' or 'cuda', auto-detect if None)
        """
        self.repo_path = Path(repo_path)
        self.persist_dir = Path(persist_dir)

        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        print("Initializing indexing pipeline...")
        self.analyzer = RepoAnalyzer(str(self.repo_path))

        # Multi-language chunkers (Phase 6)
        self.chunkers = {
            'python': CodeChunker(max_chars=max_chunk_chars, min_chars=min_chunk_chars, language='python'),
            'cpp': CodeChunker(max_chars=max_chunk_chars, min_chars=min_chunk_chars, language='cpp'),
            'cuda': CodeChunker(max_chars=max_chunk_chars, min_chars=min_chunk_chars, language='cuda'),
        }

        # Binding detector for cross-language connections (Phase 6)
        self.binding_detector = BindingDetector()

        self.embedder = CodeEmbedder(model_name=embedding_model, device=device)
        self.vector_store = TorchTalkVectorStore(
            persist_dir=str(self.persist_dir / "chroma_db"),
            collection_name=collection_name
        )

        # Store metadata
        self.metadata_path = self.persist_dir / "index_metadata.json"
        self.graph_path = self.persist_dir / "code_graph.json"
        self.bindings_path = self.persist_dir / "bindings.json"

    def build_index(
        self,
        file_extensions: List[str] = [".py"],
        max_files: Optional[int] = None,
        rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build the full index for the repository.

        Args:
            file_extensions: File extensions to index (e.g., ['.py', '.cpp', '.cu'])
            max_files: Maximum number of files to index (for testing)
            rebuild: If True, clear existing index before building

        Returns:
            Dictionary with indexing statistics
        """
        if rebuild:
            print("Rebuilding index (clearing existing data)...")
            self.vector_store.reset()

        print(f"\n{'='*60}")
        print("PHASE 1: Analyzing repository structure")
        print(f"{'='*60}")

        # Run analysis
        analysis_results = self.analyzer.analyze_repository()

        # Save graphs separately using pickle
        import pickle
        with open(self.persist_dir / 'import_graph.gpickle', 'wb') as f:
            pickle.dump(self.analyzer.import_graph, f)
        with open(self.persist_dir / 'call_graph.gpickle', 'wb') as f:
            pickle.dump(self.analyzer.call_graph, f)
        with open(self.persist_dir / 'inheritance_graph.gpickle', 'wb') as f:
            pickle.dump(self.analyzer.inheritance_graph, f)

        # Save lightweight metadata in JSON (just file info we need for chunking)
        files_metadata = {}
        for path, module_info in analysis_results.get('modules', {}).items():
            files_metadata[path] = {
                'imports': module_info.imports if hasattr(module_info, 'imports') else [],
                'symbols_defined': list(module_info.symbols_defined) if hasattr(module_info, 'symbols_defined') else []
            }

        lightweight_metadata = {
            'files': files_metadata,
            'statistics': analysis_results.get('statistics', {})
        }

        with open(self.graph_path, 'w') as f:
            json.dump(lightweight_metadata, f, indent=2)

        print(f" Saved NetworkX graphs to {self.persist_dir}/*.gpickle")
        print(f" Saved metadata to {self.graph_path}")

        # Phase 6: Detect cross-language bindings (pybind11)
        print(f"\n{'='*60}")
        print("PHASE 1.5: Detecting cross-language bindings")
        print(f"{'='*60}")

        # Check if we're indexing C++/CUDA files
        has_cpp = any(ext in ['.cpp', '.cc', '.cxx', '.cu'] for ext in file_extensions)

        if has_cpp:
            binding_graph = self.binding_detector.detect_bindings_in_directory(str(self.repo_path))
            print(f" Found {len(binding_graph.bindings)} bindings across {len(binding_graph.modules)} modules")

            # Save bindings to JSON
            bindings_data = {
                'bindings': [
                    {
                        'python_name': b.python_name,
                        'cpp_name': b.cpp_name,
                        'type': b.binding_type,
                        'file': b.file_path,
                        'line': b.line_number,
                        'module': b.module_name
                    }
                    for b in binding_graph.bindings
                ],
                'python_to_cpp': binding_graph.python_to_cpp,
                'cpp_to_python': binding_graph.cpp_to_python
            }

            with open(self.bindings_path, 'w') as f:
                json.dump(bindings_data, f, indent=2)
            print(f" Saved bindings to {self.bindings_path}")
        else:
            print(" No C++/CUDA files to scan for bindings")

        print(f"\n{'='*60}")
        print("PHASE 2: Chunking and embedding code")
        print(f"{'='*60}")

        # Get all Python files
        all_files = []
        for ext in file_extensions:
            all_files.extend(self.repo_path.rglob(f"*{ext}"))

        # Filter to max_files if specified
        if max_files:
            all_files = all_files[:max_files]

        print(f"Found {len(all_files)} files to index")

        # Process files
        all_chunks: List[CodeChunk] = []
        file_stats = []

        for file_path in tqdm(all_files, desc="Chunking files"):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Get relative path from repo root
                rel_path = file_path.relative_to(self.repo_path)

                # Detect language from extension (Phase 6 multi-language support)
                ext = file_path.suffix
                if ext == '.py':
                    language = 'python'
                elif ext in ['.cpp', '.cc', '.cxx', '.h', '.hpp']:
                    language = 'cpp'
                elif ext == '.cu':
                    language = 'cuda'
                else:
                    print(f"\n Unknown file extension {ext} for {rel_path}, skipping")
                    continue

                # Get appropriate chunker
                chunker = self.chunkers.get(language)
                if not chunker:
                    print(f"\n No chunker for language {language}, skipping")
                    continue

                # Extract base metadata from analysis
                file_info = analysis_results.get('files', {}).get(str(rel_path), {})
                base_metadata = {
                    'file': str(rel_path),
                    'imports': file_info.get('imports', []),
                    'language': language
                }

                # Chunk the file
                chunks = chunker.chunk_file(str(rel_path), content, base_metadata)
                all_chunks.extend(chunks)

                file_stats.append({
                    'file': str(rel_path),
                    'chunks': len(chunks),
                    'size': len(content),
                    'language': language
                })

            except Exception as e:
                print(f"\n Error processing {file_path}: {e}")
                continue

        print(f"\n Generated {len(all_chunks)} chunks from {len(all_files)} files")

        # Generate embeddings in batches
        print("\nGenerating embeddings...")
        chunk_contents = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(chunk_contents, batch_size=32)
        print(f" Generated {len(embeddings)} embeddings")

        print(f"\n{'='*60}")
        print("PHASE 3: Storing in vector database")
        print(f"{'='*60}")

        # Prepare data for vector store
        documents = chunk_contents

        # Clean metadata for Chroma (no lists/dicts/None, only str/int/float/bool)
        metadatas = []
        for chunk in all_chunks:
            clean_meta = {}
            for key, value in chunk.metadata.items():
                if value is None:
                    # Skip None values - Chroma doesn't accept them
                    continue
                elif isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    if value:
                        clean_meta[key] = ', '.join(str(v) for v in value)
                # Skip dicts and other complex types
            metadatas.append(clean_meta)

        # Generate IDs
        ids = []
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"{chunk.metadata['file']}:{chunk.start_line}-{chunk.end_line}"
            ids.append(chunk_id)

        # Add to vector store
        self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Save metadata
        metadata = {
            'repo_path': str(self.repo_path),
            'total_files': len(all_files),
            'total_chunks': len(all_chunks),
            'file_extensions': file_extensions,
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.get_embedding_dimension(),
            'chunk_stats': {
                'max_chars': self.chunkers['python'].max_chars,
                'min_chars': self.chunkers['python'].min_chars,
            },
            'file_stats': file_stats
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*60}")
        print("INDEX BUILD COMPLETE")
        print(f"{'='*60}")
        print(f" Repository: {self.repo_path}")
        print(f" Files indexed: {len(all_files)}")
        print(f" Chunks created: {len(all_chunks)}")
        print(f" Embeddings generated: {len(embeddings)}")
        print(f" Index saved to: {self.persist_dir}")
        print(f" Metadata: {self.metadata_path}")
        print(f" Graph: {self.graph_path}")

        return metadata

    def load_existing_index(self) -> Dict[str, Any]:
        """
        Load an existing index from disk.

        Returns:
            Index metadata dictionary
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"No existing index found at {self.persist_dir}. "
                "Run build_index() first."
            )

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f" Loaded existing index from {self.persist_dir}")
        print(f"  Files: {metadata['total_files']}")
        print(f"  Chunks: {metadata['total_chunks']}")
        print(f"  Model: {metadata['embedding_model']}")

        return metadata

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the index with a text query.

        Args:
            query_text: Natural language or code query
            n_results: Number of results to return
            filters: Metadata filters (e.g., {'type': 'function'})

        Returns:
            Query results with ids, documents, metadatas, distances
        """
        # Generate query embedding
        query_embedding = self.embedder.get_query_embedding(query_text)

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=filters
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        vector_stats = self.vector_store.get_stats()

        return {
            **metadata,
            'vector_store': vector_stats
        }
