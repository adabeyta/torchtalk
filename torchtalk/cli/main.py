#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime
from torchtalk.core.config import get_config, set_config
from torchtalk.indexing.llamaindex_builder import LlamaIndexBuilder


def get_index_dir(repo_path: str) -> Path:
    """Get persistent index directory for a repository"""
    # Create hash of absolute repo path for uniqueness
    repo_abs = str(Path(repo_path).resolve())
    repo_hash = hashlib.sha256(repo_abs.encode()).hexdigest()[:16]

    # Store in ~/.torchtalk/indexes/<hash>
    index_base = Path.home() / ".torchtalk" / "indexes"
    index_dir = index_base / repo_hash

    return index_dir


def is_index_stale(index_dir: Path, repo_path: Path, max_age_days: int = 7) -> bool:
    """Check if index needs rebuilding"""
    metadata_file = index_dir / "index_metadata.json"

    if not metadata_file.exists():
        return True

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Check if any code files were modified after indexing
        index_time = datetime.fromisoformat(metadata.get('created_at', '2000-01-01'))

        for ext in [".py", ".cpp", ".cc", ".cxx", ".cu", ".h", ".hpp"]:
            for code_file in repo_path.rglob(f"*{ext}"):
                if code_file.stat().st_mtime > index_time.timestamp():
                    return True

        return False
    except Exception:
        return True


def build_index(repo_path: str, rebuild: bool = False, max_files: int = None) -> Path:
    """Build or rebuild the index for a repository"""
    repo_path = Path(repo_path).resolve()
    index_dir = get_index_dir(str(repo_path))

    # Check if index exists and is fresh
    if not rebuild and index_dir.exists():
        if not is_index_stale(index_dir, repo_path):
            print(f" Using existing index at {index_dir}")
            return index_dir
        else:
            print(f" Index is stale, rebuilding...")

    print(f"\n{'='*60}")
    print(f"Building index for: {repo_path}")
    print(f"Index location: {index_dir}")
    print(f"{'='*60}\n")

    # Count files first (multi-language support)
    file_extensions = [".py", ".cpp", ".cc", ".cxx", ".cu", ".h", ".hpp"]
    all_code_files = []
    for ext in file_extensions:
        all_code_files.extend(repo_path.rglob(f"*{ext}"))

    total_files = len(all_code_files) if max_files is None else min(len(all_code_files), max_files)

    # Count by type
    py_count = sum(1 for f in all_code_files if f.suffix == '.py')
    cpp_count = sum(1 for f in all_code_files if f.suffix in ['.cpp', '.cc', '.cxx', '.h', '.hpp'])
    cuda_count = sum(1 for f in all_code_files if f.suffix == '.cu')

    print(f"Found {len(all_code_files)} code files:")
    print(f"  Python: {py_count}")
    print(f"  C++: {cpp_count}")
    print(f"  CUDA: {cuda_count}")

    if max_files:
        print(f"Indexing first {total_files} files (--max-files={max_files})")

    print(f"Estimated time: ~{total_files // 10} seconds on GPU\n")

    # Build index
    builder = LlamaIndexBuilder(
        repo_path=str(repo_path),
        persist_dir=str(index_dir),
        collection_name="torchtalk_code"
    )

    metadata = builder.build_index(
        file_extensions=file_extensions,
        max_files=max_files,
        rebuild=rebuild
    )

    # Add timestamp
    metadata['created_at'] = datetime.now().isoformat()
    metadata['repo_path_abs'] = str(repo_path)

    with open(index_dir / "index_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Index built successfully")
    print(f"  Location: {index_dir}")
    print(f"  Files: {metadata['total_files']}")
    print(f"  Chunks: {metadata['total_chunks']}")

    return index_dir


def update_config_for_repo(repo_path: str):
    config = get_config()
    config.repo_path = repo_path

    # Ensure artifacts directory exists
    os.makedirs(config.artifacts_dir, exist_ok=True)

    # Save updated config
    config.save()
    set_config(config)

    print(f"Configuration updated for repository: {repo_path}")
    print(f"   Artifacts will be saved to: {config.artifacts_dir}")


def start_dynamic_services():
    config = get_config()

    print(f"Starting dynamic TorchTalk services...")
    print(f"Model: {config.model_name}")

    # Set environment variables for services
    env_vars = config.to_env_vars()
    env = os.environ.copy()
    env.update(env_vars)

    processes = []

    try:
        print("   Starting vLLM server...")
        vllm_cmd = [
            "vllm", "serve", config.model_name,
            "--port", str(config.vllm_port),
            "--host", "0.0.0.0",
            "--max-num-batched-tokens", "131072"  # TODO: Set to match typical model max_model_len, can modify this later.
        ]

        # Add tensor parallelism if specified
        if config.tensor_parallel_size > 1:
            vllm_cmd.extend(["--tensor-parallel-size", str(config.tensor_parallel_size)])
            print(f"   Using tensor parallelism with {config.tensor_parallel_size} GPUs")

        vllm_proc = subprocess.Popen(vllm_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        processes.append(("vLLM", vllm_proc))

        # Wait for vLLM to be ready by monitoring its output
        import time
        print("   Waiting for vLLM server to be ready...")
        start_time = time.time()
        timeout = 300  # 5 minutes timeout

        while time.time() - start_time < timeout:
            line = vllm_proc.stdout.readline()
            if line:
                # Check for the ready message
                if "Uvicorn running on" in line or "Application startup complete" in line:
                    print("   vLLM server is ready!")
                    break
            # Check if process died
            if vllm_proc.poll() is not None:
                print("   Error: vLLM process terminated unexpectedly")
                raise RuntimeError("vLLM failed to start")
            time.sleep(0.1)
        else:
            print("   Warning: Timeout waiting for vLLM, proceeding anyway...")

        time.sleep(2)  # Extra buffer for safety

        # Start FastAPI backend
        print("   Starting FastAPI backend...")
        fastapi_proc = subprocess.Popen(
            [sys.executable, "-m", "torchtalk.web.api"],
            env=env
        )
        processes.append(("FastAPI", fastapi_proc))

        # Start Gradio UI
        print("   Starting Gradio UI...")
        gradio_proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "torchtalk.web.ui"],  # -u for unbuffered output
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(("Gradio", gradio_proc))

        # Wait for Gradio to fully initialize and capture share link
        print("   Waiting for Gradio to start...")
        shared_url = None
        local_url = None
        start_time = time.time()
        timeout = 60  # Increase timeout for share link generation

        while time.time() - start_time < timeout:
            line = gradio_proc.stdout.readline()
            if line:
                print(f"   [Gradio] {line.rstrip()}")  # Debug: show all output
                # Gradio prints: "Running on local URL:  http://..."
                if "Running on local URL:" in line:
                    local_url = line.split("Running on local URL:")[1].strip()
                # Gradio prints: "Running on public URL: https://..."
                if "Running on public URL:" in line or "public URL:" in line:
                    shared_url = line.split("public URL:")[1].strip()
                    break
                # Check if Gradio is ready (even without share link yet)
                if "Running on local URL:" in line:
                    # Give it a bit more time for share link
                    time.sleep(5)
            if gradio_proc.poll() is not None:
                print("   Warning: Gradio process terminated unexpectedly")
                break
            time.sleep(0.1)

        print(f"\n{'='*70}")
        print(f"  TorchTalk is ready!")
        print(f"{'='*70}")
        if local_url:
            print(f"\n   LOCAL:  {local_url}")
        else:
            print(f"\n   LOCAL:  http://localhost:{config.gradio_port}")
        if shared_url:
            print(f"   SHARED: {shared_url}")
        else:
            print(f"   SHARED: Generating share link...")
            print(f"           (May take 10-30 seconds, check below for public URL)")
        print(f"\n   FastAPI Backend: http://localhost:{config.fastapi_port}")
        print(f"   vLLM Server: http://localhost:{config.vllm_port}")
        print(f"\n{'='*70}")
        print(f"  Press Ctrl+C to stop all services")
        print(f"{'='*70}\n")

        # Continue monitoring Gradio output for share link and relay to user
        import threading

        def relay_gradio_output():
            """Relay Gradio stdout to main console to show share link"""
            try:
                for line in iter(gradio_proc.stdout.readline, ''):
                    if not line:
                        break
                    # Print share link prominently
                    if "public URL:" in line or "Running on public URL:" in line:
                        share_url = line.split("public URL:")[1].strip() if "public URL:" in line else line.strip()
                        print(f"\n{'='*70}")
                        print(f"  GRADIO SHARE LINK READY:")
                        print(f"  {share_url}")
                        print(f"{'='*70}\n")
                    # Also print any other Gradio messages
                    elif line.strip() and not line.startswith("INFO:"):
                        print(f"  [Gradio] {line.strip()}")
            except Exception:
                pass

        relay_thread = threading.Thread(target=relay_gradio_output, daemon=True)
        relay_thread.start()

        # Wait for all processes
        for name, proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nStopping services...")
        for name, proc in processes:
            print(f"   Stopping {name}...")
            proc.terminate()
            proc.wait()
        print("All services stopped")


def show_index_status(repo_path: str):
    """Show status of index for a repository"""
    repo_path = Path(repo_path).resolve()
    index_dir = get_index_dir(str(repo_path))

    print(f"Repository: {repo_path}")
    print(f"Index directory: {index_dir}")

    if not index_dir.exists():
        print("\n No index found")
        print("  Run 'torchtalk index <repo>' or 'torchtalk chat <repo>' to build index")
        return

    metadata_file = index_dir / "index_metadata.json"
    if not metadata_file.exists():
        print("\n Index directory exists but metadata is missing")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"\n Index exists")
    print(f"  Created: {metadata.get('created_at', 'unknown')}")
    print(f"  Files indexed: {metadata.get('total_files', 0)}")
    print(f"  Chunks: {metadata.get('total_chunks', 0)}")
    print(f"  Embedding model: {metadata.get('embedding_model', 'unknown')}")
    print(f"  Embedding dim: {metadata.get('embedding_dimension', 0)}")

    # Check staleness
    is_stale = is_index_stale(index_dir, repo_path)
    if is_stale:
        print(f"\n Index may be stale (files modified since indexing)")
        print(f"  Run 'torchtalk index <repo> --rebuild' to update")
    else:
        print(f"\n Index is fresh")


def main():
    parser = argparse.ArgumentParser(
        description='TorchTalk v2.0 - Advanced code understanding with semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start chat (auto-indexes if needed)
  torchtalk chat /path/to/pytorch

  # Build index explicitly
  torchtalk index /path/to/pytorch

  # Rebuild index
  torchtalk index /path/to/pytorch --rebuild

  # Check index status
  torchtalk status /path/to/pytorch

  # Start with specific model
  torchtalk chat /path/to/pytorch --model meta-llama/Llama-3.1-70B-Instruct

  # Start with 70B model using 8 GPUs for tensor parallelism
  torchtalk chat /path/to/pytorch --model meta-llama/Llama-3.1-70B-Instruct --tp 8
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Index command (explicit indexing)
    index_parser = subparsers.add_parser('index', help='Build search index for a repository')
    index_parser.add_argument('repo_path', help='Path to repository')
    index_parser.add_argument('--rebuild', action='store_true',
                             help='Force rebuild even if index exists')
    index_parser.add_argument('--max-files', type=int, default=None,
                             help='Maximum number of files to index (for testing)')

    # Chat command (smart: auto-index + chat)
    chat_parser = subparsers.add_parser('chat', help='Start chatbot (auto-indexes if needed)')
    chat_parser.add_argument('repo_path', help='Path to repository')
    chat_parser.add_argument('--rebuild', action='store_true',
                            help='Force rebuild index before chatting')
    chat_parser.add_argument('--no-index', action='store_true',
                            help='Skip indexing, use existing index only')
    chat_parser.add_argument('--max-files', type=int, default=None,
                            help='Maximum number of files to index (for testing)')
    chat_parser.add_argument('--model', help='Model to use (e.g., meta-llama/Llama-3.1-70B-Instruct)')
    chat_parser.add_argument('--tp', type=int, default=1,
                            help='Tensor parallel size (number of GPUs, default: 1)')
    chat_parser.add_argument('--vllm-endpoint',
                            help='Custom vLLM endpoint (e.g., http://gpu-server:8000/v1/chat/completions)')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show index status for a repository')
    status_parser.add_argument('repo_path', help='Path to repository')

    # Legacy start command (kept for compatibility)
    start_parser = subparsers.add_parser('start', help='[DEPRECATED] Use "chat" instead')
    start_parser.add_argument('--repo', required=True, help='Repository path')
    start_parser.add_argument('--model', help='Model to use')
    start_parser.add_argument('--tp', type=int, default=1,
                             help='Tensor parallel size (number of GPUs, default: 1)')
    start_parser.add_argument('--vllm-endpoint', help='Custom vLLM endpoint')
    start_parser.add_argument('--force-analyze', action='store_true', help='Force re-analysis')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle commands
    if args.command == 'index':
        # Explicit indexing
        build_index(args.repo_path, rebuild=args.rebuild, max_files=args.max_files)

    elif args.command == 'status':
        # Show index status
        show_index_status(args.repo_path)

    elif args.command == 'chat':
        # Smart chat: auto-index if needed
        repo_path = Path(args.repo_path).resolve()
        index_dir = get_index_dir(str(repo_path))

        # Check/build index
        if args.no_index:
            if not index_dir.exists():
                print("Error: --no-index specified but no index exists")
                print(f"Run 'torchtalk index {args.repo_path}' first")
                sys.exit(1)
            print(f"Using existing index at {index_dir}")
        else:
            index_dir = build_index(str(repo_path), rebuild=args.rebuild, max_files=args.max_files)

        # Update config
        update_config_for_repo(str(repo_path))
        config = get_config()
        config.index_dir = str(index_dir)  # Store index location

        if args.model:
            config.model_name = args.model
            print(f"Using model: {args.model}")

        if hasattr(args, 'tp') and args.tp:
            config.tensor_parallel_size = args.tp
            if args.tp > 1:
                print(f"Using tensor parallelism with {args.tp} GPUs")

        # Save updated config
        config.save()

        if hasattr(args, 'vllm_endpoint') and args.vllm_endpoint:
            config.vllm_endpoint = args.vllm_endpoint
            print(f"Using vLLM endpoint: {args.vllm_endpoint}")

        config.save()
        set_config(config)

        # Start services
        print(f"\nStarting TorchTalk chat interface...")
        start_dynamic_services()

    elif args.command == 'start':
        # Legacy command - warn and redirect
        print(" Warning: 'start' command is deprecated, use 'chat' instead")
        print(f"  Example: torchtalk chat {args.repo}\n")

        update_config_for_repo(args.repo)
        config = get_config()

        if args.model:
            config.model_name = args.model

        if hasattr(args, 'tp') and args.tp:
            config.tensor_parallel_size = args.tp

        if hasattr(args, 'vllm_endpoint') and args.vllm_endpoint:
            config.vllm_endpoint = args.vllm_endpoint

        config.save()
        set_config(config)

        # For legacy, just start services without v2.0 indexing
        start_dynamic_services()


if __name__ == '__main__':
    main()