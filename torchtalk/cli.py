#!/usr/bin/env python3
"""
TorchTalk CLI - Unified command-line interface.

Usage:
    torchtalk index <repo_path> [--output DIR]
    torchtalk chat [--index PATH] [--vllm-server URL] [--port PORT]
    torchtalk serve-vllm [options]
"""

import argparse
import logging
import sys
import subprocess
import time
import threading
import atexit
import signal
import os
from pathlib import Path
from urllib.parse import urlparse

# Resolve paths relative to package, not cwd
PKG_ROOT = Path(__file__).resolve().parent
VLLM_LAUNCHER = (PKG_ROOT.parent / "scripts" / "start_vllm_server.py").resolve()

log = logging.getLogger(__name__)


def cmd_index(args):
    """Build index for a repository"""
    from torchtalk.indexing import GraphEnhancedIndexer

    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        log.error(f"Repository not found: {repo_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("./index")

    log.info(f"Building index for {repo_path}")
    log.info(f"Output directory: {output_dir}")
    if args.property_graph:
        log.info("Using PropertyGraphIndex with graph traversal (recommended for cross-language tracing)")
    else:
        log.info("Using VectorStoreIndex (add --property-graph for graph traversal)")

    indexer = GraphEnhancedIndexer(repo_path=str(repo_path))
    indexer.build_index(
        persist_dir=str(output_dir),
        use_property_graph=args.property_graph,
    )

    log.info(f"Index built successfully at {output_dir}")
    return 0


def _start_or_attach_vllm(vllm_url: str, args) -> tuple:
    """
    Start vLLM server if needed, or attach to existing instance.

    Args:
        vllm_url: Target vLLM server URL
        args: CLI args containing model, tp, etc.

    Returns:
        tuple: (vllm_url, process_handle or None, model_name)
    """
    # Import requests lazily
    try:
        import requests
    except ImportError:
        log.error("Missing dependency: requests (pip install requests)")
        sys.exit(1)

    # Check if vLLM is already running
    vllm_running = False
    try:
        response = requests.get(f"{vllm_url}/health", timeout=2)
        if response.status_code == 200:
            vllm_running = True
            log.info(f"vLLM server already running at {vllm_url}")
    except Exception:
        pass

    if vllm_running:
        return vllm_url, None, args.model

    log.info("Starting vLLM server...")
    log.info("This may take 1-3 minutes for model loading and compilation...")

    vllm_cmd = [
        sys.executable, str(VLLM_LAUNCHER),
        "--model", args.model,
        "--max-len", str(args.max_len),
        "--host", args.host,
        "--gpu-util", str(args.gpu_util),
        "--tp", str(args.tp),
    ]

    if args.attention_backend:
        vllm_cmd.extend(["--attention-backend", args.attention_backend])
    if args.served_model_name:
        vllm_cmd.extend(["--served-model-name", args.served_model_name])
    if args.vllm_log_level:
        vllm_cmd.extend(["--vllm-log-level", args.vllm_log_level])

    env = os.environ.copy()

    parsed = urlparse(vllm_url)
    port_from_url = parsed.port or (80 if parsed.scheme == "http" else 443)
    if parsed.hostname in ("127.0.0.1", "localhost"):
        vllm_cmd.extend(["--port", str(port_from_url)])

    try:
        vllm_process = subprocess.Popen(
            vllm_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream vLLM logs to console
        def _stream_logs(proc):
            if proc.stdout:
                for line in proc.stdout:
                    sys.stdout.write(f"[vLLM] {line}")
                    sys.stdout.flush()

        log_thread = threading.Thread(
            target=_stream_logs,
            args=(vllm_process,),
            daemon=True
        )
        log_thread.start()

        # Graceful shutdown handler
        def _kill_child():
            if vllm_process and vllm_process.poll() is None:
                try:
                    log.info("Terminating vLLM server...")
                    vllm_process.terminate()
                    vllm_process.wait(timeout=10)
                except Exception:
                    vllm_process.kill()

        atexit.register(_kill_child)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: sys.exit(0))

        # Wait for vLLM to be ready (poll health endpoint)
        log.info("Waiting for vLLM server to be ready...")
        max_wait = 600  # 600 seconds (torch.compile + large TP can take time on first run)
        step = 1
        waited = 0
        while waited < max_wait:
            # Check if vLLM crashed during startup
            if vllm_process.poll() is not None:
                log.error("vLLM exited during startup (see logs above).")
                sys.exit(1)

            try:
                response = requests.get(f"{vllm_url}/health", timeout=1)
                if response.status_code == 200:
                    log.info("vLLM server ready")
                    break
            except Exception:
                pass
            time.sleep(step)
            waited += step
            if waited % 10 == 0:
                log.info(f"  Still waiting... ({waited}s)")
        else:
            log.error(f"vLLM server failed to start within {max_wait} seconds")
            if vllm_process:
                vllm_process.terminate()
            sys.exit(1)

        return vllm_url, vllm_process, args.model

    except Exception as e:
        log.error(f"Failed to start vLLM server: {e}")
        sys.exit(1)


def cmd_chat(args):
    """Start chat interface (launches vLLM if needed + Gradio UI)"""
    # Determine index path
    if args.index:
        index_path = Path(args.index)
    else:
        index_path = Path("./index")

    if not index_path.exists():
        log.error(f"Index not found at {index_path}")
        log.error("Build index first: torchtalk index <repo_path>")
        sys.exit(1)

    vllm_url, vllm_process, model_name = _start_or_attach_vllm(args.vllm_server, args)

    # Share link warning
    if args.share:
        log.warning("\n" + "!"*60)
        log.warning("WARNING: --share creates a PUBLIC URL accessible by anyone!")
        log.warning("Do not share sensitive code or data.")
        log.warning("!"*60 + "\n")

    # Launch Gradio UI
    log.info("Starting Gradio UI...")

    # Import app dynamically to avoid early imports
    sys.path.insert(0, str(PKG_ROOT.parent))
    from app import create_app

    app = create_app(
        index_path=str(index_path),
        vllm_server=vllm_url,
        model_name=model_name,
        context_window=args.max_len,
    )

    # Launch with prevent_thread_lock to get server handle
    # In Gradio 5.x, launch() with prevent_thread_lock=True returns (app, local_url, share_url)
    launch_result = app.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        show_error=True,
        prevent_thread_lock=True,
    )

    # Handle both tuple and object return types for compatibility
    if isinstance(launch_result, tuple):
        # Gradio 5.x returns (app, local_url, share_url)
        gradio_app, local_url, share_url = launch_result
    else:
        # Older versions return server object
        gradio_app = launch_result
        local_url = f"http://localhost:{args.port}"
        share_url = getattr(launch_result, "share_url", None) if args.share else None

    # Print stable, logged URLs
    log.info(f"\n{'='*60}")
    log.info("TorchTalk is ready!")
    log.info(f"{'='*60}")
    log.info(f"Chat UI (local): {local_url or f'http://localhost:{args.port}'}")
    if args.share and share_url:
        log.info(f"Share URL: {share_url}")
    log.info(f"vLLM API: {vllm_url}")
    log.info(f"Index: {index_path}")
    log.info(f"{'='*60}\n")

    # Keep the main thread alive
    try:
        # Use the app's internal server to block
        if hasattr(gradio_app, 'block_thread'):
            gradio_app.block_thread()
        else:
            # Fallback: just keep the main thread alive
            while True:
                time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        log.info("\nInterrupted. Cleaning up...")

    return 0


def cmd_serve_vllm(args):
    """Start vLLM server only"""
    cmd = [
        sys.executable, str(VLLM_LAUNCHER),
        "--model", args.model,
        "--max-len", str(args.max_len),
        "--port", str(args.port),
        "--host", args.host,
        "--gpu-util", str(args.gpu_util),
        "--tp", str(args.tp),
    ]

    if args.attention_backend:
        cmd.extend(["--attention-backend", args.attention_backend])
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    if args.vllm_log_level:
        cmd.extend(["--vllm-log-level", args.vllm_log_level])

    log.info("Starting vLLM server...")
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="torchtalk",
        description="TorchTalk - PyTorch codebase chatbot with cross-language tracing"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index command
    parser_index = subparsers.add_parser("index", help="Build index for a repository")
    parser_index.add_argument("repo_path", help="Path to repository")
    parser_index.add_argument("--output", "-o", help="Output directory (default: ./index)")
    parser_index.add_argument(
        "--property-graph", "-g",
        action="store_true",
        help="Use PropertyGraphIndex with graph traversal (recommended for cross-language tracing)"
    )
    parser_index.set_defaults(func=cmd_index)

    parser_chat = subparsers.add_parser("chat", help="Start chat interface")
    parser_chat.add_argument("--index", "-i", help="Index path (default: ./index)")
    parser_chat.add_argument("--vllm-server", default="http://localhost:8000", help="vLLM server URL")
    parser_chat.add_argument("--model", default="meta-llama/llama-4-maverick", help="Model to serve")
    parser_chat.add_argument("--max-len", type=int, default=1000000, help="Max context length")
    parser_chat.add_argument("--host", default="0.0.0.0", help="vLLM server host")
    parser_chat.add_argument("--gpu-util", type=float, default=0.9, help="GPU memory utilization (0-1)")
    parser_chat.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser_chat.add_argument("--attention-backend", default="", help="Attention backend")
    parser_chat.add_argument("--served-model-name", default="", help="Served model name")
    parser_chat.add_argument("--vllm-log-level", default="", help="vLLM log level")
    parser_chat.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser_chat.add_argument("--share", action="store_true", help="Create public share link")
    parser_chat.set_defaults(func=cmd_chat)

    parser_vllm = subparsers.add_parser("serve-vllm", help="Start vLLM server only")
    parser_vllm.add_argument("--model", default="meta-llama/llama-4-maverick", help="Model name")
    parser_vllm.add_argument("--max-len", type=int, default=1000000, help="Max context length")
    parser_vllm.add_argument("--port", type=int, default=8000, help="Server port")
    parser_vllm.add_argument("--host", default="0.0.0.0", help="Server host")
    parser_vllm.add_argument("--gpu-util", type=float, default=0.9, help="GPU memory utilization (0-1)")
    parser_vllm.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser_vllm.add_argument("--attention-backend", default="", help="Attention backend")
    parser_vllm.add_argument("--served-model-name", default="", help="Served model name")
    parser_vllm.add_argument("--vllm-log-level", default="", help="vLLM log level")
    parser_vllm.set_defaults(func=cmd_serve_vllm)

    args = parser.parse_args()

    # Setup logging with colorized output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - \033[92m%(levelname)s\033[0m - %(message)s"
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
