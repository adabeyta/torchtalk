#!/usr/bin/env python3
"""
vLLM server launcher with preflight checks.

Environment variables (defaults):
  MODEL_NAME: meta-llama/llama-4-maverick
  MAX_MODEL_LEN: 1000000
  PORT: 8000
  HOST: 0.0.0.0
  GPU_MEMORY_UTIL: 0.9
  CUDA_VISIBLE_DEVICES: 0
  TENSOR_PARALLEL_SIZE: 1
  VLLM_ATTENTION_BACKEND: (optional - FlashInfer/Triton)
  SERVED_MODEL_NAME: (optional - alias for model)
  VLLM_LOG_LEVEL: (optional - INFO/DEBUG/WARNING)

CLI args override envs for ad-hoc runs:
  --model, --max-len, --port, --host, --gpu-util, --tp
"""

import os
import socket
import subprocess
import sys
import shutil
import argparse


def can_bind(host: str, port: int) -> bool:
    """Check if we can bind to host:port (works for 0.0.0.0)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    # Load .env if present (optional, no new deps)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # argparse for ad-hoc runs; envs remain default
    p = argparse.ArgumentParser(description="Launch vLLM server with preflight checks")
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "meta-llama/llama-4-maverick"),
                   help="Model to serve")
    p.add_argument("--max-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "1000000")),
                   help="Max context length")
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")),
                   help="Server port")
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                   help="Server host")
    p.add_argument("--gpu-util", type=float, default=float(os.getenv("GPU_MEMORY_UTIL", "0.9")),
                   help="GPU memory utilization (0-1)")
    p.add_argument("--tp", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
                   help="Tensor parallel size (number of GPUs)")
    args = p.parse_args()

    # Passthrough envs
    cuda = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    attn = os.getenv("VLLM_ATTENTION_BACKEND", "")
    served_name = os.getenv("SERVED_MODEL_NAME", "")
    log_level = os.getenv("VLLM_LOG_LEVEL", "")

    # Basic validation
    if not (0.0 < args.gpu_util <= 1.0):
        print("ERROR: GPU_MEMORY_UTIL must be in range (0, 1].", file=sys.stderr)
        sys.exit(2)
    if not (1 <= args.port <= 65535):
        print("ERROR: PORT must be in range 1..65535.", file=sys.stderr)
        sys.exit(2)
    if args.max_len < 1:
        print("ERROR: MAX_MODEL_LEN must be positive.", file=sys.stderr)
        sys.exit(2)

    # Tool presence check
    if not shutil.which("vllm"):
        print("ERROR: 'vllm' not found in PATH.", file=sys.stderr)
        sys.exit(127)

    # vLLM version check
    try:
        import vllm
        print(f"[vLLM] version: {vllm.__version__}")
    except Exception:
        print("[vLLM] version: <unknown> (import failed)", file=sys.stderr)

    # Port bind check
    if not can_bind(args.host, args.port):
        print(f"ERROR: Cannot bind to {args.host}:{args.port} (in use or insufficient perms).",
              file=sys.stderr)
        sys.exit(98)

    # GPU visibility hint (optional warning)
    if subprocess.call(["bash", "-lc", "nvidia-smi >/dev/null 2>&1"]) != 0:
        print("WARN: nvidia-smi not found or no GPU visible", file=sys.stderr)

    # Build vLLM command
    cmd = [
        "vllm", "serve", args.model,
        "--max-model-len", str(args.max_len),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--gpu-memory-utilization", str(args.gpu_util),
        "--tensor-parallel-size", str(args.tp),
        "--port", str(args.port),
        "--host", args.host,
    ]
    if served_name:
        cmd += ["--served-model-name", served_name]

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda
    if attn:
        env["VLLM_ATTENTION_BACKEND"] = attn
    if log_level:
        env["VLLM_LOG_LEVEL"] = log_level

    print("[vLLM] Launching with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Max context length: {args.max_len}")
    print(f"  Port: {args.host}:{args.port}")
    print(f"  GPU memory utilization: {args.gpu_util}")
    print(f"  Tensor parallel size: {args.tp}")
    print(f"  CUDA devices: {cuda}")
    if attn:
        print(f"  Attention backend: {attn}")
    if served_name:
        print(f"  Served model name: {served_name}")
    if log_level:
        print(f"  Log level: {log_level}")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    # Execute vLLM server (replaces this process)
    try:
        os.execvpe(cmd[0], cmd, env)
    except Exception as e:
        print(f"ERROR: failed to exec vLLM: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
