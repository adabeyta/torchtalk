#!/usr/bin/env python3
"""
vLLM server launcher with preflight checks.
"""

import os
import socket
import subprocess
import sys
import shutil
import argparse
import logging

log = logging.getLogger(__name__)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    p = argparse.ArgumentParser(description="Launch vLLM server with preflight checks")
    p.add_argument("--model", default="meta-llama/llama-4-maverick",
                   help="Model to serve")
    p.add_argument("--max-len", type=int, default=1000000,
                   help="Max context length")
    p.add_argument("--port", type=int, default=8000,
                   help="Server port")
    p.add_argument("--host", default="0.0.0.0",
                   help="Server host")
    p.add_argument("--gpu-util", type=float, default=0.9,
                   help="GPU memory utilization (0-1)")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size")
    p.add_argument("--attention-backend", default="",
                   help="Attention backend (FlashInfer/Triton)")
    p.add_argument("--served-model-name", default="",
                   help="Served model name alias")
    p.add_argument("--vllm-log-level", default="",
                   help="vLLM log level (INFO/DEBUG/WARNING)")
    args = p.parse_args()

    if not (0.0 < args.gpu_util <= 1.0):
        log.error("GPU_MEMORY_UTIL must be in range (0, 1].")
        sys.exit(2)
    if not (1 <= args.port <= 65535):
        log.error("PORT must be in range 1..65535.")
        sys.exit(2)
    if args.max_len < 1:
        log.error("MAX_MODEL_LEN must be positive.")
        sys.exit(2)

    if not shutil.which("vllm"):
        log.error("'vllm' not found in PATH.")
        sys.exit(127)

    try:
        import vllm
        log.info(f"[vLLM] version: {vllm.__version__}")
    except Exception:
        log.warning("[vLLM] version: <unknown> (import failed)")

    if not can_bind(args.host, args.port):
        log.error(f"Cannot bind to {args.host}:{args.port} (in use or insufficient perms).")
        sys.exit(98)

    if subprocess.call(["bash", "-lc", "nvidia-smi >/dev/null 2>&1"]) != 0:
        log.warning("nvidia-smi not found or no GPU visible")

    # Auto-compute CUDA_VISIBLE_DEVICES based on --tp
    cuda_devices = ",".join(str(i) for i in range(args.tp))

    cmd = [
        "vllm", "serve", args.model,
        "--max-model-len", str(args.max_len),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--gpu-memory-utilization", str(args.gpu_util),
        "--tensor-parallel-size", str(args.tp),
        "--port", str(args.port),
        "--host", args.host,
        # Use vLLM defaults instead of HuggingFace generation config
        # (prevents model from overriding temperature/top_p with its defaults)
        "--generation-config", "vllm",
    ]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if args.attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    if args.vllm_log_level:
        env["VLLM_LOG_LEVEL"] = args.vllm_log_level

    log.info("[vLLM] Launching with configuration:")
    log.info(f"  Model: {args.model}")
    log.info(f"  Max context length: {args.max_len}")
    log.info(f"  Port: {args.host}:{args.port}")
    log.info(f"  GPU memory utilization: {args.gpu_util}")
    log.info(f"  Tensor parallel size: {args.tp}")
    log.info(f"  CUDA devices: {cuda_devices}")
    if args.attention_backend:
        log.info(f"  Attention backend: {args.attention_backend}")
    if args.served_model_name:
        log.info(f"  Served model name: {args.served_model_name}")
    if args.vllm_log_level:
        log.info(f"  Log level: {args.vllm_log_level}")
    log.info("")
    log.info(f"Command: {' '.join(cmd)}")
    log.info("")

    try:
        os.execvpe(cmd[0], cmd, env)
    except Exception as e:
        log.error(f"failed to exec vLLM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
