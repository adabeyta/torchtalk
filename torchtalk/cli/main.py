#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from torchtalk.core.config import get_config, set_config
from torchtalk.analysis.repo_analyzer import RepoAnalyzer


def analyze_repository(repo_path: str) -> str:
    print(f"Analyzing repository: {repo_path}")
    print("Building import graphs, call graphs, and extracting code...")

    analyzer = RepoAnalyzer(repo_path)
    analysis_path = analyzer.save_analysis()

    print(f"Enhanced analysis complete: {analysis_path}")
    return analysis_path


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


def start_dynamic_services(context_profile: str = "dev"):
    config = get_config()
    config.context_profile = context_profile
    set_config(config)

    print(f"Starting dynamic TorchTalk services with profile: {context_profile}")

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
        vllm_proc = subprocess.Popen(vllm_cmd, env=env)
        processes.append(("vLLM", vllm_proc))

        # Wait a bit for vLLM to start, not sure if this is needed.
        import time
        time.sleep(10)

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
            [sys.executable, "-m", "torchtalk.web.ui"],
            env=env
        )
        processes.append(("Gradio", gradio_proc))

        print(f"Services started!")
        print(f"   vLLM server: http://localhost:{config.vllm_port}")
        print(f"   FastAPI: http://localhost:{config.fastapi_port}")
        print(f"   Gradio UI: http://localhost:{config.gradio_port}")
        print(f"   vLLM endpoint: {config.vllm_endpoint}")
        print("\nPress Ctrl+C to stop all services")

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


def main():
    parser = argparse.ArgumentParser(description='Dynamic TorchTalk - Agnostic analysis and chat')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a repository dynamically')
    analyze_parser.add_argument('repo_path', help='Path to repository to analyze')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start dynamic chat services')
    start_parser.add_argument('--repo', required=True, help='Repository path to work with')
    start_parser.add_argument('--context', default='dev', choices=['dev', 'production_128k', 'production_1m'],
                             help='Context profile to use')
    start_parser.add_argument('--model', help='Model to use (e.g., meta-llama/Llama-3.1-70B-Instruct)')
    start_parser.add_argument('--vllm-endpoint', help='Custom vLLM endpoint (e.g., http://gpu-server:8000/v1/chat/completions)')
    start_parser.add_argument('--force-analyze', action='store_true',
                             help='Force re-analysis even if analysis exists')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'analyze':
        analyze_repository(args.repo_path)

    elif args.command == 'start':
        update_config_for_repo(args.repo)
        config = get_config()

        if args.model:
            config.model_name = args.model
            print(f"Using model: {args.model}")

        if hasattr(args, 'vllm_endpoint') and args.vllm_endpoint:
            config.vllm_endpoint = args.vllm_endpoint
            print(f"Using vLLM endpoint: {args.vllm_endpoint}")

        config.save()
        set_config(config)

        repo_name = Path(args.repo).name
        enhanced_analysis = f"artifacts/{repo_name}_enhanced_analysis.json"

        if not os.path.exists(enhanced_analysis) or args.force_analyze:
            print("No enhanced analysis found or forced re-analysis requested")
            analyze_repository(args.repo)
            config.analysis_file = enhanced_analysis
        else:
            print(f"Using existing enhanced analysis: {enhanced_analysis}")
            config.analysis_file = enhanced_analysis

        config.save()
        set_config(config)

        # Start services
        start_dynamic_services(args.context)


if __name__ == '__main__':
    main()