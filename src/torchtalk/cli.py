#!/usr/bin/env python3
"""
TorchTalk CLI - Cross-language binding analysis for PyTorch codebases.

One-command setup for Claude Code:
    claude mcp add torchtalk -s user -- torchtalk mcp-serve --pytorch-source /path/to/pytorch

The MCP server automatically builds and caches the binding index on first run.
"""

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def cmd_mcp_serve(args):
    """Start MCP server for Claude Code integration"""
    from torchtalk.server import run_server

    run_server(
        pytorch_source=args.pytorch_source,
        index_path=args.index,
        transport=args.transport,
    )
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="torchtalk",
        description="TorchTalk - Cross-language binding analysis for PyTorch codebases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
One-command Claude Code setup:
  claude mcp add torchtalk -s user -- torchtalk mcp-serve --pytorch-source /path/to/pytorch

The MCP server automatically:
  - Builds the binding index on first run (takes a few minutes)
  - Caches to ~/.cache/torchtalk/ for instant startup next time
  - Auto-detects PyTorch source if PYTORCH_SOURCE env var is set

Examples:
  # With explicit PyTorch source path
  torchtalk mcp-serve --pytorch-source /myworkspace/pytorch

  # Auto-detect (checks PYTORCH_SOURCE, ./pytorch, ../pytorch)
  torchtalk mcp-serve

  # Use pre-built index
  torchtalk mcp-serve --index ./pytorch_index
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # MCP server command (primary command)
    parser_mcp = subparsers.add_parser(
        "mcp-serve",
        help="Start MCP server for Claude Code integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The server automatically builds and caches the binding index.
First run takes a few minutes; subsequent runs are instant.
        """,
    )
    parser_mcp.add_argument(
        "--pytorch-source",
        "-p",
        help="Path to PyTorch source code (auto-builds and caches index)",
    )
    parser_mcp.add_argument(
        "--index", "-i", help="Path to existing bindings.json or index directory"
    )
    parser_mcp.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="MCP transport type (default: stdio for Claude Code)",
    )
    parser_mcp.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser_mcp.set_defaults(func=cmd_mcp_serve)

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
