#!/usr/bin/env python3
"""
TorchTalk CLI - Cross-language binding analysis for PyTorch codebases.

Quick start:
    torchtalk init --pytorch-source /path/to/pytorch
    claude mcp add torchtalk -s user -- torchtalk mcp-serve

The MCP server automatically builds and caches the binding index on first run.
"""

import argparse
import logging
import sys
import shutil
import json
from pathlib import Path

log = logging.getLogger(__name__)


def cmd_init(args):
    """Configure TorchTalk with PyTorch source path."""
    from torchtalk.config import load_config, save_config, validate_pytorch_path

    source_path = str(Path(args.pytorch_source).resolve())

    valid, msg = validate_pytorch_path(source_path)
    if not valid:
        log.error(msg)
        sys.exit(1)

    config = load_config()
    config.setdefault("source", {})["pytorch_source"] = source_path
    path = save_config(config)

    log.info(
        "Config saved to %s\n"
        "  pytorch_source = %s\n\n"
        "Next steps:\n"
        "  claude mcp add torchtalk -s user -- torchtalk mcp-serve\n\n"
        "Verify with:\n"
        "  torchtalk status",
        path, source_path,
    )


def cmd_status(args):
    """Show TorchTalk configuration and cache status."""
    from datetime import datetime

    from torchtalk import __version__
    from torchtalk.config import (
        CACHE_DIR,
        CONFIG_FILE,
        cache_paths,
        load_config,
        resolve_pytorch_source,
        validate_pytorch_path,
    )

    lines = [f"TorchTalk v{__version__}", ""]

    # Config file
    lines.append("Configuration:")
    if not CONFIG_FILE.exists():
        lines.append(f"  Config file:     {CONFIG_FILE} (not found)")
        lines.append("")
        lines.append("Run 'torchtalk init --pytorch-source /path/to/pytorch' to configure.")
        print("\n".join(lines))
        sys.exit(1)

    lines.append(f"  Config file:     {CONFIG_FILE}")

    # PyTorch source — show raw config value for diagnostics, even if stale
    config = load_config()
    config_source = config.get("source", {}).get("pytorch_source")

    if not config_source:
        lines.append("  PyTorch source:  not configured")
        lines.append("")
        lines.append("Run 'torchtalk init --pytorch-source /path/to/pytorch' to configure.")
        print("\n".join(lines))
        sys.exit(1)

    valid, msg = validate_pytorch_path(config_source)
    lines.append(f"  PyTorch source:  {config_source} ({'valid' if valid else 'INVALID'})")
    if not valid:
        lines.append(f"                   {msg}")
        lines.append("")
        lines.append("Run 'torchtalk init --pytorch-source /path/to/pytorch' to reconfigure.")
        print("\n".join(lines))
        sys.exit(1)

    source = resolve_pytorch_source()

    # Cache
    lines.append("")
    lines.append("Cache:")
    lines.append(f"  Cache directory:  {CACHE_DIR}")

    if CACHE_DIR.exists():
        paths = cache_paths(source)
        for key, label in [("bindings", "Bindings index"), ("callgraph", "Call graph")]:
            p = paths[key]
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                lines.append(f"  {label + ':':18s} {p.name} ({size_mb:.1f} MB, {mtime:%Y-%m-%d %H:%M})")
            else:
                lines.append(f"  {label + ':':18s} not built")
    else:
        lines.append("  Cache directory:  not created yet")

    # compile_commands.json
    if source:
        compile_cmds = Path(source) / "build" / "compile_commands.json"
        lines.append("")
        lines.append("Build artifacts:")
        if compile_cmds.exists():
            lines.append(f"  compile_commands.json: found")
        else:
            lines.append(f"  compile_commands.json: not found (call graph features unavailable)")
            lines.append(f"    Build PyTorch to enable: cd {source} && python setup.py develop")

    lines.append("")
    lines.append("Status: Ready")
    print("\n".join(lines))


def cmd_mcp_serve(args):
    """Start MCP server for Claude Code integration."""
    from torchtalk.server import run_server

    run_server(
        pytorch_source=args.pytorch_source,
        index_path=args.index,
        transport=args.transport,
    )
    return 0


def cmd_cursor_add(args):
    """Add torchtalk MCP server to Cursor and copy .claude/ to project .cursor/."""
    MCP_CONFIG_NAME = "mcp.json"
    project_root = Path(args.project_dir).resolve()
    if not project_root.is_dir():
        log.error("Project directory is not a directory: %s", project_root)
        return 1

    cursor_dir = project_root / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)

    # Copy torchtalk's .claude/ contents into project_dir/.cursor/
    pkg_root = Path(__file__).resolve().parent.parent.parent
    claude_src = pkg_root / ".claude"
    if claude_src.is_dir():
        for item in claude_src.iterdir():
            dest = cursor_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"Copied .claude/ to {cursor_dir}")
    else:
        log.error("Setup failed: no .claude/ directory found at %s", claude_src)
        return 1

    # MCP config: project or user
    if args.global_config:
        config_dir = Path.home() / ".cursor"
        config_path = config_dir / MCP_CONFIG_NAME
    else:
        config_path = cursor_dir / MCP_CONFIG_NAME

    pytorch_source = Path(args.pytorch_source).resolve()
    if not pytorch_source.is_dir():
        log.error("PyTorch source path is not a directory: %s", pytorch_source)
        return 1

    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.error("Could not read %s: %s", config_path, e)
            return 1
    else:
        data = {}

    if "mcpServers" not in data:
        data["mcpServers"] = {}

    data["mcpServers"]["torchtalk"] = {
        "command": "torchtalk",
        "args": ["mcp-serve", "--pytorch-source", str(pytorch_source)],
    }

    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        log.error("Could not write %s: %s", config_path, e)
        return 1

    scope = "user config" if args.global_config else "project"
    print(f"Added torchtalk to {scope} at {config_path}")
    print("Restart Cursor (or reload the window) to load the MCP server.")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="torchtalk",
        description="TorchTalk - Cross-language binding analysis for PyTorch codebases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  torchtalk init --pytorch-source /path/to/pytorch
  claude mcp add torchtalk -s user -- torchtalk mcp-serve

Verify setup:
  torchtalk status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    parser_init = subparsers.add_parser(
        "init",
        help="Configure TorchTalk with PyTorch source path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Saves configuration to ~/.config/torchtalk/config.toml.
Safe to re-run -- updates existing config without data loss.

Example:
  torchtalk init --pytorch-source /myworkspace/pytorch
        """,
    )
    parser_init.add_argument(
        "--pytorch-source",
        "-p",
        required=True,
        help="Path to PyTorch source code",
    )
    parser_init.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser_init.set_defaults(func=cmd_init)

    # status command
    subparsers.add_parser(
        "status",
        help="Show configuration and cache status",
    ).set_defaults(func=cmd_status)

    # mcp-serve command
    parser_mcp = subparsers.add_parser(
        "mcp-serve",
        help="Start MCP server for Claude Code integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reads PyTorch source from config. Override with --pytorch-source flag.
First run builds the index (a few minutes); subsequent runs use cache.
        """,
    )
    parser_mcp.add_argument(
        "--pytorch-source",
        "-p",
        help="Override PyTorch source path (default: from config)",
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

    # Cursor: add MCP to .cursor/mcp.json
    parser_cursor = subparsers.add_parser(
        "cursor-add",
        help="Add torchtalk MCP to Cursor's .cursor/mcp.json",
    )
    parser_cursor.add_argument(
        "--project-dir",
        "-C",
        required=True,
        help="Project root: .cursor/mcp.json and .claude contents are written here",
    )
    parser_cursor.add_argument(
        "--pytorch-source",
        "-p",
        required=True,
        help="Path to PyTorch source tree",
    )
    parser_cursor.add_argument(
        "--global",
        dest="global_config",
        action="store_true",
        help="Write to ~/.cursor/mcp.json (user-wide) instead of project .cursor/mcp.json",
    )
    parser_cursor.set_defaults(func=cmd_cursor_add)

    args = parser.parse_args()

    log_level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s" if log_level == logging.INFO else "%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
