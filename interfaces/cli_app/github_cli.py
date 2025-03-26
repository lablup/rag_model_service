#!/usr/bin/env python3
"""
GitHub CLI Tool for RAG Service

This CLI tool provides a command-line interface for:
1. Cloning GitHub repositories
2. Preparing documentation for RAG processing

Usage:
    python github_cli.py clone https://github.com/owner/repo --output-dir ./output
    python github_cli.py prepare https://github.com/owner/repo --output-dir ./output
"""

import argparse
import sys
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from interfaces.portal.github import parse_github_url, prepare_for_rag

# Initialize logger and console
logger = structlog.get_logger()
console = Console()


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="GitHub CLI Tool for RAG Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone GitHub repository")
    clone_parser.add_argument(
        "github_url",
        help="GitHub URL of repository to clone",
    )
    clone_parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for cloned repository",
        default="./github_docs",
    )
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare GitHub repository for RAG")
    prepare_parser.add_argument(
        "github_url",
        help="GitHub URL of repository to prepare",
    )
    prepare_parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for prepared repository",
        default="./github_docs",
    )
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse GitHub URL")
    parse_parser.add_argument(
        "github_url",
        help="GitHub URL to parse",
    )
    
    return parser


def handle_parse(args: argparse.Namespace) -> None:
    """Handle parse command"""
    try:
        owner, repo, branch, path = parse_github_url(args.github_url)
        
        console.print(Panel.fit(
            f"[bold]GitHub URL:[/bold] {args.github_url}\n\n"
            f"[bold]Owner:[/bold] {owner}\n"
            f"[bold]Repository:[/bold] {repo}\n"
            f"[bold]Branch:[/bold] {branch}\n"
            f"[bold]Path:[/bold] {path or '(root)'}"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


def handle_clone(args: argparse.Namespace) -> None:
    """Handle clone command"""
    try:
        console.print(f"[bold]Cloning repository:[/bold] {args.github_url}")
        console.print(f"[bold]Output directory:[/bold] {args.output_dir}")
        
        output_dir = Path(args.output_dir)
        
        # Use prepare_for_rag to clone repository
        docs_path = prepare_for_rag(args.github_url, output_dir)
        
        console.print(f"[bold green]Success![/bold green] Repository cloned to {output_dir}")
        console.print(f"[bold]Documentation path:[/bold] {docs_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


def handle_prepare(args: argparse.Namespace) -> None:
    """Handle prepare command"""
    # This is essentially the same as clone for now
    handle_clone(args)


def main() -> None:
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands
    if args.command == "parse":
        handle_parse(args)
    elif args.command == "clone":
        handle_clone(args)
    elif args.command == "prepare":
        handle_prepare(args)


if __name__ == "__main__":
    main()
