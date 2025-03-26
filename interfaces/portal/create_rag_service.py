#!/usr/bin/env python3
"""
End-to-End RAG Service Creator

This script:
1. Clones a GitHub repository containing documentation
2. Processes the documentation to create vector embeddings
3. Launches a Gradio web interface for the RAG system

Usage:
    python create_rag_service.py --github-url https://github.com/owner/repo
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import structlog
from dotenv import load_dotenv
from git import Repo
import gradio as gr
import re

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from vectordb_manager.vectordb_manager import VectorDBManager
from app.rag_chatbot import LLMConfig, RAGManager
from app.gradio_app import create_gradio_interface

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-End RAG Service Creator"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Docs path (optional, to use existing documentation)
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to existing documentation directory (skips cloning if provided)",
        default=None,
    )
    
    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the RAG service",
        default="./rag_service",
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        help="Host for the Gradio server",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the Gradio server",
        default=7860,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link",
    )
    
    # OpenAI settings
    parser.add_argument(
        "--openai-model",
        type=str,
        help="OpenAI model to use for RAG",
        default="gpt-4o",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM responses",
        default=0.2,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results to retrieve",
        default=5,
    )
    
    # UI customization
    parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the Gradio interface",
        default=None,
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Custom description for the Gradio interface",
        default=None,
    )
    parser.add_argument(
        "--suggested-questions",
        type=str,
        nargs="+",
        help="Custom suggested questions for the UI",
        default=None,
    )
    
    # Advanced options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Skip launching the Gradio interface (setup only)",
    )
    
    return parser.parse_args()


def setup_environment() -> bool:
    """
    Load .env file and validate required environment variables.
    
    Returns:
        bool: True if the environment is properly set up, False otherwise
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Check for OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            print("Error: OpenAI API key is required. Please set it in a .env file or as an environment variable.")
            return False
        
        return True
        
    except Exception as e:
        logger.error("Error setting up environment", error=str(e))
        print(f"Error setting up environment: {str(e)}")
        return False


def parse_github_url(github_url: str) -> Tuple[str, str, str, str]:
    """
    Parse GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple containing owner, repo, branch, and path
    """
    # Handle different GitHub URL formats
    # https://github.com/owner/repo
    # https://github.com/owner/repo/tree/branch
    # https://github.com/owner/repo/tree/branch/path/to/docs
    
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Basic URL pattern for GitHub
    basic_pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    match = re.match(basic_pattern, github_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    
    owner = match.group(1)
    repo = match.group(2)
    branch = match.group(3) or "main"  # Default to 'main' if branch is not specified
    path = match.group(4) or ""  # Default to empty string if path is not specified
    
    return owner, repo, branch, path


def generate_service_config(args) -> Dict:
    """
    Generate service configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Extract GitHub information
    owner, repo, branch, path = parse_github_url(args.github_url)
    
    # Determine output directory
    output_dir = Path(args.output_dir).resolve()
    
    # Create paths
    docs_dir = output_dir / f"{owner}_{repo}"
    indices_dir = output_dir / f"{owner}_{repo}_indices"
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Create configuration
    config = {
        "github": {
            "url": args.github_url,
            "owner": owner,
            "repo": repo,
            "branch": branch or "main",  # Default to main if branch is not specified
            "path": path,
        },
        "paths": {
            "output_dir": output_dir,
            "docs_dir": docs_dir,
            "indices_dir": indices_dir,
            "existing_docs_path": args.docs_path,  # Add the existing docs path
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "share": args.share,
            "debug": args.debug,
        },
        "llm": {
            "model_name": args.openai_model,
            "temperature": args.temperature,
            "max_results": args.max_results,
            "streaming": True,
            "openai_api_key": openai_api_key,
        },
        "ui": {
            "title": args.title,
            "description": args.description,
            "suggested_questions": args.suggested_questions,
        },
    }
    
    # Generate a default title and description based on repository if not provided
    repo_name = f"{owner}/{repo}"
    default_title = f"{repo_name} Documentation Assistant"
    default_description = f"Search and ask questions about {repo_name} documentation"
    
    if not config["ui"]["title"]:
        config["ui"]["title"] = default_title
    if not config["ui"]["description"]:
        config["ui"]["description"] = default_description
    
    return config


async def create_rag_service(config: Dict) -> Tuple[Path, Path]:
    """
    Create RAG service by cloning repository and processing documentation.
    
    Args:
        config: Service configuration
        
    Returns:
        Tuple of documentation path and vector indices path
    """
    try:
        # Create output directory
        output_dir = config["paths"]["output_dir"]
        docs_dir = config["paths"]["docs_dir"]
        indices_dir = config["paths"]["indices_dir"]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone repository
        print(f"Cloning {config['github']['owner']}/{config['github']['repo']} ({config['github']['branch']})...")
        
        repo_url = f"https://github.com/{config['github']['owner']}/{config['github']['repo']}.git"
        
        # Use GitPython to clone the repository
        git_repo = Repo.clone_from(repo_url, docs_dir, branch=config['github']['branch'])
        
        # Determine documentation path
        if config['github']['path']:
            docs_path = docs_dir / config['github']['path']
            if not docs_path.exists():
                raise ValueError(f"Documentation path not found: {docs_path}")
        else:
            # Default to the root of the repository
            docs_path = docs_dir
        
        print(f"Documentation path: {docs_path}")
        
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_dir)
        
        # Collect documents
        print("Collecting documents...")
        documents = await vector_manager.collect_documents()
        
        if not documents:
            logger.warning("No documents found", docs_path=str(docs_path))
            print(f"Warning: No documents found in {docs_path}")
            print("Please check that the repository contains markdown (.md) files.")
            return docs_path, indices_dir
        
        print(f"Found {len(documents)} documents")
        
        # Create vector indices
        print("Creating vector indices...")
        await vector_manager.create_indices(documents)
        
        print(f"Vector indices created successfully in {indices_dir}")
        
        return docs_path, indices_dir
        
    except Exception as e:
        logger.error("Error creating RAG service", error=str(e))
        print(f"Error creating RAG service: {str(e)}")
        raise


async def launch_service(docs_path: Path, indices_path: Path, config: Dict) -> None:
    """
    Launch RAG service with Gradio interface.
    
    Args:
        docs_path: Path to documentation
        indices_path: Path to vector indices
        config: Service configuration
    """
    try:
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_path)
        
        # Load vector indices
        await vector_manager.load_index()
        
        if not vector_manager.index:
            logger.error("Failed to load vector index", indices_path=str(indices_path))
            print(f"Error: Failed to load vector index from {indices_path}")
            return
        
        # Initialize LLMConfig
        llm_config = LLMConfig(
            openai_api_key=config["llm"]["openai_api_key"],
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_results=config["llm"]["max_results"],
            streaming=config["llm"]["streaming"],
        )
        
        # Initialize RAGManager
        rag_manager = RAGManager(llm_config, vector_manager)
        
        # Create Gradio interface
        interface = create_gradio_interface(rag_manager)
        
        # Customize interface
        interface.title = config["ui"]["title"]
        
        # Launch the interface
        print(f"\nLaunching Gradio server with {config['llm']['model_name']} for RAG...")
        print(f"Documentation path: {docs_path}")
        print(f"Vector indices path: {indices_path}")
        print(f"Server: {config['server']['host']}:{config['server']['port']}")
        
        if config['server']['share']:
            print("Creating shareable link...")
        
        # Launch the server
        interface.launch(
            server_name=config["server"]["host"],
            server_port=config["server"]["port"],
            share=config["server"]["share"],
            debug=config["server"]["debug"],
        )
        
    except Exception as e:
        logger.error("Error launching service", error=str(e))
        print(f"Error launching service: {str(e)}")
        raise


async def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Generate service configuration
        config = generate_service_config(args)
        
        # Create RAG service
        if args.docs_path:
            docs_path = Path(args.docs_path)
            indices_path = config["paths"]["indices_dir"]
            print(f"Using existing documentation directory: {docs_path}")
        else:
            docs_path, indices_path = await create_rag_service(config)
        
        # Launch service unless skip-launch is specified
        if not args.skip_launch:
            await launch_service(docs_path, indices_path, config)
        else:
            print("\nRAG service setup completed successfully!")
            print("Service is ready but not launched (--skip-launch specified)")
            print(f"\nTo launch the service manually, run:")
            print(f"python launch_gradio.py --docs-path {docs_path} --indices-path {indices_path}")
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)