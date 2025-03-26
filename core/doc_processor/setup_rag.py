#!/usr/bin/env python3
"""
GitHub Documentation Processor for RAG Service

This script:
1. Clones a GitHub repository containing documentation
2. Processes the documentation to create vector embeddings
3. Tests the RAG system with sample queries

Usage:
    python setup_rag.py --github-url https://github.com/owner/repo --output-dir ./doc

Advanced Usage:
    python setup_rag.py --github-url https://github.com/owner/repo/tree/branch/path/to/docs \
                   --indices-path ./my_indices \
                   --output-dir ./my_docs \
                   --openai-model gpt-4o \
                   --temperature 0.2 \
                   --max-results 5 \
                   --test-queries "What are the main features?" "How do I install this?"

Using Existing Docs:
    python setup_rag.py --docs-path ./path/to/docs

Skipping Usage:
    # Skip cloning (use existing docs)
    python setup_rag.py --github-url https://github.com/owner/repo --skip-clone --docs-path ./existing_docs

    # Skip indexing (use existing indices)
    python setup_rag.py --github-url https://github.com/owner/repo --skip-indexing

    # Skip testing
    python setup_rag.py --github-url https://github.com/owner/repo --skip-testing

Chatting:
    python -m app.rag_chatbot --docs-path github_docs/docs --indices-path embedding_indices
"""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import aiofiles
from dotenv import load_dotenv
import structlog
from git import Repo
import re

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from vectordb_manager.vectordb_manager import VectorDBManager
from app.rag_chatbot import LLMConfig, RAGManager

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GitHub Documentation Processor for RAG Service"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
    )
    
    # Paths
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to documentation directory (if not using GitHub URL)",
    )
    parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to store vector indices",
        default="./embedding_indices",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for cloned repository",
        default="./github_docs",
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
    
    # Test queries
    parser.add_argument(
        "--test-queries",
        type=str,
        nargs="+",
        help="Test queries to run against the RAG system",
        default=[
            "What are the main features?",
            "How do I install this?",
            "What configuration options are available?",
        ],
    )

    # Actions
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning the repository (use existing docs path)",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing the documentation (use existing indices)",
    )
    parser.add_argument(
        "--skip-testing",
        action="store_true",
        help="Skip testing the RAG system",
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


async def clone_github_repo(github_url: str, target_dir: Path) -> Path:
    """
    Clone repository from GitHub URL to target directory.
    
    Args:
        github_url: GitHub URL
        target_dir: Target directory for the cloned repository
        
    Returns:
        Path to documentation directory
    """
    try:
        # Parse GitHub URL
        owner, repo, branch, path = parse_github_url(github_url)
        logger.info(
            "Parsed GitHub URL",
            owner=owner,
            repo=repo,
            branch=branch,
            path=path
        )
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone repository
        print(f"Cloning repository {owner}/{repo} (branch: {branch})...")
        repo_url = f"https://github.com/{owner}/{repo}.git"
        
        # Use GitPython to clone the repository
        git_repo = Repo.clone_from(repo_url, target_dir, branch=branch)
        
        logger.info("Repository cloned successfully", target_dir=str(target_dir))
        print(f"Repository cloned to {target_dir}")
        
        # Determine documentation directory
        if path:
            docs_path = target_dir / path
            if not docs_path.exists():
                raise ValueError(f"Documentation path not found: {docs_path}")
        else:
            # Default to the root of the repository
            docs_path = target_dir
        
        return docs_path
        
    except Exception as e:
        logger.error("Error cloning repository", error=str(e))
        print(f"Error cloning repository: {str(e)}")
        raise


async def process_documentation(docs_path: Path, indices_path: Path) -> None:
    """
    Process documentation to create vector indices.
    
    Args:
        docs_path: Path to documentation directory
        indices_path: Path to store vector indices
    """
    try:
        print(f"Processing documentation in {docs_path}...")
        
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_path)
        
        # Collect documents
        print("Collecting documents...")
        documents = await vector_manager.collect_documents()
        
        if not documents:
            logger.warning("No documents found", docs_path=str(docs_path))
            print(f"Warning: No documents found in {docs_path}")
            return
        
        print(f"Found {len(documents)} documents")
        
        # Create vector indices
        print("Creating vector indices...")
        await vector_manager.create_indices(documents)
        
        print(f"Vector indices created successfully in {indices_path}")
        
    except Exception as e:
        logger.error("Error processing documentation", error=str(e))
        print(f"Error processing documentation: {str(e)}")
        raise


async def test_rag_system(indices_path: Path, docs_path: Path, test_queries: List[str], model_name: str = "gpt-4o", temperature: float = 0.2, max_results: int = 5) -> None:
    """
    Test RAG system with sample queries.
    
    Args:
        indices_path: Path to vector indices
        docs_path: Path to documentation directory
        test_queries: List of test queries
        model_name: OpenAI model name
        temperature: Temperature for LLM responses
        max_results: Maximum number of results to retrieve
    """
    try:
        print("\nTesting RAG system...")
        
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_path)
        
        # Load vector index
        await vector_manager.load_index()
        
        if not vector_manager.index:
            logger.error("Failed to load vector index", indices_path=str(indices_path))
            print(f"Error: Failed to load vector index from {indices_path}")
            return
        
        # Initialize RAGManager
        llm_config = LLMConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            model_name=model_name,
            temperature=temperature,
            max_results=max_results,
            streaming=False,  # Disable streaming for testing
        )
        
        rag_manager = RAGManager(llm_config, vector_manager)
        
        # Run test queries
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest Query {i}: \"{query}\"")
            
            # Get context
            context, results = await rag_manager._get_relevant_context(query)
            
            print(f"Found {len(results)} relevant documents")
            
            # Print top 2 documents for reference
            for j, result in enumerate(results[:2], 1):
                metadata = result.get("metadata", {})
                similarity = result.get("similarity_score", 0.0)
                relative_path = metadata.get("relative_path", "unknown")
                
                print(f"  Document {j}: {relative_path} (score: {similarity:.4f})")
            
            # Generate response
            print("Generating response...")
            full_response = ""
            async for chunk in rag_manager.generate_response_with_context(query, context):
                full_response += chunk
            
            print(f"Response: {full_response}")
            print("-" * 80)
        
    except Exception as e:
        logger.error("Error testing RAG system", error=str(e))
        print(f"Error testing RAG system: {str(e)}")
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
        
        # Resolve paths
        output_dir = Path(args.output_dir) if args.output_dir else Path("./github_docs")
        indices_path = Path(args.indices_path) if args.indices_path else Path("./embedding_indices")
        
        # Clone repository if URL provided and not skipped
        docs_path = None
        if args.github_url and not args.skip_clone:
            docs_path = await clone_github_repo(args.github_url, output_dir)
        elif args.docs_path:
            docs_path = Path(args.docs_path)
            if not docs_path.exists():
                logger.error("Documentation path does not exist", path=str(docs_path))
                print(f"Error: Documentation path does not exist: {docs_path}")
                return 1
        else:
            logger.error("No GitHub URL or documentation path provided")
            print("Error: Either --github-url or --docs-path must be provided")
            return 1
        
        # Process documentation if not skipped
        if not args.skip_indexing:
            await process_documentation(docs_path, indices_path)
        
        # Test RAG system if not skipped
        if not args.skip_testing:
            await test_rag_system(
                indices_path,
                docs_path,
                args.test_queries,
                args.openai_model,
                args.temperature,
                args.max_results
            )
        
        print("\nSuccess! RAG system is set up and ready to use.")
        print(f"Documentation: {docs_path}")
        print(f"Vector indices: {indices_path}")
        
        # Provide hint for next steps
        print("\nNext steps:")
        print("1. Run the RAG chat interface:")
        print(f"   python -m app.rag_chatbot --docs-path {docs_path} --indices-path {indices_path}")
        print("2. Or run the Gradio web interface:")
        print(f"   python app/gradio_app.py")
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
