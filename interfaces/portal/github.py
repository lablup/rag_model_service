#!/usr/bin/env python3
"""
GitHub Repository Handler for RAG Service

This module provides functionality for:
1. Parsing GitHub repository URLs
2. Cloning GitHub repositories
3. Preparing documentation for RAG processing
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import structlog

# Initialize logger
logger = structlog.get_logger()


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
        target_dir: Target directory for cloned repository
        
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
            path=path,
        )
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Clone repository
        print(f"Cloning {owner}/{repo} repository...")
        
        repo_url = f"https://github.com/{owner}/{repo}.git"
        clone_dir = target_dir / "repo"
        
        # Remove existing directory if it exists
        if clone_dir.exists():
            print(f"Removing existing repository at {clone_dir}...")
            subprocess.run(["rm", "-rf", str(clone_dir)], check=True)
        
        # Clone repository
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(clone_dir)],
            check=True,
        )
        
        print(f"Repository cloned successfully to {clone_dir}")
        
        # Determine docs directory
        if path:
            docs_dir = clone_dir / path
        else:
            # Look for common documentation directories
            common_doc_dirs = ["docs", "doc", "documentation", "wiki"]
            for doc_dir in common_doc_dirs:
                potential_dir = clone_dir / doc_dir
                if potential_dir.exists() and potential_dir.is_dir():
                    docs_dir = potential_dir
                    break
            else:
                # If no common doc directory found, use the repo root
                docs_dir = clone_dir
        
        # Create docs directory if it doesn't exist
        docs_target = target_dir / "docs"
        docs_target.mkdir(exist_ok=True, parents=True)
        
        # Copy documentation to docs directory
        print(f"Copying documentation from {docs_dir} to {docs_target}...")
        subprocess.run(["cp", "-r", f"{str(docs_dir)}/.", str(docs_target)], check=True)
        
        print(f"Documentation copied successfully to {docs_target}")
        
        return docs_target
        
    except Exception as e:
        logger.error("Error cloning repository", error=str(e))
        print(f"Error cloning repository: {str(e)}")
        raise


def prepare_for_rag(github_url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Prepare GitHub repository for RAG processing.
    
    Args:
        github_url: GitHub URL
        output_dir: Output directory for cloned repository (default: ./github_docs)
        
    Returns:
        Path to documentation directory
    """
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = Path("./github_docs")
        
        # Ensure output directory is a Path object
        output_dir = Path(output_dir)
        
        # Clone repository
        import asyncio
        docs_path = asyncio.run(clone_github_repo(github_url, output_dir))
        
        print(f"Repository prepared for RAG processing. Documentation path: {docs_path}")
        
        return docs_path
        
    except Exception as e:
        logger.error("Error preparing repository for RAG", error=str(e))
        print(f"Error preparing repository for RAG: {str(e)}")
        raise