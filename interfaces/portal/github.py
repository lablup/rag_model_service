#!/usr/bin/env python3
"""
GitHub Repository Handler for RAG Service

This module provides functionality for:
1. Parsing GitHub repository URLs
2. Cloning GitHub repositories
3. Preparing documentation for RAG processing

Note: This module now delegates to utils.github_utils for core GitHub functionality.
"""

import os
import asyncio
from pathlib import Path
from typing import Tuple, Optional

import structlog

# Import centralized GitHub utilities
from utils.github_utils import parse_github_url as utils_parse_github_url
from utils.github_utils import clone_github_repo as utils_clone_github_repo

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
    # Use the centralized implementation but convert the return format
    github_info = utils_parse_github_url(github_url)
    return github_info.owner, github_info.repo, github_info.branch, github_info.path


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
        # Parse GitHub URL using centralized utility
        github_info = utils_parse_github_url(github_url)
        
        logger.info(
            "Parsed GitHub URL",
            owner=github_info.owner,
            repo=github_info.repo,
            branch=github_info.branch,
            path=github_info.path,
        )
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Clone repository using centralized utility
        print(f"Cloning {github_info.owner}/{github_info.repo} repository...")
        
        docs_path, error = utils_clone_github_repo(github_info, target_dir)
        
        if error:
            raise error
            
        # Create docs directory if it doesn't exist
        docs_target = target_dir / "docs"
        docs_target.mkdir(exist_ok=True, parents=True)
        
        # If docs_path is different from docs_target, copy the contents
        if docs_path != docs_target and docs_path.exists():
            print(f"Copying documentation from {docs_path} to {docs_target}...")
            # Use os.system instead of subprocess to simplify the async environment
            os.system(f"cp -r {str(docs_path)}/. {str(docs_target)}")
            print(f"Documentation copied successfully to {docs_target}")
            return docs_target
        else:
            return docs_path
        
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
        docs_path = asyncio.run(clone_github_repo(github_url, output_dir))
        
        print(f"Repository prepared for RAG processing. Documentation path: {docs_path}")
        
        return docs_path
        
    except Exception as e:
        logger.error("Error preparing repository for RAG", error=str(e))
        print(f"Error preparing repository for RAG: {str(e)}")
        raise