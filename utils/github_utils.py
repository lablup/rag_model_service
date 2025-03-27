#!/usr/bin/env python3
"""
GitHub Utilities for RAG Model Service

This module provides functions for:
1. Parsing GitHub URLs into components
2. Cloning GitHub repositories
3. Extracting documentation from repositories
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import structlog
from git import Repo
from pydantic import BaseModel, Field


# Initialize logger
logger = structlog.get_logger()


class GitHubInfo(BaseModel):
    """Model for GitHub repository information"""
    url: str
    owner: str
    repo: str
    branch: str = "main"
    path: str = ""


def parse_github_url(github_url: str) -> GitHubInfo:
    """
    Parse a GitHub URL into its components.
    
    Args:
        github_url: GitHub URL (https://github.com/owner/repo[/tree/branch][/path/to/docs])
        
    Returns:
        GitHubInfo object with parsed components
        
    Raises:
        ValueError: If the URL is invalid
    """
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Basic URL pattern for GitHub
    pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    match = re.match(pattern, github_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    
    owner = match.group(1)
    repo = match.group(2)
    branch = match.group(3) or "main"  # Default to 'main' if branch is not specified
    path = match.group(4) or ""  # Default to empty string if path is not specified
    
    return GitHubInfo(
        url=github_url,
        owner=owner,
        repo=repo,
        branch=branch,
        path=path
    )


def validate_github_url(url: str) -> bool:
    """
    Validate a GitHub URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Basic GitHub URL pattern
    pattern = r"^https?://github\.com/[^/]+/[^/]+(?:/tree/[^/]+(?:/.*)?)?$"
    return bool(re.match(pattern, url))


def clone_github_repo(
    github_info: GitHubInfo, 
    target_dir: Union[str, Path]
) -> Tuple[Path, Optional[Exception]]:
    """
    Clone a GitHub repository to a local directory.
    
    Args:
        github_info: GitHubInfo object with repository information
        target_dir: Directory to clone the repository to
        
    Returns:
        Tuple of (repository path, exception if any)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Construct repo URL
        repo_url = f"https://github.com/{github_info.owner}/{github_info.repo}.git"
        
        # Clone the repository
        logger.info(
            "Cloning repository", 
            repo=f"{github_info.owner}/{github_info.repo}",
            branch=github_info.branch
        )
        git_repo = Repo.clone_from(repo_url, target_dir, branch=github_info.branch)
        
        # Determine documentation path
        docs_path = target_dir
        if github_info.path:
            docs_path = target_dir / github_info.path
            if not docs_path.exists():
                raise ValueError(f"Documentation path not found: {docs_path}")
                
        return docs_path, None
        
    except Exception as e:
        logger.error(
            "Error cloning repository", 
            repo=f"{github_info.owner}/{github_info.repo}",
            error=str(e)
        )
        return target_dir, e
