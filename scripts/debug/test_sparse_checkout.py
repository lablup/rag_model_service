#!/usr/bin/env python3
"""
Test script for sparse checkout functionality.

This script tests the ability to clone only a specific directory from a GitHub repository
instead of the entire repository.
"""

import os
import sys
import shutil
import asyncio
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.github_utils import parse_github_url, clone_github_repo
from interfaces.portal.github import clone_github_repo as portal_clone_github_repo


async def test_sparse_checkout():
    """Test the sparse checkout functionality"""
    # Test URL with a specific path (docs directory)
    test_url = "https://github.com/vllm-project/vllm/tree/main/docs"
    
    # Create a temporary directory for testing
    test_dir = Path("./test_sparse_checkout")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    print(f"Testing sparse checkout with URL: {test_url}")
    print(f"Target directory: {test_dir}")
    
    # Parse the GitHub URL
    github_info = parse_github_url(test_url)
    print(f"Parsed GitHub URL: {github_info}")
    
    # Test the utils.github_utils.clone_github_repo function
    print("\nTesting utils.github_utils.clone_github_repo...")
    utils_test_dir = test_dir / "utils_test"
    if utils_test_dir.exists():
        shutil.rmtree(utils_test_dir)
    utils_test_dir.mkdir(exist_ok=True)
    
    docs_path, error = clone_github_repo(github_info, utils_test_dir)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Success! Docs path: {docs_path}")
        # List the contents of the docs directory
        print("\nContents of docs directory:")
        for item in docs_path.iterdir():
            print(f"  {item.name}")
    
    # Test the interfaces.portal.github.clone_github_repo function
    print("\nTesting interfaces.portal.github.clone_github_repo...")
    portal_test_dir = test_dir / "portal_test"
    if portal_test_dir.exists():
        shutil.rmtree(portal_test_dir)
    portal_test_dir.mkdir(exist_ok=True)
    
    try:
        portal_docs_path = await portal_clone_github_repo(test_url, portal_test_dir)
        print(f"Success! Portal docs path: {portal_docs_path}")
        # List the contents of the portal docs directory
        print("\nContents of portal docs directory:")
        for item in portal_docs_path.iterdir():
            print(f"  {item.name}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_sparse_checkout())
