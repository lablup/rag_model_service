#!/usr/bin/env python3
"""
Test Model Definition Generator

This script tests the model definition generator with various GitHub URL formats.
"""

import os
import sys
from pathlib import Path
import tempfile

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the function to test
from auto_rag_service.generate_model_definition import (
    parse_github_url,
    generate_model_name,
    generate_docs_name,
    generate_model_definition
)

def test_url_parsing():
    """Test URL parsing with various GitHub URL formats."""
    test_urls = [
        "https://github.com/lablup/backend.ai",
        "https://github.com/lablup/backend.ai/tree/main",
        "https://github.com/lablup/backend.ai/tree/main/docs",
        "https://github.com/reflex-dev/reflex-web/tree/main/docs",
        "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs",
        "https://github.com/pytorch/pytorch/tree/master/docs/source",
        # Test URLs with 'blob' instead of 'tree'
        "https://github.com/lablup/backend.ai/blob/main/README.md",
        "https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md",
        # Test URLs with different branch names
        "https://github.com/huggingface/transformers/tree/v4.28.0/docs",
        # Test URLs with trailing slashes
        "https://github.com/lablup/backend.ai/",
        "https://github.com/lablup/backend.ai/tree/main/docs/"
    ]
    
    print("Testing URL parsing...")
    for url in test_urls:
        owner, repo, branch, path = parse_github_url(url)
        model_name = generate_model_name(owner, repo, path, "RAG Service for")
        docs_name = generate_docs_name(owner, repo, path)
        
        print(f"\nURL: {url}")
        print(f"  Owner: {owner}")
        print(f"  Repo: {repo}")
        print(f"  Branch: {branch}")
        print(f"  Path: {path}")
        print(f"  Model Name: {model_name}")
        print(f"  Docs Name: {docs_name}")
        print(f"  YAML Filename: model-definition-{docs_name}.yaml")

def test_blob_url_parsing():
    """Test parsing of GitHub blob URLs specifically."""
    test_urls = [
        "https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md",
        "https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md",
        "https://github.com/huggingface/transformers/blob/main/docs/source/en/quicktour.md"
    ]
    
    print("\nTesting blob URL parsing...")
    for url in test_urls:
        owner, repo, branch, path = parse_github_url(url)
        
        print(f"\nURL: {url}")
        print(f"  Owner: {owner}")
        print(f"  Repo: {repo}")
        print(f"  Branch: {branch}")
        print(f"  Path: {path}")
        
        # Generate a model definition with this URL
        model_name = generate_model_name(owner, repo, path, "RAG Service for")
        model_def = generate_model_definition(url, model_name, 8000, "gradio")
        
        print(f"  Model Name: {model_name}")
        print(f"  Start Command Path Arg: {model_def['models'][0]['service']['start_command'][2]}")

def test_model_definition_generation():
    """Test model definition generation with various GitHub URL formats."""
    test_urls = [
        "https://github.com/lablup/backend.ai",
        "https://github.com/reflex-dev/reflex-web/tree/main/docs",
        "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs",
        "https://github.com/pytorch/pytorch/tree/master/docs/source",
    ]
    
    print("\nTesting model definition generation...")
    for url in test_urls:
        owner, repo, branch, path = parse_github_url(url)
        
        # Generate a model name
        model_name = generate_model_name(owner, repo, path, "RAG Service for")
        
        # Generate a model definition
        model_def = generate_model_definition(url, model_name, 8000, "gradio")
        
        print(f"\nURL: {url}")
        print(f"  Model Name: {model_name}")
        print(f"  Path: {path}")
        print("  Model Definition:")
        
        # Print the model definition in a readable format
        model_info = model_def["models"][0]
        service_value = model_info["service"]
        
        print(f"    name: {model_info['name']}")
        print(f"    model_path: {model_info['model_path']}")
        print(f"    service:")
        print(f"      pre_start_actions: {service_value['pre_start_actions']}")
        print(f"      start_command: {service_value['start_command']}")
        print(f"      docs_path_arg: {service_value['docs_path_arg']}")
        print(f"      port: {service_value['port']}")

def main():
    """Main function."""
    test_url_parsing()
    test_blob_url_parsing()
    test_model_definition_generation()
    return 0

if __name__ == "__main__":
    sys.exit(main())
