#!/usr/bin/env python3
"""
Test Model Definition Generator

This script tests the model definition generator with various GitHub URL formats.
"""

import os
import sys
from pathlib import Path
import tempfile
import yaml

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the functions to test
from interfaces.portal.generate_model_definition import (
    parse_github_url,
    generate_model_name,
    generate_docs_name
)

# Import our local implementation for model definition generation
from interfaces.cli_app.generate_model_definition_cli import generate_model_definition_local
from config.config import BACKEND_MODEL_PATH

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
        service_id = "test123"
        model_def = generate_model_definition_local(url, model_name, 8000, "gradio", service_id)
        
        print(f"  Model Name: {model_name}")
        # Find the docs-path argument in the start command
        docs_path_index = model_def['models'][0]['service']['start_command'].index('--docs-path') + 1
        docs_path = model_def['models'][0]['service']['start_command'][docs_path_index]
        print(f"  Docs Path: {docs_path}")

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
        service_id = "test456"
        
        # Generate a model definition
        model_def = generate_model_definition_local(url, model_name, 8000, "gradio", service_id)
        
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
        print(f"      port: {service_value['port']}")
        print(f"      pre_start_actions: {len(service_value['pre_start_actions'])} action(s)")
        
        # Find docs-path in start command
        docs_path_index = service_value['start_command'].index('--docs-path') + 1
        docs_path = service_value['start_command'][docs_path_index]
        print(f"      docs_path: {docs_path}")
        
        # Find service-id in start command
        service_id_index = service_value['start_command'].index('--service-id') + 1
        service_id_value = service_value['start_command'][service_id_index]
        print(f"      service_id: {service_id_value}")

def test_write_model_definition():
    """Test writing model definition to a YAML file."""
    url = "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs"
    owner, repo, branch, path = parse_github_url(url)
    model_name = generate_model_name(owner, repo, path, "RAG Service for")
    service_id = "test789"
    
    # Generate a model definition
    model_def = generate_model_definition_local(url, model_name, 8000, "gradio", service_id)
    
    # Create a temporary file to write the model definition
    with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Write the model definition to the temporary file
        with open(temp_path, 'w') as f:
            yaml.dump(model_def, f, default_flow_style=False)
        
        # Read the model definition back from the file
        with open(temp_path, 'r') as f:
            loaded_def = yaml.safe_load(f)
        
        # Verify that the loaded definition matches the original
        print("\nTesting model definition writing...")
        print(f"  Original model name: {model_def['models'][0]['name']}")
        print(f"  Loaded model name: {loaded_def['models'][0]['name']}")
        print(f"  Match: {model_def['models'][0]['name'] == loaded_def['models'][0]['name']}")
        
        # Verify that the start command is preserved
        original_start = model_def['models'][0]['service']['start_command']
        loaded_start = loaded_def['models'][0]['service']['start_command']
        print(f"  Start command preserved: {original_start == loaded_start}")
        
        print(f"  Model definition successfully written to and read from: {temp_path}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

def main():
    """Main function."""
    test_url_parsing()
    test_blob_url_parsing()
    test_model_definition_generation()
    test_write_model_definition()
    return 0

if __name__ == "__main__":
    sys.exit(main())
