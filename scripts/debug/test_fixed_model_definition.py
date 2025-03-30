#!/usr/bin/env python3
"""
Test Fixed Model Definition Generator

This script tests the updated model definition generator with various configurations
to ensure proper parameter formatting for Backend.AI compatibility.
"""

import os
import sys
from pathlib import Path
import yaml
import json
import tempfile

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the function to test
from interfaces.portal.generate_model_definition import generate_model_definition, write_model_definition

def print_command_args(start_command_args):
    """Pretty print command args for inspection."""
    print("Command arguments:")
    for i in range(0, len(start_command_args), 2):
        if i + 1 < len(start_command_args):
            print(f"  {start_command_args[i]}: {start_command_args[i+1]}")
        else:
            print(f"  {start_command_args[i]}")

def test_gradio_model_definition():
    """Test Gradio model definition generation with various configurations."""
    print("\n=== Testing Gradio Model Definition Generation ===")
    
    # Test case 1: Basic configuration
    github_url = "https://github.com/lablup/backend.ai"
    model_name = "RAG Service for Backend.AI"
    port = 8000
    service_type = "gradio"
    service_id = "test123"
    
    print(f"\nTest Case 1: Basic Gradio Configuration")
    print(f"  GitHub URL: {github_url}")
    print(f"  Model Name: {model_name}")
    print(f"  Port: {port}")
    print(f"  Service Type: {service_type}")
    print(f"  Service ID: {service_id}")
    
    model_def = generate_model_definition(github_url, model_name, port, service_type, service_id)
    
    print(f"  Model Definition Generated Successfully")
    
    # Print the start command for inspection
    start_command = model_def['models'][0]['service']['start_command']
    print_command_args(start_command)
    
    # Write the model definition to a temporary file for inspection
    with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as temp_file:
        temp_path = temp_file.name
        write_model_definition(model_def, temp_path)
        print(f"  Written to temporary file: {temp_path}")
    
    # Test case 2: With base_url
    print("\nTest Case 2: Gradio with base_url")
    os.environ["BASE_URL"] = "https://api.openai.com/v1"
    
    model_def = generate_model_definition(github_url, model_name, port, service_type, service_id)
    
    # Print the start command for inspection
    start_command = model_def['models'][0]['service']['start_command']
    print_command_args(start_command)
    
    # Test case 3: With different port
    print("\nTest Case 3: Gradio with different port")
    port = 9000
    
    model_def = generate_model_definition(github_url, model_name, port, service_type, service_id)
    
    # Print the start command for inspection
    start_command = model_def['models'][0]['service']['start_command']
    print_command_args(start_command)
    print(f"  Service Port in Model Definition: {model_def['models'][0]['service']['port']}")

def test_fastapi_model_definition():
    """Test FastAPI model definition generation."""
    print("\n=== Testing FastAPI Model Definition Generation ===")
    
    github_url = "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs"
    model_name = "RAG Service for TensorRT-LLM"
    port = 8080
    service_type = "fastapi"
    service_id = "test456"
    
    print(f"\nTest Case 4: Basic FastAPI Configuration")
    print(f"  GitHub URL: {github_url}")
    print(f"  Model Name: {model_name}")
    print(f"  Port: {port}")
    print(f"  Service Type: {service_type}")
    print(f"  Service ID: {service_id}")
    
    model_def = generate_model_definition(github_url, model_name, port, service_type, service_id)
    
    print(f"  Model Definition Generated Successfully")
    
    # Print the start command for inspection
    start_command = model_def['models'][0]['service']['start_command']
    print_command_args(start_command)

def test_yaml_format():
    """Test the YAML format of the generated model definition."""
    print("\n=== Testing YAML Format ===")
    
    github_url = "https://github.com/pytorch/pytorch/tree/master/docs/source"
    model_name = "RAG Service for PyTorch"
    port = 8000
    service_type = "gradio"
    service_id = "test789"
    
    model_def = generate_model_definition(github_url, model_name, port, service_type, service_id)
    
    print(f"  Model Definition Generated Successfully")
    
    # Write to a temporary file for inspection
    with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as temp_file:
        temp_path = temp_file.name
        write_model_definition(model_def, temp_path)
    
    # Read the raw YAML content
    with open(temp_path, 'r') as f:
        yaml_content = f.read()
    
    print("\nYAML Content Preview:")
    print("---")
    print("\n".join(yaml_content.split("\n")[:20]))  # Print first 20 lines
    print("...")
    
    # Verify the YAML can be parsed
    try:
        parsed_model_def = yaml.safe_load(yaml_content)
        print("\nYAML successfully parsed!")
        
        # Verify the structure
        assert 'models' in parsed_model_def, "Missing 'models' key"
        assert len(parsed_model_def['models']) > 0, "No models defined"
        assert 'service' in parsed_model_def['models'][0], "Missing 'service' key"
        assert 'start_command' in parsed_model_def['models'][0]['service'], "Missing 'start_command' key"
        
        print("YAML structure validation passed!")
    except Exception as e:
        print(f"Error parsing YAML: {e}")
    finally:
        # Clean up
        os.unlink(temp_path)

def test_empty_parameters():
    """Test model definition generation with empty parameters."""
    print("\n=== Testing Empty Parameters Handling ===")
    
    github_url = "https://github.com/lablup/backend.ai"
    model_name = "RAG Service for Backend.AI"
    port = 8000
    service_type = "gradio"
    service_id = "test123"
    
    # Save original environment variables
    original_base_model = os.environ.get("BASE_MODEL_NAME", "")
    original_base_url = os.environ.get("BASE_URL", "")
    
    try:
        # Set empty values for testing
        os.environ["BASE_MODEL_NAME"] = ""
        os.environ["BASE_URL"] = ""
        
        print("\nTest Case 5: Empty BASE_MODEL_NAME and BASE_URL")
        print(f"  GitHub URL: {github_url}")
        print(f"  Model Name: {model_name}")
        print(f"  Port: {port}")
        print(f"  Service Type: {service_type}")
        print(f"  Service ID: {service_id}")
        print(f"  BASE_MODEL_NAME: '{os.environ['BASE_MODEL_NAME']}'")
        print(f"  BASE_URL: '{os.environ['BASE_URL']}'")
        
        model_def = generate_model_definition(github_url, model_name, port, service_type, service_id)
        
        print(f"  Model Definition Generated Successfully")
        
        # Print the start command for inspection
        start_command = model_def['models'][0]['service']['start_command']
        print_command_args(start_command)
        
        # Check if base_model_name and base_url are NOT in the command
        command_str = " ".join(start_command)
        has_base_model = "--base_model_name" in command_str
        has_base_url = "--base_url" in command_str
        
        print(f"  Contains --base_model_name parameter: {has_base_model}")
        print(f"  Contains --base_url parameter: {has_base_url}")
        
        if not has_base_model and not has_base_url:
            print("  Test passed: Empty parameters are correctly omitted from the command")
        else:
            print("  Test failed: Empty parameters should be omitted from the command")
    
    finally:
        # Restore original environment variables
        if original_base_model:
            os.environ["BASE_MODEL_NAME"] = original_base_model
        else:
            os.environ.pop("BASE_MODEL_NAME", None)
            
        if original_base_url:
            os.environ["BASE_URL"] = original_base_url
        else:
            os.environ.pop("BASE_URL", None)

def main():
    """Main function."""
    # Save current environment variables
    original_env = os.environ.copy()
    
    try:
        # Set environment variables for testing
        os.environ["BACKEND_MODEL_PATH"] = "/models"
        os.environ["RAG_SERVICE_PATH"] = "/models/RAGModelService/rag_services/"
        os.environ["BASE_MODEL_NAME"] = "QwQ-32B"
        
        # Run tests
        test_gradio_model_definition()
        test_fastapi_model_definition()
        test_yaml_format()
        test_empty_parameters()
        
        print("\nAll tests completed successfully!")
        return 0
    finally:
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(original_env)

if __name__ == "__main__":
    sys.exit(main())
