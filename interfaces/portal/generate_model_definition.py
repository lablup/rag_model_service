#!/usr/bin/env python3
"""
Generate Model Definition

This script generates a model definition YAML file for a RAG service based on a GitHub URL.
It extracts the repository name from the URL to use as the documentation name.

Usage:
    python generate_model_definition.py --github-url https://github.com/owner/repo
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import configuration
from config.config import load_config

# Load configuration
config = load_config()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate model definition YAML for RAG service"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Output file
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the model definition file",
        default=".",
    )
    
    # Model name prefix
    parser.add_argument(
        "--name-prefix",
        type=str,
        help="Prefix for the model name",
        default="RAG Service for",
    )
    
    # Port
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the service",
        default=None,
    )
    
    # Service type
    parser.add_argument(
        "--service-type",
        type=str,
        help="Type of service (gradio or fastapi)",
        choices=["gradio", "fastapi", "Gradio UI", "FastAPI Server"],
        default=None,
    )
    
    return parser.parse_args()


def parse_github_url(github_url: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Parse a GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple of (owner, repo, branch, path)
    """
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Match GitHub URL with optional branch and path (tree format)
    tree_pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    tree_match = re.match(tree_pattern, github_url)
    
    if tree_match:
        owner = tree_match.group(1)
        repo = tree_match.group(2)
        branch = tree_match.group(3)  # This will be None if branch is not specified
        path = tree_match.group(4)    # This will be None if path is not specified
        return owner, repo, branch, path
    
    # Match GitHub URL with blob format
    blob_pattern = r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
    blob_match = re.match(blob_pattern, github_url)
    
    if blob_match:
        owner = blob_match.group(1)
        repo = blob_match.group(2)
        branch = blob_match.group(3)
        file_path = blob_match.group(4)
        
        # For blob URLs, extract the directory path
        # First, check if the file path contains a directory
        if '/' in file_path:
            # Get the directory part of the path (everything before the last slash)
            dir_path = '/'.join(file_path.split('/')[:-1])
            return owner, repo, branch, dir_path
        else:
            # It's a file in the root directory
            return owner, repo, branch, None
    
    # Basic GitHub URL (just owner/repo)
    basic_pattern = r"https?://github\.com/([^/]+)/([^/]+)"
    basic_match = re.match(basic_pattern, github_url)
    
    if basic_match:
        owner = basic_match.group(1)
        repo = basic_match.group(2)
        return owner, repo, None, None
    
    # If URL doesn't match any expected patterns
    raise ValueError(f"Invalid GitHub URL: {github_url}")


def generate_model_name(owner: str, repo: str, path: Optional[str], prefix: str) -> str:
    """
    Generate a model name based on the GitHub URL components.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        path: Path within the repository (if any)
        prefix: Prefix for the model name
        
    Returns:
        Model name
    """
    # Use the full repository name
    repo_name = repo
    
    # Format model name
    if path:
        if path == "docs":
            return f"{prefix} {repo_name} Documentation"
        else:
            return f"{prefix} {repo_name} Documentation ({path})"
    else:
        return f"{prefix} {repo_name}"


def generate_docs_name(owner: str, repo: str, path: Optional[str]) -> str:
    """
    Generate a documentation name for the YAML filename based on the GitHub URL components.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        path: Path within the repository (if any)
        
    Returns:
        Documentation name
    """
    # Use the full repository name (lowercase)
    repo_name = repo.lower()
    
    # Format docs name
    if path:
        # Replace slashes with hyphens and remove any special characters
        path_part = re.sub(r'[^a-zA-Z0-9-]', '', path.replace('/', '-'))
        return f"{repo_name}-{path_part}"
    else:
        return repo_name


def generate_model_definition(github_url: str, model_name: str, port: int = None, service_type: str = None, service_id: str = None) -> Dict:
    """
    Generate a model definition for the RAG service.
    
    Args:
        github_url: GitHub URL
        model_name: Model name
        port: Port number (if None, will use the value from config)
        service_type: Service type (gradio or fastapi) (if None, will use the value from config)
        service_id: Service ID (if None, will be generated from GitHub URL)
        
    Returns:
        Model definition as a dictionary
    """
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Get the current MAX_RESULTS from environment
    max_results = os.environ.get("MAX_RESULTS", "5")
    base_model_name = os.environ.get("BASE_MODEL_NAME", config.llm.model_name)
    base_url = os.environ.get("BASE_URL", config.llm.base_url)

    # Use provided values or defaults from config
    if port is None:
        # Use server config instead of service config
        port = config.server.port if hasattr(config, 'server') else 8000
    
    if service_type is None:
        # Default to gradio if not specified
        service_type = "gradio"
    
    # Parse the GitHub URL
    owner, repo, branch, path = parse_github_url(github_url)
    
    # Determine the docs path argument
    docs_path_arg = path if path else ""
    
    # Use the provided service_id or generate one from the GitHub URL
    if service_id is None:
        # For backward compatibility, but this should not be used
        service_id = f"{owner}/{repo}"
    
    # Update path configuration with service ID
    path_config.service_id = service_id
    
    # Get BACKEND_MODEL_PATH from environment variable or use a default
    backend_model_path = os.environ.get("BACKEND_MODEL_PATH", "/models")
    
    # Get RAG_SERVICE_PATH from environment variable or use a default
    rag_service_path = os.environ.get("RAG_SERVICE_PATH", f"{backend_model_path}/RAGModelService/rag_services/")
    
    # Ensure rag_service_path ends with a slash
    if not rag_service_path.endswith('/'):
        rag_service_path += '/'
    
    # Build the service-specific paths using the configuration
    service_dir_path = f"{rag_service_path}{service_id}"
    indices_path = f"{service_dir_path}/indices"
    docs_path = f"{service_dir_path}/docs"
    
    # Determine the start command based on service type
    if service_type in ['gradio', 'Gradio UI']:
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/cli_app/launch_gradio.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--max-results',
            str(max_results),
            '--base_model_name',
            base_model_name,
            '--base_url',
            base_url,
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    elif service_type in ['fastapi', 'FastAPI Server']:  # fastapi
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/fastapi_app/fastapi_server.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--max-results',
            str(max_results),
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    
    # Create the model definition
    model_definition = {
        'models': [
            {
                'name': model_name,
                'model_path': '/models',
                'service': {
                    'port': port,
                    'pre_start_actions': [
                        {
                            'action': 'run_command',
                            'args': {
                                'command': ['/bin/bash', f'{backend_model_path}/RAGModelService/deployment/scripts/setup_{"gradio" if service_type in ["gradio", "Gradio UI"] else "fastapi"}.sh']
                            }
                        }
                    ],
                    'start_command': start_command
                }
            }
        ]
    }
    
    return model_definition


def write_model_definition(model_definition: Dict, output_path: Path) -> None:
    """
    Write model definition to YAML file.
    
    Args:
        model_definition: Model definition dictionary
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        yaml.dump(model_definition, f, default_flow_style=False)
    
    print(f"Model definition written to {output_path}")


def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Parse GitHub URL
        owner, repo, branch, path = parse_github_url(args.github_url)
        
        # Generate model name
        model_name = generate_model_name(owner, repo, path, args.name_prefix)
        
        # Generate docs name for the filename
        docs_name = generate_docs_name(owner, repo, path)
        
        # Generate model definition
        model_definition = generate_model_definition(
            args.github_url,
            model_name,
            args.port,
            args.service_type
        )
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file name
        output_path = output_dir / f"model-definition-{docs_name}.yaml"
        
        # Write model definition to file
        write_model_definition(model_definition, output_path)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())