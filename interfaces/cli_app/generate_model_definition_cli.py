#!/usr/bin/env python3
"""
Model Definition Generator CLI

This script generates a model definition YAML file for a RAG service based on a GitHub URL.
It provides a command-line interface to the model definition generation functionality.

Usage:
    python generate_model_definition_cli.py --github-url https://github.com/owner/repo
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import structlog
from dotenv import load_dotenv

# Import the model definition generator functionality
from interfaces.portal.generate_model_definition import (
    parse_github_url,
    generate_model_name,
    generate_docs_name,
    write_model_definition
)
from config.config import load_config, BACKEND_MODEL_PATH

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    # Load configuration to use as defaults
    config = load_config()
    
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
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the model definition file (if not provided, uses config default)",
        default=None,
    )
    
    # Service ID (optional, will be generated if not provided)
    parser.add_argument(
        "--service-id",
        type=str,
        help="Service ID for the RAG service (will be generated if not provided)",
        default=None,
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
        default=8000,
    )
    
    # Service type
    parser.add_argument(
        "--service-type",
        type=str,
        help="Type of service (gradio or fastapi)",
        choices=["gradio", "fastapi"],
        default="gradio",
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
        return True
        
    except Exception as e:
        logger.error("Error setting up environment", error=str(e))
        print(f"Error setting up environment: {str(e)}")
        return False


def generate_model_definition_local(github_url: str, model_name: str, port: int, service_type: str, service_id: str) -> Dict:
    """
    Generate a model definition for the RAG service - local version that doesn't rely on PathConfig.backend_model_path.
    
    Args:
        github_url: GitHub URL
        model_name: Model name
        port: Port number
        service_type: Service type (gradio or fastapi)
        service_id: Service ID
        
    Returns:
        Model definition as a dictionary
    """
    # Parse the GitHub URL
    owner, repo, branch, path = parse_github_url(github_url)
    
    # Determine the docs path argument
    docs_path_arg = path if path else ""
    
    # Build the service-specific paths using the environment variable
    backend_model_path = str(BACKEND_MODEL_PATH)
    service_dir_path = f"{backend_model_path}/RAGModelService/rag_services/{service_id}"
    indices_path = f"{service_dir_path}/indices"
    docs_path = f"{service_dir_path}/docs"
    
    # Determine the start command based on service type
    if service_type == 'gradio':
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/cli_app/launch_gradio.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    else:  # fastapi
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/fastapi_app/fastapi_server.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
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
                                'command': ['/bin/bash', f'{backend_model_path}/RAGModelService/deployment/scripts/setup_gradio.sh']
                            }
                        }
                    ],
                    'start_command': start_command
                }
            }
        ]
    }
    
    return model_definition


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    if not setup_environment():
        return 1
    
    try:
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        # Parse the GitHub URL
        github_url = args.github_url
        owner, repo, branch, path = parse_github_url(github_url)
        
        # Generate names
        model_name = generate_model_name(owner, repo, path, args.name_prefix)
        service_id = args.service_id if args.service_id else os.urandom(3).hex()
        yaml_name = f"model-definition-{service_id}.yml"
        
        # Generate the model definition using our local implementation
        logger.info(f"Generating model definition for {github_url}")
        model_def = generate_model_definition_local(github_url, model_name, args.port, args.service_type, service_id)
        
        # Resolve output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            print(f"Using provided output directory: {output_dir}")
        else:
            # Use the project path with deployment/setup subdirectory
            output_dir = path_config.project_path / "deployment" / "setup"
            print(f"Using default output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the model definition to YAML file
        output_path = output_dir / yaml_name
        write_model_definition(model_def, output_path)
        
        logger.info(f"Model definition written to {output_path}")
        print(f"✅ Model definition successfully generated: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error("Error generating model definition", error=str(e))
        print(f"❌ Error generating model definition: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
