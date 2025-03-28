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
    generate_model_definition,
    write_model_definition
)
from config.config import load_config, PathConfig

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
        
        # Generate the model definition
        logger.info(f"Generating model definition for {github_url}")
        model_def = generate_model_definition(github_url, model_name, args.port, args.service_type)
        
        # Resolve output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            print(f"Using provided output directory: {output_dir}")
        else:
            # Use the deployment directory from config
            output_dir = path_config.base_path / "deployment" / "setup"
            print(f"Using default output directory from config: {output_dir}")
        
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
