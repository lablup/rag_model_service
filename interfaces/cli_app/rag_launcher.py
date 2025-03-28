#!/usr/bin/env python3
"""
RAG Launcher

This script serves as the main entry point for launching the complete RAG service system.
It can set up example repositories and start the RAG service portal.

Features:
1. Environment validation for required dependencies
2. Optional setup of example repositories for demonstration
3. Launch of the RAG service portal web interface
4. Support for service-specific configurations
5. Configurable server parameters

Basic Usage:
    python rag_launcher.py

Portal Only (Skip Example Setup):
    python rag_launcher.py --portal-only

With Custom Port:
    python rag_launcher.py --port 8080

With Service ID:
    python rag_launcher.py --service-id my_service_id

Environment Variables:
    OPENAI_API_KEY - Required for LLM access
    RAG_SERVICE_PATH - Base path for RAG services
    PROJECT_PATH - Base project path
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from config.config import load_config, PathConfig

def parse_args():
    """Parse command line arguments."""
    # Load configuration to use as defaults
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Launch the RAG service system"
    )
    
    parser.add_argument(
        "--portal-only",
        action="store_true",
        help="Launch only the portal interface (without example repositories)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the portal interface",
    )
    
    parser.add_argument(
        "--service-id",
        type=str,
        help="Service ID for service-specific paths",
        default=None,
    )
    
    return parser.parse_args()

def check_environment():
    """Check if the environment is properly set up."""
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with the following content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Get the script directory
    script_dir = Path(__file__).resolve().parent
    
    # Check for required scripts
    required_scripts = [
        script_dir / "launch_gradio.py",
        script_dir / "vectorstore_cli.py",
        script_dir / "rag_cli.py"
    ]
    
    for script in required_scripts:
        if not script.exists():
            print(f"Error: Required script {script} not found.")
            return False
    
    return True

def launch_rag_portal(port, service_id=None):
    """Launch the RAG service portal."""
    print(f"Launching RAG service portal on port {port}...")
    
    # Get the script directory
    script_dir = Path(__file__).resolve().parent
    
    # Build command
    cmd = [
        sys.executable,
        str(script_dir / "launch_gradio.py"),
        "--port", str(port),
    ]
    
    # Add service_id if provided
    if service_id:
        cmd.extend(["--service-id", service_id])
    
    # Execute in the current process (blocking)
    os.execv(sys.executable, cmd)

def setup_example_repositories(service_id=None):
    """Set up example repositories."""
    print("Setting up example repositories...")
    
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Update service_id if provided
    if service_id:
        path_config.service_id = service_id
    
    # Get the script directory
    script_dir = Path(__file__).resolve().parent
    
    example_repos = [
        ("fastai/fastdoc", "FastAI Documentation"),
        ("scikit-learn/scikit-learn", "Scikit-Learn"),
        ("huggingface/transformers", "Hugging Face Transformers"),
    ]
    
    for repo, name in example_repos:
        print(f"Setting up {name} repository...")
        
        # Create output directory
        output_dir = path_config.get_service_docs_path(f"examples/{repo.split('/')[1]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create indices directory
        indices_dir = path_config.get_service_indices_path(f"examples/{repo.split('/')[1]}")
        indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Run github_cli.py to clone repository
        cmd = [
            sys.executable,
            str(script_dir / "github_cli.py"),
            "clone",
            f"https://github.com/{repo}",
            "--output-dir", str(output_dir),
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {name} repository cloned.")
            
            # Run vectorstore_cli.py to process documents
            cmd = [
                sys.executable,
                str(script_dir / "vectorstore_cli.py"),
                "process",
                "--docs-path", str(output_dir),
                "--indices-path", str(indices_dir),
            ]
            
            subprocess.run(cmd, check=True)
            print(f"✓ {name} repository processed and indexed.")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error setting up {name} repository: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    if not check_environment():
        return 1
    
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Update service_id if provided
    if args.service_id:
        path_config.service_id = args.service_id
    
    # Create required directories
    rag_services_dir = path_config.get_service_docs_path("rag_services")
    rag_services_dir.mkdir(exist_ok=True, parents=True)
    
    if not args.portal_only:
        print("This script will:")
        print("1. Set up example repositories (if requested)")
        print("2. Launch the RAG service portal")
        
        setup_option = input("Would you like to set up example repositories? (y/n): ").lower()
        if setup_option == 'y':
            setup_example_repositories(args.service_id)
    
    # Launch the portal
    launch_rag_portal(args.port, args.service_id)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())