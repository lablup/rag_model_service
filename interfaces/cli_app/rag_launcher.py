#!/usr/bin/env python3
"""
RAG Launcher

A simple script to launch the complete RAG service system.

Usage:
    python rag_launcher.py [--portal-only]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
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
    
    return parser.parse_args()

def check_environment():
    """Check if the environment is properly set up."""
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            pass
    
    if not openai_api_key:
        print("Error: OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with the following content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Check for required scripts
    required_scripts = [
        "auto_rag_service/setup_rag.py",
        "auto_rag_service/launch_gradio.py",
        "auto_rag_service/create_rag_service.py",
        "auto_rag_service/rag_service_portal.py"
    ]
    
    for script in required_scripts:
        if not Path(script).exists():
            breakpoint()
            print(f"Error: Required script {script} not found.")
            return False
    
    return True

def launch_rag_portal(port):
    """Launch the RAG service portal."""
    print(f"Launching RAG service portal on port {port}...")
    
    # Run the portal script
    cmd = [
        sys.executable,
        "rag_service_portal.py",
        "--port", str(port),
    ]
    
    # Execute in the current process (blocking)
    os.execv(sys.executable, cmd)

def setup_example_repositories():
    """Set up example repositories."""
    print("Setting up example repositories...")
    
    example_repos = [
        ("fastai/fastdoc", "FastAI Documentation"),
        ("scikit-learn/scikit-learn", "Scikit-Learn"),
        ("huggingface/transformers", "Hugging Face Transformers"),
    ]
    
    for repo, name in example_repos:
        print(f"Setting up {name} repository...")
        
        # Create output directory
        output_dir = Path(f"./examples/{repo.split('/')[1]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run setup_rag.py
        cmd = [
            sys.executable,
            "auto_rag_service/setup_rag.py",
            "--github-url", f"https://github.com/{repo}",
            "--output-dir", str(output_dir),
            "--indices-path", str(output_dir / "indices"),
            "--skip-testing",
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {name} repository setup complete.")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error setting up {name} repository: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    if not check_environment():
        return 1
    
    # Create required directories
    Path("./rag_services").mkdir(exist_ok=True)
    
    if not args.portal_only:
        print("This script will:")
        print("1. Set up example repositories (if requested)")
        print("2. Launch the RAG service portal")
        
        setup_option = input("Would you like to set up example repositories? (y/n): ").lower()
        if setup_option == 'y':
            setup_example_repositories()
    
    # Launch the portal
    launch_rag_portal(args.port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())