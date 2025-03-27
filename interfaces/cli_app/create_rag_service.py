#!/usr/bin/env python3
"""
End-to-End RAG Service Creator

This script:
1. Clones a GitHub repository containing documentation
2. Processes the documentation to create vector embeddings
3. Launches a Gradio web interface for the RAG system

Usage:
    python create_rag_service.py --github-url https://github.com/owner/repo
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import structlog
from dotenv import load_dotenv

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import utility modules
from utils.github_utils import parse_github_url, validate_github_url
from utils.service_utils import (
    setup_environment, 
    create_service_config, 
    create_rag_service as create_service
)

# Import core modules
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from config.config import LLMConfig
from interfaces.gradio_app.gradio_app import create_gradio_interface

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-End RAG Service Creator"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Docs path (optional, to use existing documentation)
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to existing documentation directory (skips cloning if provided)",
        default=None,
    )
    
    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the RAG service",
        default="./rag_service",
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        help="Host for the Gradio server",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the Gradio server",
        default=7860,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link",
    )
    
    # OpenAI settings
    parser.add_argument(
        "--openai-model",
        type=str,
        help="OpenAI model to use for RAG",
        default="gpt-4o",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM responses",
        default=0.2,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results to retrieve",
        default=5,
    )
    
    # UI customization
    parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the Gradio interface",
        default=None,
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Custom description for the Gradio interface",
        default=None,
    )
    parser.add_argument(
        "--suggested-questions",
        type=str,
        nargs="+",
        help="Custom suggested questions for the UI",
        default=None,
    )
    
    # Advanced options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Skip launching the Gradio interface (setup only)",
    )
    
    return parser.parse_args()


async def launch_service(docs_path: Path, indices_path: Path, config: Dict) -> None:
    """
    Launch RAG service with Gradio interface.
    
    Args:
        docs_path: Path to documentation
        indices_path: Path to vector indices
        config: Service configuration
    """
    try:
        # Initialize VectorStore
        vector_store = VectorStore(docs_path, indices_path)
        
        # Load vector indices
        await vector_store.load_index()
        
        if not vector_store.index:
            logger.error("Failed to load vector index", indices_path=str(indices_path))
            print(f"Error: Failed to load vector index from {indices_path}")
            return
        
        # Initialize LLMConfig
        llm_config = LLMConfig(
            openai_api_key=config["llm"]["openai_api_key"],
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_tokens=1024,
            streaming=True
        )
        
        # Initialize LLMInterface
        llm_interface = LLMInterface(llm_config)
        
        # Initialize RAGEngine
        rag_engine = RAGEngine(vector_store, llm_interface)
        
        # Create Gradio interface
        interface = create_gradio_interface(rag_engine, docs_path)
        
        # Customize interface
        interface.title = config["ui"]["title"]
        
        # Launch the interface
        print(f"\nLaunching Gradio server with {config['llm']['model_name']} for RAG...")
        print(f"Documentation path: {docs_path}")
        print(f"Vector indices path: {indices_path}")
        print(f"Server: {config['server']['host']}:{config['server']['port']}")
        
        if config['server']['share']:
            print("Creating shareable link...")
        
        # Launch the server
        interface.launch(
            server_name=config["server"]["host"],
            server_port=config["server"]["port"],
            share=config["server"]["share"],
            debug=config["server"]["debug"],
        )
        
    except Exception as e:
        logger.error("Error launching service", error=str(e))
        print(f"Error launching service: {str(e)}")
        raise


async def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate GitHub URL
        if not validate_github_url(args.github_url):
            logger.error("Invalid GitHub URL", url=args.github_url)
            print(f"Error: Invalid GitHub URL: {args.github_url}")
            return 1
        
        # Setup environment
        if not setup_environment():
            logger.error("Failed to set up environment")
            print("Error: Failed to set up environment")
            return 1
        
        # Create service configuration
        service_config = create_service_config(
            github_url=args.github_url,
            output_dir=args.output_dir,
            server_config={
                "host": args.host,
                "port": args.port,
                "share": args.share,
                "debug": args.debug,
            },
            llm_config={
                "model_name": args.openai_model,
                "temperature": args.temperature,
                "max_results": args.max_results,
            },
            ui_config={
                "title": args.title,
                "description": args.description,
                "suggested_questions": args.suggested_questions,
            },
        )
        
        # Create RAG service
        service_id, docs_path, indices_path = await create_service(
            service_config, 
            existing_docs_path=args.docs_path
        )
        
        if not service_id:
            logger.error("Failed to create RAG service")
            print("Error: Failed to create RAG service")
            return 1
        
        print(f"\nRAG service created successfully!")
        print(f"Service ID: {service_id}")
        print(f"Documentation: {docs_path}")
        print(f"Vector indices: {indices_path}")
        
        # Launch the service unless skip-launch flag is set
        if not args.skip_launch:
            await launch_service(docs_path, indices_path, service_config)
            
        return 0
            
    except Exception as e:
        logger.exception("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)