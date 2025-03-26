#!/usr/bin/env python3
"""
Gradio Server Launcher for RAG Service

This script:
1. Takes vector store and documentation paths
2. Configures and initializes the RAG system
3. Launches a Gradio web interface

Usage:
    python launch_gradio.py --indices-path ./embedding_indices --docs-path ./github_docs

Advanced Options:
    python launch_gradio.py --indices-path ./my_indices \
                       --docs-path ./my_docs \
                       --host 127.0.0.1 \
                       --port 8080 \
                       --share \
                       --openai-model gpt-4o \
                       --temperature 0.2 \
                       --max-results 5 \
                       --title "My Custom Documentation Assistant" \
                       --description "Search through project documentation" \
                       --suggested-questions "How do I install?" "What are the features?"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
from dotenv import load_dotenv
import gradio as gr

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from vectordb_manager.vectordb_manager import VectorDBManager
from app.rag_chatbot import LLMConfig, RAGManager
from app.gradio_app import create_gradio_interface

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gradio Server Launcher for RAG Service"
    )
    
    # Paths
    parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to vector indices",
        default="./embedding_indices",
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to documentation directory",
        default="./github_docs",
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
    
    # Customization
    parser.add_argument(
        "--title",
        type=str,
        help="Title for the Gradio interface",
        default=None,
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description for the Gradio interface",
        default="Documentation search with vector database",
    )
    parser.add_argument(
        "--suggested-questions",
        type=str,
        nargs="+",
        help="Suggested questions to display in the UI",
        default=None,
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
        
        # Check for OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            print("Error: OpenAI API key is required. Please set it in a .env file or as an environment variable.")
            return False
        
        return True
        
    except Exception as e:
        logger.error("Error setting up environment", error=str(e))
        print(f"Error setting up environment: {str(e)}")
        return False


def configure_rag_system(args) -> Dict:
    """
    Configure the RAG system based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Resolve paths
    indices_path = Path(args.indices_path)
    docs_path = Path(args.docs_path)
    
    # Validate paths
    if not indices_path.exists():
        logger.warning("Vector indices path does not exist", path=str(indices_path))
        print(f"Warning: Vector indices path does not exist: {indices_path}")
        print("Creating directory...")
        indices_path.mkdir(parents=True, exist_ok=True)
    
    if not docs_path.exists():
        logger.warning("Documentation path does not exist", path=str(docs_path))
        print(f"Warning: Documentation path does not exist: {docs_path}")
        print("Creating directory...")
        docs_path.mkdir(parents=True, exist_ok=True)
    
    # Create configuration dictionary
    config = {
        "paths": {
            "indices_path": indices_path,
            "docs_path": docs_path,
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "share": args.share,
        },
        "llm": {
            "model_name": args.openai_model,
            "temperature": args.temperature,
            "max_results": args.max_results,
            "streaming": True,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
        "ui": {
            "title": args.title or f"{args.openai_model} Documentation Assistant",
            "description": args.description,
            "suggested_questions": args.suggested_questions or [
                "What are the main features?",
                "How do I install this?",
                "What configuration options are available?",
                "How do I contribute to this project?",
                "What license is this project under?",
            ],
        },
    }
    
    return config


async def initialize_server(config: Dict) -> Tuple[VectorDBManager, RAGManager]:
    """
    Initialize the server components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of VectorDBManager and RAGManager
    """
    try:
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(
            docs_root=config["paths"]["docs_path"],
            indices_path=config["paths"]["indices_path"]
        )
        
        # Load vector indices
        await vector_manager.load_index()
        
        if not vector_manager.index:
            logger.warning(
                "No vector index found, trying to create one",
                docs_path=str(config["paths"]["docs_path"]),
                indices_path=str(config["paths"]["indices_path"])
            )
            print("No vector index found. Attempting to create one from documentation...")
            
            # Collect documents
            documents = await vector_manager.collect_documents()
            
            if not documents:
                logger.error("No documents found for indexing")
                print(f"Error: No documents found in {config['paths']['docs_path']}. Please check the path.")
                return None, None
            
            # Create indices
            await vector_manager.create_indices(documents)
            
            # Load the newly created index
            await vector_manager.load_index()
            
            if not vector_manager.index:
                logger.error("Failed to create vector index")
                print("Error: Failed to create vector index.")
                return None, None
        
        # Initialize LLMConfig
        llm_config = LLMConfig(
            openai_api_key=config["llm"]["openai_api_key"],
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_results=config["llm"]["max_results"],
            streaming=config["llm"]["streaming"],
        )
        
        # Initialize RAGManager
        rag_manager = RAGManager(llm_config, vector_manager)
        
        return vector_manager, rag_manager
        
    except Exception as e:
        logger.error("Error initializing server", error=str(e))
        print(f"Error initializing server: {str(e)}")
        return None, None


def customize_gradio_interface(interface: gr.Blocks, config: Dict) -> gr.Blocks:
    """
    Apply customization to the Gradio interface.
    
    Args:
        interface: Gradio interface
        config: Configuration dictionary
        
    Returns:
        Customized Gradio interface
    """
    interface.title = config["ui"]["title"]
    
    # Change suggested questions if provided
    if "suggested_questions" in config["ui"]:
        # Find Markdown elements containing "Suggested Questions"
        for component in interface.blocks.values():
            if isinstance(component, gr.Markdown) and "Suggested Questions" in component.value:
                component.value = "### Suggested Questions"
    
    return interface


async def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Configure RAG system
        config = configure_rag_system(args)
        
        # Initialize server components
        vector_manager, rag_manager = await initialize_server(config)
        
        if not vector_manager or not rag_manager:
            return 1
        
        # Create Gradio interface
        interface = create_gradio_interface(rag_manager, docs_path=config['paths']['docs_path'])
        
        # Apply customization
        interface = customize_gradio_interface(interface, config)
        
        # Launch the interface
        print(f"\nLaunching Gradio server with {config['llm']['model_name']} for RAG...")
        print(f"Documents path: {config['paths']['docs_path']}")
        print(f"Vector indices path: {config['paths']['indices_path']}")
        
        # Set up suggested questions if provided through arguments
        if args.suggested_questions:
            suggestion_questions = args.suggested_questions
            print(f"Using custom suggested questions: {suggestion_questions}")
        
        # Launch the server
        interface.launch(
            server_name=config["server"]["host"],
            server_port=config["server"]["port"],
            share=config["server"]["share"],
            debug=True,
        )
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
