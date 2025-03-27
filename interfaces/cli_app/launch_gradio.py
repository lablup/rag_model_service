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
    python interfaces/cli_app/launch_gradio.py --indices-path ./embedding_indices \
                       --docs-path ./github_docs \
                       --host 127.0.0.1 \
                       --port 8080 \
                       --share \
                       --openai-model gpt-4o-mini \
                       --temperature 0.2 \
                       --max-results 15 \
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
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import components from the refactored structure
from config.config import LLMConfig, OpenAIConfig, RetrievalSettings
from core.llm import LLMInterface
from core.retrieval import RetrievalEngine
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from interfaces.gradio_app.gradio_app import create_gradio_interface

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
    indices_path = Path(args.indices_path).resolve()
    docs_path = Path(args.docs_path).resolve()
    
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
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
        "ui": {
            "title": args.title,
            "description": args.description,
            "suggested_questions": args.suggested_questions,
        },
    }
    
    return config


async def initialize_server(config: Dict) -> Tuple[VectorStore, RAGEngine]:
    """
    Initialize the server components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of VectorStore and RAGManager
    """
    try:
        # Extract paths from config
        indices_path = config["paths"]["indices_path"]
        docs_path = config["paths"]["docs_path"]
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            docs_root=docs_path,
            indices_path=indices_path
        )
        
        # Load vector indices
        logger.info("Loading vector indices...")
        await vector_store.load_index()
        
        # Initialize LLM settings
        llm_config = LLMConfig(
            openai_api_key=config["llm"]["openai_api_key"],
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            streaming=True,
            max_results=config["llm"]["max_results"],
        )
        
        # Initialize retrieval settings
        retrieval_settings = RetrievalSettings(
            max_results=config["llm"]["max_results"],
            docs_path=str(docs_path),
            indices_path=str(indices_path),
        )
        
        # Initialize components
        llm_interface = LLMInterface(llm_config)
        retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
        rag_engine = RAGEngine(retrieval_engine, llm_interface)
        
        return vector_store, rag_engine
        
    except Exception as e:
        logger.error("Error initializing server components", error=str(e))
        print(f"Error initializing server components: {str(e)}")
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
    # Apply title if provided
    if config["ui"]["title"]:
        # Find and update the title markdown
        for component in interface.blocks.values():
            if isinstance(component, gr.Markdown) and component.value.startswith("# "):
                component.value = f"# {config['ui']['title']}"
                break
    
    # Apply description if provided
    if config["ui"]["description"]:
        # Find and update the description markdown
        description_found = False
        for component in interface.blocks.values():
            if isinstance(component, gr.Markdown) and not component.value.startswith("#"):
                component.value = config["ui"]["description"]
                description_found = True
                break
        
        if not description_found:
            # If no description component found, add one
            for component in interface.blocks.values():
                if isinstance(component, gr.Markdown) and component.value.startswith("# "):
                    # Add description after title
                    interface.blocks.insert(
                        list(interface.blocks.keys()).index(id(component)) + 1,
                        gr.Markdown(config["ui"]["description"])
                    )
                    break
    
    # Apply suggested questions if provided
    if config["ui"]["suggested_questions"]:
        # This is more complex and would require modifying the interface structure
        # For now, we'll just log that custom questions were provided
        logger.info("Custom suggested questions provided", questions=config["ui"]["suggested_questions"])
    
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
        vector_store, rag_engine = await initialize_server(config)
        
        if not vector_store or not rag_engine:
            return 1
        
        # Create Gradio interface
        interface = create_gradio_interface(
            rag_engine=rag_engine, 
            docs_path=config['paths']['docs_path']
        )
        
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
