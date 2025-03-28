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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
from dotenv import load_dotenv
import gradio as gr

# Import components from the refactored structure
from config.config import load_config, LLMConfig, OpenAIConfig, RetrievalSettings, PathConfig
from core.llm import LLMInterface
from core.retrieval import RetrievalEngine
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from interfaces.gradio_app.gradio_app import create_gradio_interface

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    # Load configuration first to use as defaults
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Gradio Server Launcher for RAG Service"
    )
    
    # Paths
    parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to vector indices (if not provided, uses config default)",
        default=None,
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to documentation directory (if not provided, uses config default)",
        default=None,
    )
    parser.add_argument(
        "--service-id",
        type=str,
        help="Service ID for service-specific paths",
        default=None,
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
        default=8000,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=True,
        help="Create a shareable link",
    )
    
    # OpenAI settings
    parser.add_argument(
        "--openai-model",
        type=str,
        help=f"OpenAI model to use for RAG (default: {config.llm.model_name})",
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=f"Temperature for LLM responses (default: {config.llm.temperature})",
        default=None,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help=f"Maximum number of results to retrieve (default: {config.rag.max_results})",
        default=None,
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
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Update service_id if provided
    if args.service_id:
        path_config.service_id = args.service_id
    
    # Resolve paths using config if not explicitly provided
    if args.docs_path:
        docs_path = Path(args.docs_path).resolve()
        print(f"Using provided docs path: {docs_path}")
    else:
        docs_path = path_config.get_service_docs_path(path_config.service_id)
        print(f"Using docs path from config: {docs_path}")
    
    if args.indices_path:
        indices_path = Path(args.indices_path).resolve()
        print(f"Using provided indices path: {indices_path}")
    else:
        indices_path = path_config.get_service_indices_path(path_config.service_id)
        print(f"Using indices path from config: {indices_path}")
    
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
    system_config = {
        "paths": {
            "indices_path": indices_path,
            "docs_path": docs_path,
            "path_config": path_config,
            "service_id": args.service_id or path_config.service_id,
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "share": args.share,
        },
        "llm": {
            "model": args.openai_model or config.llm.model_name,
            "temperature": args.temperature if args.temperature is not None else config.llm.temperature,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", config.llm.openai_api_key),
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", config.llm.openai_base_url),
        },
        "retrieval": {
            "max_results": args.max_results or config.rag.max_results,
            "max_tokens_per_doc": config.rag.max_tokens_per_doc,
            "filter_threshold": config.rag.filter_threshold,
        },
        "ui": {
            "title": args.title or "RAG Documentation Assistant",
            "description": args.description,
            "suggested_questions": args.suggested_questions or [
                "How do I install this project?",
                "What are the main features?",
                "How do I configure the system?",
            ],
        },
    }
    
    return system_config


async def initialize_server(config: Dict) -> Tuple[VectorStore, RAGEngine]:
    """
    Initialize the server components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of VectorStore and RAGManager
    """
    # Extract configuration
    paths = config["paths"]
    llm_config = config["llm"]
    retrieval_config = config["retrieval"]
    
    # Create LLM config
    llm_settings = LLMConfig(
        openai_api_key=llm_config["openai_api_key"],
        openai_base_url=llm_config["openai_base_url"],
        model_name=llm_config["model"],
        temperature=llm_config["temperature"],
        streaming=True,
    )
    
    # Create retrieval settings
    retrieval_settings = RetrievalSettings(
        max_results=retrieval_config["max_results"],
        max_tokens_per_doc=retrieval_config["max_tokens_per_doc"],
        filter_threshold=retrieval_config["filter_threshold"],
        docs_path=str(paths["docs_path"]),
        indices_path=str(paths["indices_path"]),
    )
    
    # Initialize vector store
    print(f"Initializing vector store with indices path: {paths['indices_path']}")
    vector_store = VectorStore(
        docs_root=paths["docs_path"],
        indices_path=paths["indices_path"],
        llm_config=llm_settings,
        retrieval_settings=retrieval_settings,
        path_config=paths["path_config"],
        service_id=paths["service_id"]
    )
    
    # Load vector index
    print("Loading vector index...")
    await vector_store.load_index()
    
    # Initialize LLM interface
    llm_interface = LLMInterface(llm_settings)
    
    # Initialize retrieval engine
    retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
    
    # Initialize RAG engine
    rag_engine = RAGEngine(retrieval_engine, llm_interface)
    
    return vector_store, rag_engine


def customize_gradio_interface(interface: gr.Blocks, config: Dict) -> gr.Blocks:
    """
    Apply customization to the Gradio interface.
    
    Args:
        interface: Gradio interface
        config: Configuration dictionary
        
    Returns:
        Customized Gradio interface
    """
    # Extract UI configuration
    ui_config = config["ui"]
    
    # Set title and description
    if hasattr(interface, "title"):
        interface.title = ui_config["title"]
    
    # Find the markdown component with the description
    for component in interface.blocks.values():
        if isinstance(component, gr.Markdown):
            if "description" in component.elem_id or "header" in component.elem_id:
                component.value = ui_config["description"]
                break
    
    # Find the suggested questions component
    for component in interface.blocks.values():
        if isinstance(component, gr.Examples) and hasattr(component, "examples"):
            component.examples = ui_config["suggested_questions"]
            break
    
    return interface


async def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set up environment (load .env, validate required vars)
        if not setup_environment():
            return 1
        
        # Configure the RAG system
        config = configure_rag_system(args)
        
        # Initialize server components
        vector_store, rag_engine = await initialize_server(config)
        
        # Create Gradio interface
        interface = create_gradio_interface(
            rag_engine=rag_engine,
            docs_path=config["paths"]["docs_path"],
            indices_path=config["paths"]["indices_path"],
            service_id=config["paths"]["service_id"]  # Pass service_id to create_gradio_interface
        )
        
        # Apply customization
        interface = customize_gradio_interface(interface, config)
        
        # Launch the server
        server_config = config["server"]
        interface.launch(
            server_name=server_config["host"],
            server_port=server_config["port"],
            share=server_config["share"],
            debug=True,
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e), exc_info=True)
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
