#!/usr/bin/env python3
"""
End-to-End RAG Service Creator

This script provides a complete solution for creating a RAG service from a GitHub repository.
It handles the entire process from cloning the repository to launching the web interface.

Features:
1. Clones a GitHub repository containing documentation
2. Processes the documentation to create vector embeddings
3. Launches a Gradio web interface for the RAG system
4. Supports service-specific configurations
5. Configurable LLM and retrieval parameters

Basic Usage:
    python create_rag_service.py --github-url https://github.com/owner/repo

With Custom Service ID:
    python create_rag_service.py --github-url https://github.com/owner/repo --service-id my_service_id

With Custom Paths:
    python create_rag_service.py --github-url https://github.com/owner/repo \
                                --output-dir ./my_service \
                                --docs-path ./my_docs \
                                --indices-path ./my_indices

With Custom Server Configuration:
    python create_rag_service.py --github-url https://github.com/owner/repo \
                                --host 127.0.0.1 \
                                --port 8080 \
                                --share

With Custom LLM Parameters:
    python create_rag_service.py --github-url https://github.com/owner/repo \
                                --openai-model gpt-4o-mini \
                                --temperature 0.3 \
                                --max-results 10

Environment Variables:
    OPENAI_API_KEY - OpenAI API key for LLM access
    OPENAI_BASE_URL - Optional custom OpenAI API endpoint
    RAG_SERVICE_PATH - Base path for RAG services
    HOST - Default host for the Gradio server
    PORT - Default port for the Gradio server
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import structlog
from dotenv import load_dotenv

# Import utility modules
from utils.github_utils import parse_github_url, validate_github_url, GitHubInfo
from utils.service_utils import (
    setup_environment, 
    create_service_config, 
    create_rag_service as create_service,
    ServiceConfig, UIConfig, ServicePaths, ServerConfig, LLMConfig
)

# Import core modules
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from core.retrieval import RetrievalEngine
from data.vector_store import VectorStore
from config.config import load_config, RetrievalSettings, LLMConfig as ConfigLLMConfig, ServerConfig as ConfigServerConfig, ChunkingSettings
from interfaces.gradio_app.gradio_app import create_gradio_interface
from core.document_processor import DocumentProcessor  # Import DocumentProcessor

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    # Load configuration to use as defaults
    config = load_config()
    llm_config = config.llm
    retrieval_settings = RetrievalSettings()  # Create a new instance instead of accessing from config
    
    parser = argparse.ArgumentParser(
        description="End-to-End RAG Service Creator"
    )
    
    # GitHub URL argument
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL to clone (e.g., https://github.com/owner/repo)",
        required=False,
    )
    
    # Service ID argument for processing existing services
    parser.add_argument(
        "--service-id",
        type=str,
        help="Process an existing service with the given ID (e.g., NVIDIA_TensorRT-LLM)",
        required=False,
    )
    
    # Docs path (optional, to use existing documentation)
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to existing documentation directory (skips cloning if provided)",
        default=None,
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the RAG service (default: from config)",
        default=None,
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        help=f"Host for the Gradio server (default: 0.0.0.0)",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"Port for the Gradio server (default: 7860)",
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
        help=f"OpenAI model to use for RAG (default: {llm_config.model_name})",
        default=llm_config.model_name,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=f"Temperature for LLM responses (default: {llm_config.temperature})",
        default=llm_config.temperature,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help=f"Maximum number of results to retrieve (default: {retrieval_settings.max_results})",
        default=retrieval_settings.max_results,
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


async def launch_service(docs_path: Path, indices_path: Path, config: ServiceConfig, service_id: Optional[str] = None) -> None:
    """
    Launch RAG service with Gradio interface.
    
    Args:
        docs_path: Path to documentation
        indices_path: Path to vector indices
        config: Service configuration
        service_id: Optional service ID for service-specific paths
    """
    try:
        # Load configuration
        app_config = load_config()
        path_config = app_config.paths
        
        # Update service_id if provided
        if service_id:
            path_config.service_id = service_id
        
        # Initialize LLMConfig
        llm_config = ConfigLLMConfig(
            openai_api_key=config.llm.openai_api_key,
            model_name=config.llm.model_name,
            temperature=config.llm.temperature,
            max_results=config.llm.max_results,
            streaming=True,
            memory_k=25,
            max_tokens=4096,
            max_tokens_per_doc=8000,
            filter_model=config.llm.model_name,
            base_url=os.environ.get("OPENAI_BASE_URL", "")
        )
        
        # Initialize retrieval settings
        retrieval_settings = RetrievalSettings(
            max_results=config.llm.max_results,
            filter_threshold=0.0
        )
        
        # Initialize VectorStore with configuration
        vector_store = VectorStore(
            docs_root=docs_path, 
            indices_path=indices_path,
            llm_config=llm_config,
            retrieval_settings=retrieval_settings,
            path_config=path_config,
            service_id=service_id
        )
        
        # Load vector indices
        await vector_store.load_index()
        
        if not vector_store.index:
            logger.error("Failed to load vector index", indices_path=str(indices_path))
            print(f"Error: Failed to load vector index from {indices_path}")
            return
        
        # Initialize LLMInterface
        llm_interface = LLMInterface(llm_config)
        
        # Initialize RetrievalEngine
        retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
        
        # Initialize RAGEngine
        rag_engine = RAGEngine(retrieval_engine, llm_interface)
        
        # Create Gradio interface
        interface = create_gradio_interface(rag_engine, docs_path)
        
        # Customize interface
        interface.title = config.ui.title
        
        # Launch the interface
        print(f"\nLaunching Gradio server with {config.llm.model_name} for RAG...")
        print(f"Documentation path: {docs_path}")
        print(f"Vector indices path: {indices_path}")
        print(f"Server: {config.server.host}:{config.server.port}")
        
        if config.server.share:
            print("Creating shareable link...")
        
        # Launch the server
        interface.launch(
            server_name=config.server.host,
            server_port=config.server.port,
            share=config.server.share,
            debug=config.server.debug,
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
        
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        # Update service_id if provided
        if args.service_id:
            path_config.service_id = args.service_id
            print(f"Using service ID: {args.service_id}")
        
        # Validate GitHub URL
        if args.github_url and not validate_github_url(args.github_url):
            logger.error("Invalid GitHub URL", url=args.github_url)
            print(f"Error: Invalid GitHub URL: {args.github_url}")
            return 1
        
        # Setup environment
        if not setup_environment():
            logger.error("Failed to set up environment")
            print("Error: Failed to set up environment")
            return 1
        
        # Resolve output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            print(f"Using provided output directory: {output_dir}")
        else:
            # Use the base RAG service directory from config
            output_dir = Path(path_config.rag_service_path)
            print(f"Using default output directory from config: {output_dir}")
        
        # Create service configuration
        if args.github_url:
            service_config = create_service_config(
                github_url=args.github_url,
                output_dir=str(output_dir),
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
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                existing_docs_path=args.docs_path,
            )
        else:
            # Create a minimal service config when only service_id is provided
            from utils.service_utils import UIConfig, ServicePaths
            
            # Parse service_id to extract owner and repo
            parts = args.service_id.split('_', 1)
            if len(parts) == 2:
                owner, repo = parts
            else:
                # Default values if service_id doesn't follow the expected format
                owner = "unknown"
                repo = args.service_id
            
            # Create a minimal GitHubInfo object
            github_info = GitHubInfo(
                url=f"https://github.com/{owner}/{repo}",
                owner=owner,
                repo=repo,
                branch="main",
                path=""
            )
            
            # Create LLMConfig from utils.service_utils
            llm_config = LLMConfig(
                model_name=args.openai_model,
                temperature=args.temperature,
                max_results=args.max_results,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "")
            )
            
            # Create ServerConfig
            server_config = ServerConfig(
                host=args.host,
                port=args.port,
                share=args.share,
                debug=args.debug
            )
            
            service_config = ServiceConfig(
                github=github_info,
                paths=ServicePaths(
                    output_dir=Path(output_dir),
                    docs_dir=Path(output_dir) / args.service_id / "docs",
                    indices_dir=Path(output_dir) / args.service_id / "indices",
                ),
                server=server_config,
                llm=llm_config,
                ui=UIConfig(
                    title=args.title,
                    description=args.description,
                    suggested_questions=args.suggested_questions
                )
            )
        
        # Create RAG service
        if args.service_id:
            # Process existing service
            docs_path = Path(path_config.rag_service_path) / args.service_id / "docs"
            indices_path = Path(path_config.rag_service_path) / args.service_id / "indices"
            error = None
        else:
            # Create new service
            docs_path, indices_path, error = await create_service(
                service_config
            )
        
        if error:
            logger.error("Failed to create RAG service", error=str(error))
            print(f"Error: {str(error)}")
            return 1
            
        # Extract service_id from the GitHub repository name
        if not args.service_id:
            service_id = f"{service_config.github.owner}_{service_config.github.repo}"
        else:
            service_id = args.service_id
        
        # Process documents and create vector indices
        print("\nProcessing documents to create vector indices...")
        try:
            # Initialize document processor
            doc_processor = DocumentProcessor(
                docs_root=docs_path,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Initialize vector store
            vector_store = VectorStore(
                docs_root=docs_path, 
                indices_path=indices_path,
                llm_config=ConfigLLMConfig(),
                path_config=path_config,
                service_id=service_id
            )
            
            # Define a file filter for markdown files
            file_filter = lambda p: p.is_file() and p.suffix.lstrip(".").lower() == "md"
            
            # Collect documents using DocumentProcessor
            documents = await doc_processor.collect_documents(
                directory=docs_path,
                recursive=True,
                chunk=True,
                file_filter=file_filter
            )
            
            if not documents:
                logger.warning("No documents found to process", docs_path=str(docs_path))
                print(f"Warning: No documents found to process in {docs_path}")
            else:
                print(f"Creating vector indices for {len(documents)} documents...")
                
                # Create vector indices
                await vector_store.create_indices(documents)
                print(f"Successfully processed {len(documents)} documents")
        except Exception as e:
            logger.error("Error processing documents", error=str(e))
            print(f"Error processing documents: {str(e)}")
            # Continue with the service creation even if indexing fails
        
        print(f"\nRAG service created successfully!")
        print(f"Service ID: {service_id}")
        print(f"Documentation: {docs_path}")
        print(f"Vector indices: {indices_path}")
        
        # Launch the service unless skip-launch flag is set
        if not args.skip_launch:
            await launch_service(docs_path, indices_path, service_config, service_id)
            
        return 0
            
    except Exception as e:
        logger.exception("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)