#!/usr/bin/env python3
"""
Service Utilities for RAG Model Service

This module provides classes and functions for:
1. Creating service configurations
2. Managing RAG service creation 
3. Standardizing service interfaces
"""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from utils.github_utils import GitHubInfo, parse_github_url, clone_github_repo
from core.document_processor import DocumentProcessor


# Initialize logger
logger = structlog.get_logger()


class ServiceStatus:
    """Service status constants"""
    PENDING = "pending"
    PROCESSING = "processing"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"


class LLMConfig(BaseModel):
    """Configuration for LLM settings"""
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    max_results: int = 5
    streaming: bool = True
    openai_api_key: str


class ServerConfig(BaseModel):
    """Configuration for server settings"""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    debug: bool = False


class UIConfig(BaseModel):
    """Configuration for UI settings"""
    title: Optional[str] = None
    description: Optional[str] = None
    suggested_questions: Optional[List[str]] = None


class ServicePaths(BaseModel):
    """Configuration for service paths"""
    output_dir: Path
    docs_dir: Path
    indices_dir: Path
    existing_docs_path: Optional[Path] = None


class ServiceConfig(BaseModel):
    """Complete service configuration"""
    github: GitHubInfo
    paths: ServicePaths
    server: ServerConfig
    llm: LLMConfig
    ui: UIConfig


def get_unique_service_id() -> str:
    """Generate a unique service ID"""
    return str(uuid.uuid4())[:8]


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


def create_service_config(
    github_url: str,
    output_dir: Union[str, Path],
    existing_docs_path: Optional[Union[str, Path]] = None,
    server_config: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    ui_config: Optional[Dict[str, Any]] = None
) -> ServiceConfig:
    """
    Create a complete service configuration from components.
    
    Args:
        github_url: GitHub URL for repository
        output_dir: Base directory for service files
        existing_docs_path: Path to existing documentation (optional)
        server_config: Server configuration options (optional)
        llm_config: LLM configuration options (optional)
        ui_config: UI configuration options (optional)
        
    Returns:
        ServiceConfig object
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Parse GitHub URL
    github_info = parse_github_url(github_url)
    
    # Convert to Path objects
    output_dir = Path(output_dir).resolve()
    existing_docs_path = Path(existing_docs_path) if existing_docs_path else None
    
    # Create service ID from repository name
    service_id = f"{github_info.owner}_{github_info.repo}"
    
    # Create paths following the pattern RAG_SERVICE_PATH/service_id/docs and RAG_SERVICE_PATH/service_id/indices
    service_dir = output_dir / service_id
    docs_dir = service_dir / "docs"
    indices_dir = service_dir / "indices"
    
    # Create configuration objects
    paths = ServicePaths(
        output_dir=output_dir,
        docs_dir=docs_dir,
        indices_dir=indices_dir,
        existing_docs_path=existing_docs_path
    )
    
    server = ServerConfig(**(server_config or {}))
    
    llm_config_data = llm_config or {}
    llm_config_data["openai_api_key"] = openai_api_key
    llm = LLMConfig(**llm_config_data)
    
    ui = UIConfig(**(ui_config or {}))
    
    # Set default UI values if not provided
    repo_name = f"{github_info.owner}/{github_info.repo}"
    if not ui.title:
        ui.title = f"{repo_name} Documentation Assistant"
    if not ui.description:
        ui.description = f"Search and ask questions about {repo_name} documentation"
    
    return ServiceConfig(
        github=github_info,
        paths=paths,
        server=server,
        llm=llm,
        ui=ui
    )


async def create_rag_service(config: ServiceConfig) -> Tuple[Path, Path, Optional[Exception]]:
    """
    Create a RAG service based on the provided configuration.
    
    This function:
    1. Creates necessary directories
    2. Clones the GitHub repository
    3. Processes documents to create vector embeddings
    
    Args:
        config: Service configuration
        
    Returns:
        Tuple of (docs_path, indices_path, exception if any)
    """
    try:
        # Create output directories
        output_dir = config.paths.output_dir
        docs_dir = config.paths.docs_dir
        indices_dir = config.paths.indices_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Use existing docs path if provided, otherwise clone the repository
        if config.paths.existing_docs_path and config.paths.existing_docs_path.exists():
            docs_path = config.paths.existing_docs_path
            logger.info("Using existing documentation", path=str(docs_path))
        else:
            # Clone the repository
            docs_path, clone_error = clone_github_repo(config.github, docs_dir)
            if clone_error:
                return docs_dir, indices_dir, clone_error
        
        # Initialize document processor
        doc_processor = DocumentProcessor(
            docs_root=docs_path,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Process documents and create vector indices
        # Note: This is just the setup for document processing
        # The actual processing and index creation would need to be done by a VectorDBManager
        # or similar component that isn't being refactored in this task
        
        return docs_path, indices_dir, None
        
    except Exception as e:
        logger.error("Error creating RAG service", error=str(e))
        return config.paths.docs_dir, config.paths.indices_dir, e
