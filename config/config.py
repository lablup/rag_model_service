"""
Configuration module for the RAG Model Service.
Loads configuration from environment variables and provides default values.
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator, field_validator

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Define deployment environments
class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""
    LOCAL = "local"
    REMOTE = "remote"
    PRODUCTION = "production"

# Get current deployment environment
DEPLOYMENT_ENV = DeploymentEnvironment(os.environ.get("DEPLOYMENT_ENV", DeploymentEnvironment.LOCAL))

# Get base directories from environment variables with fallbacks
def get_default_project_path() -> Path:
    """Get default project path based on deployment environment"""
    if DEPLOYMENT_ENV == DeploymentEnvironment.LOCAL:
        return Path(__file__).resolve().parent.parent
    elif DEPLOYMENT_ENV == DeploymentEnvironment.REMOTE:
        return Path("/home/work/auto_rag/RAGModelService")
    elif DEPLOYMENT_ENV == DeploymentEnvironment.PRODUCTION:
        return Path("/models/RAGModelService")
    return Path(__file__).resolve().parent.parent

# Base paths
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", str(get_default_project_path())))
BACKEND_MODEL_PATH = Path(os.environ.get("BACKEND_MODEL_PATH", "/models"))
RAG_SERVICE_PATH = os.environ.get("RAG_SERVICE_PATH", str(PROJECT_PATH / "rag_services"))
RAG_SERVICE_NAME = os.environ.get("RAG_SERVICE_NAME", "rag_document_service")
INDICES_PATH = Path(os.environ.get("INDICES_PATH", str(PROJECT_PATH / "embedding_indices")))

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


class PathConfig(BaseModel):
    """Path configuration with dynamic path resolution for service-specific paths."""
    project_path: Path = Field(default_factory=lambda: PROJECT_PATH)
    rag_service_path: str = Field(default_factory=lambda: RAG_SERVICE_PATH)
    indices_path: Path = Field(default_factory=lambda: INDICES_PATH)
    service_id: Optional[str] = None
    
    @field_validator("*", mode="before")
    @classmethod
    def validate_paths(cls, v, info):
        """Ensure all path values are Path objects except for string fields"""
        field_name = info.field_name
        if isinstance(v, str) and field_name not in ["rag_service_path", "service_id"]:
            return Path(v)
        return v
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_service_docs_path(self, service_id: Optional[str] = None) -> Path:
        """
        Get the documentation path for a specific service.
        
        Args:
            service_id: Optional service ID. If None, uses the configured service_id.
            
        Returns:
            Path to the service's documentation directory
        """
        sid = service_id or self.service_id
        if not sid:
            # If no service ID is provided, return a default path
            return self.project_path / "docs"
        
        # Construct the path to the service's docs directory
        return Path(self.rag_service_path) / sid / "docs"
    
    def get_service_indices_path(self, service_id: Optional[str] = None) -> Path:
        """
        Get the indices path for a specific service.
        
        Args:
            service_id: Optional service ID. If None, uses the configured service_id.
            
        Returns:
            Path to the service's indices directory
        """
        sid = service_id or self.service_id
        if not sid:
            # If no service ID is provided, return the default indices path
            return self.indices_path
        
        # Construct the path to the service's indices directory
        return Path(self.rag_service_path) / sid / "indices"
    
    def resolve_path(self, path: Union[str, Path], create_if_missing: bool = False) -> Path:
        """
        Resolve a path relative to the project path if it's not absolute.
        Optionally create the directory if it doesn't exist.
        
        Args:
            path: Path to resolve
            create_if_missing: Whether to create the directory if it doesn't exist
            
        Returns:
            Resolved path
        """
        path_obj = Path(path)
        resolved_path = path_obj if path_obj.is_absolute() else self.project_path / path_obj
        
        if create_if_missing and not resolved_path.exists():
            logger.info(f"Creating directory: {resolved_path}")
            resolved_path.mkdir(parents=True, exist_ok=True)
            
        return resolved_path
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments
        """
        # Only update if argument is provided and not None
        if args.get("service_id"):
            self.service_id = args["service_id"]
        
        if args.get("indices_path"):
            self.indices_path = Path(args["indices_path"])
            
        if args.get("rag_service_path"):
            self.rag_service_path = args["rag_service_path"]


class LLMConfig(BaseModel):
    """Configuration for LLM"""
    model_config = {"protected_namespaces": ()}
    
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o"))
    max_tokens: int = Field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "2048")))
    temperature: float = Field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.2")))
    streaming: bool = True
    memory_k: int = Field(default_factory=lambda: int(os.environ.get("MEMORY_K", "25")))
    max_results: int = Field(default_factory=lambda: int(os.environ.get("MAX_RESULTS", "5")))
    max_tokens_per_doc: int = 8000  # New: limit tokens per document
    filter_model: str = Field(default_factory=lambda: os.environ.get("FILTER_MODEL", "gpt-4o"))
    base_url: str = Field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", ""))
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments
        """
        if args.get("openai_model"):
            self.model_name = args["openai_model"]
            
        if args.get("temperature") is not None:
            self.temperature = args["temperature"]
            
        if args.get("max_tokens") is not None:
            self.max_tokens = args["max_tokens"]
            
        if args.get("max_results") is not None:
            self.max_results = args["max_results"]
            
        if args.get("base_url"):
            self.base_url = args["base_url"]


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    model_config = {"protected_namespaces": ()}
    
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o"))
    temperature: float = Field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.2")))


class LLMSettings(BaseModel):
    """Configuration settings for LLM interface."""
    model_config = {"protected_namespaces": ()}
    
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o"))
    max_tokens: int = Field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "2048")))
    temperature: float = Field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.2")))
    streaming: bool = True
    memory_k: int = Field(default_factory=lambda: int(os.environ.get("MEMORY_K", "25")))
    base_url: Optional[str] = Field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", None))


class RetrievalSettings(BaseModel):
    """Configuration settings for retrieval engine."""
    max_results: int = Field(default_factory=lambda: int(os.environ.get("MAX_RESULTS", "5")))
    max_tokens_per_doc: int = Field(default_factory=lambda: int(os.environ.get("MAX_TOKENS_PER_DOC", "8000")))
    filter_threshold: float = Field(default_factory=lambda: float(os.environ.get("FILTER_THRESHOLD", "0.7")))
    docs_path: Optional[str] = None
    indices_path: Optional[str] = None
    service_id: Optional[str] = None
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments
        """
        if args.get("docs_path"):
            self.docs_path = args["docs_path"]
            
        if args.get("indices_path"):
            self.indices_path = args["indices_path"]
            
        if args.get("service_id"):
            self.service_id = args["service_id"]
            
        if args.get("max_results") is not None:
            self.max_results = args["max_results"]

class ChunkingSettings(BaseModel):
    """Configuration settings for chunking."""
    chunk: bool = Field(default_factory=lambda: bool(os.environ.get("CHUNK", "true")))
    chunk_size: int = Field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "200")))

class RAGConfig(BaseModel):
    """RAG-specific configuration."""
    max_results: int = Field(default_factory=lambda: int(os.environ.get("MAX_RESULTS", "5")))
    memory_k: int = Field(default_factory=lambda: int(os.environ.get("MEMORY_K", "25")))


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    share_enabled: bool = Field(default_factory=lambda: os.environ.get("SHARE", "false").lower() == "true")
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments
        """
        if args.get("host"):
            self.host = args["host"]
            
        if args.get("port") is not None:
            self.port = args["port"]
            
        if args.get("share") is not None:
            self.share_enabled = args["share"]


class AppConfig(BaseModel):
    """Main application configuration."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    chunking_settings: ChunkingSettings = Field(default_factory=ChunkingSettings)
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments
        """
        self.paths.update_from_args(args)
        self.llm.update_from_args(args)
        self.server.update_from_args(args)
        self.rag.update_from_args(args)
        self.chunking_settings.update_from_args(args)


def load_config() -> AppConfig:
    """Load and return the application configuration."""
    return AppConfig()


def load_config_from_args(args: Dict[str, Any]) -> AppConfig:
    """
    Load configuration and update from command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Updated application configuration
    """
    config = AppConfig()
    config.update_from_args(args)
    return config
