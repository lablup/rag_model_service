"""
Configuration module for the RAG Model Service.
Loads configuration from environment variables and provides default values.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables from .env file
load_dotenv()

# Constants
DOCS_PATH = Path(os.environ.get("DOCS_PATH", str(Path(__file__).resolve().parent.parent.parent / "rag_service" / "docs" / "docs")))

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



class LLMConfig(BaseModel):
    """Configuration for LLM"""

    openai_api_key: str
    model_name: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.2
    streaming: bool = True
    memory_k: int = 25
    max_results: int = 5  # Reduced from 20 to limit context size
    max_tokens_per_doc: int = 8000  # New: limit tokens per document
    filter_model: str = "gpt-4o"  # Model for document filtering
    base_url: str = ""  # Base URL for custom model endpoints



class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o"))
    temperature: float = Field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.2")))


class LLMSettings(BaseModel):
    """Configuration settings for LLM interface."""
    
    openai_api_key: str
    model_name: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.2
    streaming: bool = True
    memory_k: int = 25
    base_url: Optional[str] = None


class RetrievalSettings(BaseModel):
    """Configuration settings for retrieval engine."""
    
    max_results: int = 5
    max_tokens_per_doc: int = 8000
    filter_threshold: float = 0.7
    docs_path: Optional[str] = None
    indices_path: Optional[str] = None


class RAGConfig(BaseModel):
    """RAG-specific configuration."""
    max_results: int = Field(default_factory=lambda: int(os.environ.get("MAX_RESULTS", "5")))
    memory_k: int = Field(default_factory=lambda: int(os.environ.get("MEMORY_K", "25")))


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    share_enabled: bool = Field(default_factory=lambda: os.environ.get("SHARE", "false").lower() == "true")


class PathConfig(BaseModel):
    """Path configuration."""
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def docs_root(self) -> Path:
        """Path to documentation root directory."""
        return DOCS_PATH
    
    @property
    def indices_path(self) -> Path:
        """Path to vector indices directory."""
        return self.base_dir / "embedding_indices"


class AppConfig(BaseModel):
    """Main application configuration."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


def load_config() -> AppConfig:
    """Load and return the application configuration."""
    return AppConfig()
