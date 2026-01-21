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
DOCS_PATH = Path("/models/RAGModelService/TensorRT-LLM/docs/source")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    temperature: float = Field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.2")))


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
