# Codebase Collection

This document contains merged code files for LLM context.

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/ai_guides/backendai_commands.sh

```bash
source bai_manager/backendai/env-dogbowl.sh

List sessions:
bai_manager/backendai/backendai-client ps


List vFolders;
bai_manager/backendai/backendai-client vfolder list

bai_manager/backendai/backendai-client/vfolder list-hosts
Default vfolder host: seoul-h100:flash01
Usable hosts: seoul-h100:flash01, seoul-h100:flash02, seoul-h100:flash03


Create new vFolder:
bai_manager/backendai/backendai-client/backendai-client vfolder create reflexdev_rag_service seoul-h100:flash03 --usage-mode "model"


bai_manager/backendai/backendai-client create --owner-access-key AKIATRQHTZ2Z4IJ7ICAB \
  --startup-command "echo /home/work/my-vol/script.py" \
  --bootstrap-script ./setup_test.sh \
  --tag "rag_deploy_session" \
  --architecture x86_64 \
  --type interactive \
  --name reflexdev_rag_service_session \
  --env VAR1=value1 \
  --env VAR2=value2 \
  --volume reflexdev_rag_service=my-vol \
  --resources cpu=4 --resources mem=8g --resources cuda.shares=1 \
  --group default \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2

Session ID 35203741-6a7f-4ccb-b2ea-47e6d9c65422 is created and ready.
∙ This session provides the following app services: sshd, ttyd, jupyter, jupyterlab, vscode, tensorboard, mlflow-ui, nniboard


SSH into B.AI Remote Session, interactive:
bai_manager/backendai/backendai-client session ssh 35203741-6a7f-4ccb-b2ea-47e6d9c65422

Copy Current project into remote directory. vFolder:
backend.ai session scp YOUR_SESSION_NAME -p 9922 -r ./ work@localhost:/home/work/my-vol/

backend.ai service create cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 08fc5b55-370a-4793-b582-f167309a6f0 1 -r gpu=5 -r mem=32 -r cpu=4 --tag agentic --name agentic

# full_version
./backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  ai_apps \
  1 \
  --name readlm \
  --tag rag_model_service \
  --project default \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models \
  --model-definition-path model-definition-readlm.yaml \
  --mount reflexdev_rag_service \
  --public \
  -o AKIATRQHTZ2Z4IJ7ICAB \
  -r cuda.shares=4 \
  -r mem=32g \
  -r cpu=4
  
  # short version
  ./backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  ai_apps \
  1 \
  --name readlm2 \
  --tag rag_model_service \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models \
  --model-definition-path model-definition-readlm.yaml \
  --public \
  -r cuda.shares=4 \
  -r mem=32g \
  -r cpu=4
  
  
  
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/config/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/config/config.py

```python
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
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/core/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/core/document_processor.py

```python
#!/usr/bin/env python3
"""
Document Processor for RAG Service

This module provides functionality for:
1. Reading files from various sources
2. Creating langchain Document objects
3. Chunking documents for efficient processing
4. Processing files for RAG applications
5. Handling GitHub repository cloning and processing
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import aiofiles
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from pydantic import BaseModel, Field

# Import GitHub utilities
from utils.github_utils import GitHubInfo, parse_github_url, clone_github_repo

class DocumentMetadata(BaseModel):
    """Metadata for processed documents"""

    source_path: str
    relative_path: Optional[str] = None
    filename: str
    last_updated: datetime
    file_size: int
    file_type: str
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentProcessor:
    """
    Handles document processing operations for RAG applications.
    
    This class provides functionality for:
    1. Reading files from disk
    2. Creating langchain Document objects
    3. Chunking documents for efficient processing
    4. Processing files for RAG applications
    5. Cloning and processing GitHub repositories
    """

    def __init__(
        self,
        docs_root: Optional[Union[str, Path]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        supported_extensions: List[str] = None,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        """
        Initialize the DocumentProcessor.
        
        Args:
            docs_root: Root directory for documents
            chunk_size: Size of chunks for document splitting
            chunk_overlap: Overlap between chunks
            supported_extensions: List of supported file extensions (without dots)
            logger: Logger instance
        """
        self.docs_root = Path(docs_root) if docs_root else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions or ["md", "txt", "rst", "html"]
        self.logger = logger or structlog.get_logger().bind(component="DocumentProcessor")
        
        # Initialize text splitters
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size // 4,  # Approximate token count
            chunk_overlap=self.chunk_overlap // 4
        )

    async def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Read file content asynchronously.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string or None if file cannot be read
        """
        file_path = Path(file_path)
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                self.logger.debug("File read successfully", path=str(file_path))
                return content
        except Exception as e:
            self.logger.error("File read error", path=str(file_path), error=str(e))
            return None

    def create_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Create metadata for a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentMetadata object
        """
        stats = file_path.stat()
        
        metadata = DocumentMetadata(
            source_path=str(file_path.absolute()),
            filename=file_path.name,
            last_updated=datetime.fromtimestamp(stats.st_mtime),
            file_size=stats.st_size,
            file_type=file_path.suffix.lstrip(".").lower(),
        )
        
        # Add relative path if docs_root is set
        if self.docs_root and file_path.is_relative_to(self.docs_root):
            metadata.relative_path = str(file_path.relative_to(self.docs_root))
            
        return metadata

    def get_text_splitter(self, file_type: str) -> Any:
        """
        Get appropriate text splitter based on file type.
        
        Args:
            file_type: Type of file (extension without dot)
            
        Returns:
            Text splitter instance
        """
        if file_type.lower() == "md":
            return self.markdown_splitter
        else:
            return self.text_splitter

    def chunk_document(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        file_type: str = "txt"
    ) -> List[Document]:
        """
        Split document into chunks.
        
        Args:
            content: Document content
            metadata: Document metadata
            file_type: Type of file
            
        Returns:
            List of Document objects
        """
        if not content:
            return []
            
        splitter = self.get_text_splitter(file_type)
        chunks = splitter.split_text(content)
        
        documents = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Create a copy of metadata to avoid modifying the original
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = total_chunks
            
            documents.append(
                Document(page_content=chunk, metadata=chunk_metadata)
            )
            
        self.logger.debug(
            "Document chunked", 
            chunk_count=len(documents), 
            filename=metadata.get("filename", "unknown")
        )
        
        return documents

    async def process_file(
        self, 
        file_path: Union[str, Path],
        chunk: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a single file into Document objects.
        
        Args:
            file_path: Path to the file
            chunk: Whether to chunk the document
            custom_metadata: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error("File not found", path=str(file_path))
            return []
            
        content = await self.read_file(file_path)
        if content is None:
            return []
            
        metadata = self.create_metadata(file_path)
        
        # Add custom metadata if provided
        if custom_metadata:
            metadata.custom_metadata.update(custom_metadata)
            
        # Convert metadata to dict for Document creation
        metadata_dict = metadata.model_dump()
        
        if chunk:
            return self.chunk_document(
                content, 
                metadata_dict, 
                file_type=metadata.file_type
            )
        else:
            return [Document(page_content=content, metadata=metadata_dict)]

    async def collect_documents(
        self,
        directory: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        chunk: bool = True,
        file_filter: Optional[Callable[[Path], bool]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Collect all documents from a directory.
        
        Args:
            directory: Directory to search for documents (defaults to docs_root)
            recursive: Whether to search recursively
            chunk: Whether to chunk the documents
            file_filter: Optional function to filter files
            custom_metadata: Additional metadata to include for all documents
            
        Returns:
            List of Document objects
        """
        search_dir = Path(directory) if directory else self.docs_root
        
        if not search_dir:
            self.logger.error("No directory specified and no docs_root set")
            return []
            
        if not search_dir.exists():
            self.logger.warning("Directory not found", path=str(search_dir))
            return []
            
        # Default file filter based on supported extensions
        if file_filter is None:
            file_filter = lambda p: p.is_file() and p.suffix.lstrip(".").lower() in self.supported_extensions
            
        # Find all matching files
        if recursive:
            files = [p for p in search_dir.rglob("*") if file_filter(p)]
        else:
            files = [p for p in search_dir.iterdir() if file_filter(p)]
            
        self.logger.info("Found files", count=len(files), directory=str(search_dir))
        print(f"Found {len(files)} files in {search_dir}")
        
        # Process all files
        tasks = [
            self.process_file(file_path, chunk=chunk, custom_metadata=custom_metadata)
            for file_path in files
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten the list of lists
        documents = [doc for sublist in results for doc in sublist]
        
        self.logger.info(
            "Collected documents", 
            file_count=len(files), 
            document_count=len(documents)
        )
        print(f"Collected {len(documents)} documents from {len(files)} files")
        
        return documents

    async def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
        file_type: str = "txt"
    ) -> List[Document]:
        """
        Process text content into Document objects.
        
        Args:
            text: Text content
            metadata: Metadata for the document
            chunk: Whether to chunk the document
            file_type: Type of file for choosing appropriate splitter
            
        Returns:
            List of Document objects
        """
        if not text:
            return []
            
        # Create default metadata if not provided
        if metadata is None:
            metadata = {
                "source_path": "text_input",
                "filename": "text_input.txt",
                "last_updated": datetime.now(),
                "file_size": len(text),
                "file_type": file_type,
            }
            
        if chunk:
            return self.chunk_document(text, metadata, file_type)
        else:
            return [Document(page_content=text, metadata=metadata)]

    def filter_documents(
        self,
        documents: List[Document],
        filter_func: Callable[[Document], bool]
    ) -> List[Document]:
        """
        Filter documents based on a filter function.
        
        Args:
            documents: List of documents to filter
            filter_func: Function that returns True for documents to keep
            
        Returns:
            Filtered list of documents
        """
        filtered = [doc for doc in documents if filter_func(doc)]
        
        self.logger.debug(
            "Filtered documents", 
            original_count=len(documents), 
            filtered_count=len(filtered)
        )
        
        return filtered

    def merge_documents(
        self,
        documents: List[Document],
        separator: str = "\n\n",
        max_tokens: Optional[int] = None
    ) -> Document:
        """
        Merge multiple documents into a single document.
        
        Args:
            documents: List of documents to merge
            separator: Separator to use between documents
            max_tokens: Maximum number of tokens in the merged document
            
        Returns:
            Merged document
        """
        if not documents:
            return Document(page_content="", metadata={})
            
        # Start with the first document's metadata
        merged_metadata = documents[0].metadata.copy()
        merged_metadata["merged_count"] = len(documents)
        merged_metadata["sources"] = [
            doc.metadata.get("source_path", "unknown") 
            for doc in documents
        ]
        
        # Merge content
        content = separator.join(doc.page_content for doc in documents)
        
        # Truncate if max_tokens is specified
        if max_tokens:
            # Approximate token count using token splitter
            if len(content) > max_tokens * 4:  # Rough estimate of 4 chars per token
                chunks = self.token_splitter.split_text(content)
                content = separator.join(chunks[:max_tokens // self.token_splitter.chunk_size])
                merged_metadata["truncated"] = True
                
        return Document(page_content=content, metadata=merged_metadata)

            
    async def process_documentation(self, docs_path: Path, vector_store_manager: Any) -> List[Document]:
        """
        Process documentation to create vector indices.
        
        Args:
            docs_path: Path to documentation directory
            vector_store_manager: Vector store manager instance
            
        Returns:
            List of processed documents
        """
        try:
            print(f"Processing documentation in {docs_path}...")
            
            # Set docs_root to the provided path
            self.docs_root = docs_path
            
            # Collect documents
            print("Collecting documents...")
            documents = await self.collect_documents()
            
            if not documents:
                self.logger.warning("No documents found", docs_path=str(docs_path))
                print(f"Warning: No documents found in {docs_path}")
                return []
            
            print(f"Found {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            self.logger.error("Error processing documentation", error=str(e))
            print(f"Error processing documentation: {str(e)}")
            raise

    async def clone_github_repository(
        self, 
        github_url: str, 
        target_dir: Union[str, Path]
    ) -> Tuple[Path, Optional[Exception]]:
        """
        Clone a GitHub repository and return the documentation path.
        
        Args:
            github_url: GitHub URL of the repository
            target_dir: Directory to clone the repository to
            
        Returns:
            Tuple of (documentation path, exception if any)
        """
        try:
            # Parse GitHub URL using github_utils
            github_info = parse_github_url(github_url)
            
            print(f"Cloning {github_info.owner}/{github_info.repo} ({github_info.branch})...")
            
            # Clone the repository using github_utils
            docs_path, error = clone_github_repo(github_info, target_dir)
            
            if error:
                self.logger.error(
                    "Error cloning repository", 
                    repo=f"{github_info.owner}/{github_info.repo}",
                    error=str(error)
                )
                return target_dir, error
            
            print(f"Documentation path: {docs_path}")
            
            # Update the docs_root for future operations
            self.docs_root = docs_path
            
            return docs_path, None
            
        except Exception as e:
            self.logger.error("Error cloning GitHub repository", error=str(e))
            print(f"Error cloning GitHub repository: {str(e)}")
            return Path(target_dir), e

    async def process_github_repository(
        self,
        github_url: str,
        target_dir: Union[str, Path],
        vector_store_manager: Any
    ) -> Tuple[List[Document], Path, Optional[Exception]]:
        """
        Process a GitHub repository for RAG.
        
        This function:
        1. Clones the repository
        2. Processes its documents
        3. Creates vector indices
        
        Args:
            github_url: GitHub URL of the repository
            target_dir: Directory to clone the repository to
            vector_store_manager: Vector store manager instance
            
        Returns:
            Tuple of (processed documents, docs path, exception if any)
        """
        try:
            # Clone the repository
            docs_path, error = await self.clone_github_repository(github_url, target_dir)
            
            if error:
                return [], docs_path, error
            
            # Process documentation
            documents = await self.process_documentation(docs_path, vector_store_manager)
            
            if not documents:
                self.logger.warning(
                    "No documents found in repository",
                    repo=github_url,
                    docs_path=str(docs_path)
                )
                return [], docs_path, None
            
            return documents, docs_path, None
            
        except Exception as e:
            self.logger.error(
                "Error processing GitHub repository",
                repo=github_url,
                error=str(e)
            )
            return [], Path(target_dir), e
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/core/llm.py

```python
"""LLM interface for RAG responses."""
from typing import AsyncGenerator, List, Optional, Any
import structlog

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.config import LLMConfig

class LLMInterface:
    """Interface for language model interactions."""
    
    def __init__(self, settings: LLMConfig):
        """
        Initialize the LLM interface.
        
        Args:
            settings: Configuration settings for the language model
        """
        self.logger = structlog.get_logger().bind(component="LLMInterface")
        self.settings = settings
        
        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            temperature=settings.temperature,
            streaming=settings.streaming,
            max_tokens=settings.max_tokens,
            timeout=120,  # Increased timeout for reliability
            max_retries=3,
        )
        
        # Initialize memory for chat history
        self.messages = []
        self.memory_k = settings.memory_k if hasattr(settings, 'memory_k') else 25
        
        # Create system prompt
        system_prompt = """
        You are a helpful AI Assistant with document search and retrieval capabilities. Answer questions based on the provided context.
        Provide the detailed explanation.
        The provided context is a list of documents from a vector store knowledge base.
        The similarity score for each document is also provided as Euclidean distance where the lower the number the more similar.
        If the context doesn't contain relevant information, use your general knowledge but mention this fact. Keep answers focused and relevant to the query.
        If there is no context provided and you don't know, then answer "I don't know".
        """
        self.system_prompt = SystemMessage(content=system_prompt)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                self.system_prompt,
                ("placeholder", "{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Question: {input}"),
            ]
        )
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def generate_response(
        self, 
        user_input: str,
        context: str
    ) -> AsyncGenerator[str, None]:
        """
        Generate RAG-enhanced streaming response.
        
        Args:
            user_input: User's question
            context: Retrieved context from documents
            
        Yields:
            Response chunks
        """
        try:
            # Debug context information
            self.logger.debug(
                "Context information",
                context_length=len(context),
                context_snippet=context[:100] + "..." if len(context) > 100 else context,
                user_input=user_input
            )
            
            # Format context as a message
            context_msg = HumanMessage(content=f"<context>\n{context}\n</context>\n")
            
            # Get chat history
            history = self.get_chat_history()
            self.logger.debug(
                "Chat history information",
                history_length=len(history),
                messages_count=len(self.messages)
            )
            
            # Debug prompt information
            self.logger.debug(
                "Preparing to send to LLM",
                model=self.settings.model_name,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            
            # Stream response
            response_content = ""
            async for chunk in self.chain.astream(
                {
                    "input": user_input,
                    "context": [context_msg],
                    "chat_history": history,
                }
            ):
                yield chunk
                response_content += chunk
            
            # Update memory
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=response_content))
            
        except Exception as e:
            self.logger.error(
                "Error generating response",
                error=str(e),
                user_input=user_input,
            )
            yield f"Error: {str(e)}"
    
    def get_chat_history(self) -> List[Any]:
        """
        Get the chat history for the context window.
        
        Returns:
            List of message objects
        """
        # Limit to the last k exchanges
        if len(self.messages) > self.memory_k * 2:
            return self.messages[-self.memory_k * 2:]
        return self.messages
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/core/rag_engine.py

```python
"""RAG engine that coordinates retrieval and LLM components."""
from typing import AsyncGenerator
import structlog

from core.retrieval import RetrievalEngine
from core.llm import LLMInterface

class RAGEngine:
    """Coordinates retrieval and language model components for RAG functionality."""
    
    def __init__(self, retrieval_engine: RetrievalEngine, llm_interface: LLMInterface):
        """
        Initialize the RAG engine.
        
        Args:
            retrieval_engine: Engine for retrieving relevant context
            llm_interface: Interface for language model interactions
        """
        self.logger = structlog.get_logger().bind(component="RAGEngine")
        self.retrieval_engine = retrieval_engine
        self.llm_interface = llm_interface
    
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's question
            
        Yields:
            Response chunks
        """
        try:
            # Get relevant context
            context, _ = await self.retrieval_engine.get_relevant_context(query)
            
            # Generate response with context
            async for chunk in self.llm_interface.generate_response(query, context):
                yield chunk
                
        except Exception as e:
            self.logger.error("Error processing query", error=str(e), query=query)
            yield f"Error processing your query: {str(e)}"
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/core/retrieval.py

```python
"""Retrieval engine for RAG functionality."""
from typing import Dict, List, Tuple, Any, Optional
import structlog

from data.vector_store import VectorStore
from data.filtering import DocumentFilter
from config.config import RetrievalSettings

class RetrievalEngine:
    """Handles document retrieval and context preparation for RAG."""
    
    def __init__(
        self,
        settings: RetrievalSettings,
        vector_store: VectorStore,
        document_filter: Optional[DocumentFilter] = None
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            settings: Configuration settings for retrieval
            vector_store: Vector database for similarity search
            document_filter: Optional filter for search results
        """
        self.logger = structlog.get_logger().bind(component="RetrievalEngine")
        self.settings = settings
        self.vector_store = vector_store
        self.document_filter = document_filter or DocumentFilter()
        
    async def get_relevant_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve and format relevant context for a query.
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (formatted context string, filtered results)
        """
        try:
            self.logger.debug(
                "Starting context retrieval",
                query=query,
                max_results=self.settings.max_results,
                docs_path=self.settings.docs_path,
                indices_path=self.settings.indices_path
            )
            
            # Search for relevant documents
            results = await self.vector_store.search_documents(
                query=query, 
                k=self.settings.max_results
            )
            
            self.logger.debug(
                "Retrieved documents from vector store",
                result_count=len(results),
                first_doc_snippet=results[0].get("content", "")[:100] + "..." if results else "No results"
            )
            
            # Filter results if a document filter is provided
            if self.document_filter:
                filtered_results = await self.document_filter.filter_search_results(query, results)
                self.logger.debug(
                    "Filtered search results",
                    original_count=len(results),
                    filtered_count=len(filtered_results)
                )
            else:
                filtered_results = results
                
            # Format context from filtered results
            context_parts = []
            
            for i, doc in enumerate(filtered_results):
                # Get metadata fields safely with defaults
                metadata = doc.get("metadata", {})
                relative_path = metadata.get("relative_path", "unknown_path")
                similarity_score = doc.get("similarity_score", 0.0)
                content = doc.get("content", "")
                
                self.logger.debug(
                    f"Document {i+1} details",
                    path=relative_path,
                    score=similarity_score,
                    content_length=len(content)
                )
                
                # Truncate content to limit tokens
                if len(content) > self.settings.max_tokens_per_doc * 4:
                    content = content[:self.settings.max_tokens_per_doc * 4] + "..."
                
                # Format context entry
                context_parts.append(
                    f"[Source: {relative_path} (Similarity: {similarity_score:.2f})]\n{content}\n"
                )
            
            if not context_parts:
                self.logger.warning("No documents passed filtering", query=query)
                return "No relevant context found after filtering.", []
            
            final_context = "\n".join(context_parts)
            self.logger.debug(
                "Final context prepared",
                context_length=len(final_context),
                part_count=len(context_parts)
            )
            
            return final_context, filtered_results
            
        except Exception as e:
            self.logger.error("Error retrieving context", error=str(e), query=query)
            return f"Error retrieving context: {str(e)}", []
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/data/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/data/document_filter.py

```python
from typing import Dict, List

import structlog
from langchain_openai import ChatOpenAI

logger = structlog.get_logger()


class DocumentFilter:
    """Filter for relevance-based document filtering"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        """Initialize DocumentFilter"""
        self.logger = logger.bind(component="DocumentFilter")
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
        )

    async def filter_search_results(
        self, query: str, results: List[Dict]
    ) -> List[Dict]:
        """
        Filter search results based on relevance to query

        Args:
            query: The user query
            results: List of search results

        Returns:
            Filtered list of documents
        """
        if not results:
            return []

        # Currently just pass through results
        # In the future, implement actual filtering logic here
        self.logger.info("Filtered documents", count=len(results))
        return results
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/data/filtering.py

```python
"""Document filtering for RAG functionality."""
from typing import Dict, List, Any, Optional
import structlog

class DocumentFilter:
    """Filter for search results to improve relevance."""
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the document filter.
        
        Args:
            threshold: Similarity threshold for filtering (lower is more permissive)
        """
        self.logger = structlog.get_logger().bind(component="DocumentFilter")
        self.threshold = threshold
        
    async def filter_search_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter search results based on relevance.
        
        Args:
            query: User's question
            results: List of search results
            
        Returns:
            Filtered list of results
        """
        if not results:
            return []
            
        # Get the best score (lowest value is best for Euclidean distance)
        best_score = min(result.get("similarity_score", float("inf")) for result in results)
        
        # Set a dynamic threshold based on the best score
        dynamic_threshold = best_score * (1 + self.threshold)
        
        # Filter results
        filtered_results = [
            result for result in results 
            if result.get("similarity_score", float("inf")) <= dynamic_threshold
        ]
        
        # Log filtering results
        self.logger.debug(
            "Filtered search results",
            original_count=len(results),
            filtered_count=len(filtered_results),
            best_score=best_score,
            threshold=dynamic_threshold,
        )
        
        return filtered_results
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/data/vector_store.py

```python
#!/usr/bin/env python3
"""
Vector Store Operations for RAG Service

This module provides functionality for:
1. Creating vector indices from documents
2. Loading and searching vector indices
3. Managing vector store operations for RAG applications
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from core.document_processor import DocumentProcessor
from config.config import LLMConfig, RetrievalSettings, PathConfig

# Initialize logger
logger = structlog.get_logger()


class VectorStore:
    """Manager for vector store operations"""

    def __init__(
        self, 
        docs_root: Union[str, Path], 
        indices_path: Union[str, Path],
        llm_config: Optional[LLMConfig] = None,
        retrieval_settings: Optional[RetrievalSettings] = None,
        path_config: Optional[PathConfig] = None,
        service_id: Optional[str] = None
    ):
        """
        Initialize VectorStore
        
        Args:
            docs_root: Path to documentation directory
            indices_path: Path to store vector indices
            llm_config: Optional LLM configuration for embeddings
            retrieval_settings: Optional retrieval settings
            path_config: Optional PathConfig instance for path resolution
            service_id: Optional service ID for service-specific paths
        """
        self.docs_root = Path(docs_root)
        self.indices_path = Path(indices_path)
        self.llm_config = llm_config
        self.retrieval_settings = retrieval_settings
        self.path_config = path_config
        self.service_id = service_id
        
        # Initialize embeddings with configuration if provided
        embedding_model = "text-embedding-3-small"  # Default model
        embedding_kwargs = {}
        
        if llm_config:
            # Use API key from config if available
            if llm_config.openai_api_key:
                embedding_kwargs["openai_api_key"] = llm_config.openai_api_key
            
        self.embeddings = OpenAIEmbeddings(model=embedding_model, **embedding_kwargs)
        self.logger = logger.bind(component="VectorStore")
        self.index: Optional[FAISS] = None
        self.index_name = "vectorstore"
        self.document_processor = DocumentProcessor(docs_root=self.docs_root)

    async def collect_documents(self) -> List[Document]:
        """
        Collect all documents from the documentation directory
        
        Returns:
            List of Document objects
        """
        if not self.docs_root.exists():
            self.logger.warning("Documentation directory not found", path=str(self.docs_root))
            return []

        try:
            documents = await self.document_processor.process_directory(
                directory=self.docs_root,
                file_pattern="*.md"
            )
            self.logger.info("Collected documents", count=len(documents))
            return documents
        except Exception as e:
            self.logger.error("Error collecting documents", error=str(e))
            return []

    async def create_indices(self, documents: List[Document]) -> None:
        """
        Create a single FAISS index from collected documents
        
        Args:
            documents: List of Document objects
        """
        self.indices_path.mkdir(exist_ok=True, parents=True)

        if not documents:
            self.logger.warning("No documents to index")
            return

        try:
            # Create and save index
            index = FAISS.from_documents(documents, self.embeddings)
            index.save_local(str(self.indices_path / self.index_name))
            self.logger.info(
                "Created and saved index", doc_count=len(documents)
            )
        except Exception as e:
            self.logger.error(
                "Failed to create index", error=str(e)
            )

    async def load_index(self) -> None:
        """Load the FAISS index into memory"""
        # Primary index path
        index_path = self.indices_path / self.index_name
        self.logger.info(f"Attempting to load index from: {index_path}")
        
        # If the primary path doesn't exist, try to resolve using PathConfig
        if not index_path.exists() and self.path_config:
            self.logger.info("Primary index path not found, trying to resolve using PathConfig")
            
            try:
                # Try to get service-specific indices path
                if self.service_id:
                    resolved_path = self.path_config.get_service_indices_path(self.service_id) / self.index_name
                    self.logger.info(f"Trying service-specific path: {resolved_path}")
                    if resolved_path.exists():
                        index_path = resolved_path
                        self.logger.debug(f"Found index at service-specific path: {index_path}")
                
                # If still not found, try the default indices path
                if not index_path.exists():
                    resolved_path = self.path_config.indices_path / self.index_name
                    self.logger.info(f"Trying default indices path: {resolved_path}")
                    if resolved_path.exists():
                        index_path = resolved_path
                        self.logger.debug(f"Found index at default indices path: {index_path}")
                
                # If still not found, try to resolve the path
                if not index_path.exists():
                    resolved_path = self.path_config.resolve_path(self.indices_path / self.index_name)
                    self.logger.info(f"Trying resolved path: {resolved_path}")
                    if resolved_path.exists():
                        index_path = resolved_path
                        self.logger.debug(f"Found index at resolved path: {index_path}")
            
            except Exception as e:
                self.logger.error(f"Error resolving paths: {str(e)}", exc_info=True)
        
        # If index still not found, log warning and return
        if not index_path.exists():
            self.logger.warning(f"Index not found at any path: {index_path}")
            # List all files in the indices directory to help debugging
            try:
                self.logger.debug(f"Contents of indices directory {self.indices_path}:")
                if self.indices_path.exists():
                    for file in self.indices_path.iterdir():
                        self.logger.debug(f"  - {file.name} ({file.stat().st_size} bytes)")
                else:
                    self.logger.debug(f"  Directory does not exist")
            except Exception as e:
                self.logger.error(f"Error listing indices directory: {str(e)}")
            return
        
        try:
            self.logger.info(f"Loading index from: {index_path}")
            # Check if both required files exist
            faiss_file = index_path / "index.faiss"
            pkl_file = index_path / "index.pkl"
            
            if not faiss_file.exists() or not pkl_file.exists():
                self.logger.warning(
                    "Missing index files",
                    faiss_exists=faiss_file.exists(),
                    pkl_exists=pkl_file.exists(),
                    path=str(index_path)
                )
            
            self.index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info(f"Successfully loaded index from: {index_path}")
            
            # Add debug info about the loaded index
            if hasattr(self.index, 'docstore') and hasattr(self.index.docstore, '_dict'):
                doc_count = len(self.index.docstore._dict)
                self.logger.debug(f"Loaded index contains {doc_count} documents")
                
                # Sample a few document IDs for debugging
                if doc_count > 0:
                    sample_ids = list(self.index.docstore._dict.keys())[:3]
                    self.logger.debug(f"Sample document IDs: {sample_ids}")
            
        except Exception as e:
            self.logger.error(
                "Failed to load index", error=str(e), path=str(index_path), exc_info=True
            )

    async def search_documents(
        self, query: str, k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search documents in the index
        
        Args:
            query: Search query
            k: Number of results to return (uses retrieval_settings.max_results if not provided)
            
        Returns:
            List of document dictionaries with content, metadata, and similarity score
        """
        self.logger.info(f"Searching for query: '{query}'")
        
        # Use retrieval settings for max_results if available
        if k is None and self.retrieval_settings and self.retrieval_settings.max_results:
            k = self.retrieval_settings.max_results
        elif k is None:
            k = 5  # Default value
        
        # Safe access to embeddings model name
        embeddings_model = getattr(self.embeddings, 'model_name', 'unknown')
        self.logger.debug(f"Search parameters: k={k}, embeddings_model={embeddings_model}")
        
        if not self.index:
            self.logger.info("Index not loaded, attempting to load it now")
            await self.load_index()
            
        if not self.index:
            self.logger.error("Failed to load index for search", indices_path=str(self.indices_path))
            raise ValueError("No index loaded")

        try:
            self.logger.info(f"Performing similarity search with k={k}")
            docs_with_scores = self.index.similarity_search_with_score(query, k=k)
            self.logger.info(f"Search successful, found {len(docs_with_scores)} results")
            
            # Debug the first few results
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                self.logger.debug(
                    f"Search result {i+1}",
                    score=score,
                    content_preview=doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    metadata=doc.metadata
                )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                }
                for doc, score in docs_with_scores
            ]

        except Exception as e:
            self.logger.error("Search failed", error=str(e), exc_info=True)
            raise

    async def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Return documents for a given search query
        
        Args:
            query: Search query
            k: Number of results to return (uses retrieval_settings.max_results if not provided)
            
        Returns:
            List of Document objects
        """
        # Use retrieval settings for max_results if available
        if k is None and self.retrieval_settings and self.retrieval_settings.max_results:
            k = self.retrieval_settings.max_results
        elif k is None:
            k = 5  # Default value
            
        if self.index is None:
            await self.load_index()
            if self.index is None:
                raise ValueError("No loaded index available.")
        
        return self.index.similarity_search(query, k=k)

    async def process_documentation(self, docs_path: Optional[Path] = None, indices_path: Optional[Path] = None) -> List[Document]:
        """
        Process documentation to create vector indices.
        
        Args:
            docs_path: Optional override for documentation directory
            indices_path: Optional override for indices directory
            
        Returns:
            List of collected documents
        """
        try:
            # Use provided paths or instance paths
            docs_path = docs_path or self.docs_root
            indices_path = indices_path or self.indices_path
            
            print(f"Processing documentation in {docs_path}...")
            
            # Update document processor with new docs_path if provided
            if docs_path != self.docs_root:
                self.document_processor = DocumentProcessor(docs_root=docs_path)
            
            # Collect documents
            print("Collecting documents...")
            documents = await self.collect_documents()
            
            if not documents:
                logger.warning("No documents found", docs_path=str(docs_path))
                print(f"Warning: No documents found in {docs_path}")
                return []
            
            print(f"Found {len(documents)} documents")
            
            # Create vector indices
            print("Creating vector indices...")
            await self.create_indices(documents)
            
            print(f"Vector indices created successfully in {indices_path}")
            
            return documents
            
        except Exception as e:
            logger.error("Error processing documentation", error=str(e))
            print(f"Error processing documentation: {str(e)}")
            raise
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/scripts/setup_fastapi.sh

```bash
#!/bin/bash
# Setup script for the RAG Model Service FastAPI interface
# This script installs dependencies and sets up environment variables

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Install the package in development mode
pip install -e .

# Install FastAPI dependencies
pip install fastapi==0.109.2 uvicorn==0.27.1 typer==0.9.0

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Add environment variables to bashrc for persistence
echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
echo "export BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH}" >> ~/.bashrc
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    cat ${BACKEND_MODEL_PATH}/RAGModelService/.env >> ~/.bashrc
fi
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc

echo "RAG Model Service FastAPI setup completed successfully"
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/scripts/setup_gradio.sh

```bash
#!/bin/bash
# Setup script for the RAG Model Service Gradio interface
# This script installs dependencies and sets up environment variables

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Install the package in development mode
pip install -e .

# Install Gradio dependency
pip install gradio==5.23.1

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Add environment variables to bashrc for persistence
echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
echo "export BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH}" >> ~/.bashrc
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    cat ${BACKEND_MODEL_PATH}/RAGModelService/.env >> ~/.bashrc
fi
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc

echo "RAG Model Service setup completed successfully"
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/scripts/setup_portal.sh

```bash
#!/bin/bash
# Setup script for the RAG Model Service Portal interface
# This script installs dependencies and sets up environment variables

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Install the package in development mode
pip install -e .

# Install Gradio dependency
pip install gradio==5.23.1

# Install Backend.AI client in the background
nohup pip install backend.ai-client==25.4.0 > backend_install.log 2>&1 &
echo "Installing Backend.AI client in the background"

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Add environment variables to bashrc for persistence
echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
echo "export BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH}" >> ~/.bashrc
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    cat ${BACKEND_MODEL_PATH}/RAGModelService/.env >> ~/.bashrc
fi
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc

echo "RAG Model Service Portal setup completed successfully"
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/scripts/start_fastapi_server.sh

```bash
#!/bin/bash
# Start script for the RAG Model Service FastAPI server
# This script starts the FastAPI server with the specified configuration

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Set default values if not provided in environment
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
SERVICE_ID=${SERVICE_ID:-default}

# Construct paths for docs and indices
DOCS_PATH=${DOCS_PATH:-${BACKEND_MODEL_PATH}/rag_services/${SERVICE_ID}/docs}
INDICES_PATH=${INDICES_PATH:-${BACKEND_MODEL_PATH}/rag_services/${SERVICE_ID}/indices}

echo "Starting RAG Model Service FastAPI server with:"
echo "  - Host: ${HOST}"
echo "  - Port: ${PORT}"
echo "  - Service ID: ${SERVICE_ID}"
echo "  - Docs path: ${DOCS_PATH}"
echo "  - Indices path: ${INDICES_PATH}"

# Start the FastAPI server
python3 ${BACKEND_MODEL_PATH}/RAGModelService/interfaces/fastapi_app/fastapi_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --docs-path "${DOCS_PATH}" \
    --indices-path "${INDICES_PATH}" \
    --service-id "${SERVICE_ID}"

# Keep the container running
tail -f /dev/null
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/scripts/start_gradio.sh

```bash
#!/bin/bash
# Start script for the RAG Model Service Gradio interface
# This script starts the RAG service with the specified configuration

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}
RAG_SERVICE_PATH=${RAG_SERVICE_PATH:-rag_services}

# Source environment variables
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Set default values if not provided in environment
SERVICE_ID=${SERVICE_ID:-default}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Construct paths for docs and indices
DOCS_PATH=${DOCS_PATH:-${BACKEND_MODEL_PATH}/${RAG_SERVICE_PATH}/${SERVICE_ID}/docs}
INDICES_PATH=${INDICES_PATH:-${BACKEND_MODEL_PATH}/${RAG_SERVICE_PATH}/${SERVICE_ID}/indices}

echo "Starting RAG Model Service with:"
echo "  - Docs path: ${DOCS_PATH}"
echo "  - Indices path: ${INDICES_PATH}"
echo "  - Host: ${HOST}"
echo "  - Port: ${PORT}"
echo "  - Service ID: ${SERVICE_ID}"

# Launch the RAG service using launch_gradio.py
python ${BACKEND_MODEL_PATH}/RAGModelService/interfaces/cli_app/launch_gradio.py \
    --docs-path "${DOCS_PATH}" \
    --indices-path "${INDICES_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --service-id "${SERVICE_ID}"

# Keep the container running
tail -f /dev/null
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/scripts/start_portal.sh

```bash
#!/bin/bash
# Start script for the RAG Model Service Portal interface
# This script starts the Portal service with the specified configuration

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Set default values if not provided in environment
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7860}

echo "Starting RAG Model Service Portal with:"
echo "  - Host: ${HOST}"
echo "  - Port: ${PORT}"
echo "  - Base path: ${BACKEND_MODEL_PATH}"

# Start the Portal interface
python3 ${BACKEND_MODEL_PATH}/RAGModelService/interfaces/portal/app.py \
    --host "${HOST}" \
    --port "${PORT}"

# Keep the container running
tail -f /dev/null
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/setup/model-definition-fastapi.yaml

```yaml
models:
  - name: "TensorRT-LLM RAG Assistant PoC"
    model_path: "/models"
    service:
      pre_start_actions:
        - action: run_command
          args:
            command: ["/bin/bash", "/models/RAGModelService/setup.sh"]
      start_command:
        - /bin/bash
        - /models/RAGModelService/run_fastapi_server.sh
      port: 8000
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/setup/model-definition-gradio.yml

```yaml
models:
- model_path: /models
  name: RAG Service for repo
  service:
    port: 8000
    pre_start_actions:
    - action: run_command
      args:
        command:
        - /bin/bash
        - /models/RAGModelService/deployment/scripts/setup.sh
    start_command:
    - python3
    - /models/RAGModelService/interfaces/gradio_app/gradio_app.py
    - --indices-path
    - /models/RAGModelService/rag_services/owner/repo/indices
    - --docs-path
    - /models/RAGModelService/rag_services/owner/repo/docs
    - --host
    - 0.0.0.0
    - --port
    - '8000'
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/deployment/setup/model-definition-portal.yaml

```yaml
models:
  - name: "TensroRT-LLM RAG Portal"
    model_path: "/models"
    service:
      pre_start_actions:
        - action: run_command
          args:
            command: ["/bin/bash", "/models/RAGModelService/deployment/scripts/setup_portal.sh"]
      start_command:
        - /bin/bash
        - /models/RAGModelService/deployment/scripts/start_portal.sh
      port: 8000
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/create_rag_service.py

```python
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
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/generate_model_definition_cli.py

```python
#!/usr/bin/env python3
"""
Model Definition Generator CLI

This script generates a model definition YAML file for a RAG service based on a GitHub URL.
It provides a command-line interface to the model definition generation functionality.

Usage:
    python generate_model_definition_cli.py --github-url https://github.com/owner/repo
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import structlog
from dotenv import load_dotenv

# Import the model definition generator functionality
from interfaces.portal.generate_model_definition import (
    parse_github_url,
    generate_model_name,
    generate_docs_name,
    write_model_definition
)
from config.config import load_config, BACKEND_MODEL_PATH

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    # Load configuration to use as defaults
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Generate model definition YAML for RAG service"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the model definition file (if not provided, uses config default)",
        default=None,
    )
    
    # Service ID (optional, will be generated if not provided)
    parser.add_argument(
        "--service-id",
        type=str,
        help="Service ID for the RAG service (will be generated if not provided)",
        default=None,
    )
    
    # Model name prefix
    parser.add_argument(
        "--name-prefix",
        type=str,
        help="Prefix for the model name",
        default="RAG Service for",
    )
    
    # Port
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the service",
        default=8000,
    )
    
    # Service type
    parser.add_argument(
        "--service-type",
        type=str,
        help="Type of service (gradio or fastapi)",
        choices=["gradio", "fastapi"],
        default="gradio",
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
        return True
        
    except Exception as e:
        logger.error("Error setting up environment", error=str(e))
        print(f"Error setting up environment: {str(e)}")
        return False


def generate_model_definition_local(github_url: str, model_name: str, port: int, service_type: str, service_id: str) -> Dict:
    """
    Generate a model definition for the RAG service - local version that doesn't rely on PathConfig.backend_model_path.
    
    Args:
        github_url: GitHub URL
        model_name: Model name
        port: Port number
        service_type: Service type (gradio or fastapi)
        service_id: Service ID
        
    Returns:
        Model definition as a dictionary
    """
    # Parse the GitHub URL
    owner, repo, branch, path = parse_github_url(github_url)
    
    # Determine the docs path argument
    docs_path_arg = path if path else ""
    
    # Build the service-specific paths using the environment variable
    backend_model_path = str(BACKEND_MODEL_PATH)
    service_dir_path = f"{backend_model_path}/RAGModelService/rag_services/{service_id}"
    indices_path = f"{service_dir_path}/indices"
    docs_path = f"{service_dir_path}/docs"
    
    # Determine the start command based on service type
    if service_type == 'gradio':
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/cli_app/launch_gradio.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    else:  # fastapi
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/fastapi_app/fastapi_server.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    
    # Create the model definition
    model_definition = {
        'models': [
            {
                'name': model_name,
                'model_path': '/models',
                'service': {
                    'port': port,
                    'pre_start_actions': [
                        {
                            'action': 'run_command',
                            'args': {
                                'command': ['/bin/bash', f'{backend_model_path}/RAGModelService/deployment/scripts/setup_gradio.sh']
                            }
                        }
                    ],
                    'start_command': start_command
                }
            }
        ]
    }
    
    return model_definition


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    if not setup_environment():
        return 1
    
    try:
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        # Parse the GitHub URL
        github_url = args.github_url
        owner, repo, branch, path = parse_github_url(github_url)
        
        # Generate names
        model_name = generate_model_name(owner, repo, path, args.name_prefix)
        service_id = args.service_id if args.service_id else os.urandom(3).hex()
        yaml_name = f"model-definition-{service_id}.yml"
        
        # Generate the model definition using our local implementation
        logger.info(f"Generating model definition for {github_url}")
        model_def = generate_model_definition_local(github_url, model_name, args.port, args.service_type, service_id)
        
        # Resolve output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            print(f"Using provided output directory: {output_dir}")
        else:
            # Use the project path with deployment/setup subdirectory
            output_dir = path_config.project_path / "deployment" / "setup"
            print(f"Using default output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the model definition to YAML file
        output_path = output_dir / yaml_name
        write_model_definition(model_def, output_path)
        
        logger.info(f"Model definition written to {output_path}")
        print(f"✅ Model definition successfully generated: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error("Error generating model definition", error=str(e))
        print(f"❌ Error generating model definition: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/github_cli.py

```python
#!/usr/bin/env python3
"""
GitHub CLI Tool for RAG Service

This CLI tool provides a command-line interface for:
1. Cloning GitHub repositories
2. Preparing documentation for RAG processing

Usage:
    python github_cli.py clone https://github.com/owner/repo --output-dir ./output
    python github_cli.py prepare https://github.com/owner/repo --output-dir ./output
"""

import argparse
import sys
import asyncio
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel

# Import from portal.github for backward compatibility
from interfaces.portal.github import parse_github_url, prepare_for_rag, clone_github_repo

# Import centralized GitHub utilities for direct access when needed
from utils.github_utils import GitHubInfo
from config.config import load_config, PathConfig

# Initialize logger and console
logger = structlog.get_logger()
console = Console()


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser"""
    # Load configuration to use as defaults
    config = load_config()
    path_config = config.paths
    
    parser = argparse.ArgumentParser(
        description="GitHub CLI Tool for RAG Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone GitHub repository")
    clone_parser.add_argument(
        "github_url",
        help="GitHub URL of repository to clone",
    )
    clone_parser.add_argument(
        "--output-dir",
        "-o",
        help=f"Output directory for cloned repository (default: {path_config.get_service_docs_path()})",
        default=None,
    )
    clone_parser.add_argument(
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
        default=None,
    )
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare GitHub repository for RAG")
    prepare_parser.add_argument(
        "github_url",
        help="GitHub URL of repository to prepare",
    )
    prepare_parser.add_argument(
        "--output-dir",
        "-o",
        help=f"Output directory for prepared repository (default: {path_config.get_service_docs_path()})",
        default=None,
    )
    prepare_parser.add_argument(
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
        default=None,
    )
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse GitHub URL")
    parse_parser.add_argument(
        "github_url",
        help="GitHub URL to parse",
    )
    
    return parser


def handle_parse(args: argparse.Namespace) -> None:
    """Handle parse command"""
    try:
        owner, repo, branch, path = parse_github_url(args.github_url)
        
        console.print(Panel.fit(
            f"[bold]GitHub URL:[/bold] {args.github_url}\n\n"
            f"[bold]Owner:[/bold] {owner}\n"
            f"[bold]Repository:[/bold] {repo}\n"
            f"[bold]Branch:[/bold] {branch}\n"
            f"[bold]Path:[/bold] {path or '(root)'}"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


async def async_clone_repo(github_url: str, output_dir: Path) -> Path:
    """
    Asynchronously clone a repository.
    
    Args:
        github_url: GitHub URL
        output_dir: Output directory
        
    Returns:
        Path to documentation directory
    """
    try:
        return await clone_github_repo(github_url, output_dir)
    except Exception as e:
        console.print(f"[bold red]Error cloning repository:[/bold red] {str(e)}")
        raise


def handle_clone(args: argparse.Namespace) -> None:
    """Handle clone command"""
    try:
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        # Update service_id if provided
        if args.service_id:
            path_config.service_id = args.service_id
        
        # Resolve output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            console.print(f"Using provided output directory: [bold]{output_dir}[/bold]")
        else:
            output_dir = path_config.get_service_docs_path(path_config.service_id)
            console.print(f"Using docs path from config: [bold]{output_dir}[/bold]")
        
        # Parse GitHub URL to check if a specific path is provided
        owner, repo, branch, path = parse_github_url(args.github_url)
        if path:
            console.print(f"[bold]Sparse cloning directory:[/bold] {path} from {owner}/{repo} repository")
        else:
            console.print(f"[bold]Cloning repository:[/bold] {owner}/{repo}")
        
        # Use the async clone function directly for better control
        docs_path = asyncio.run(async_clone_repo(args.github_url, output_dir))
        
        console.print(f"[bold green]Success![/bold green] Repository cloned to {output_dir}")
        console.print(f"[bold]Documentation path:[/bold] {docs_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


def handle_prepare(args: argparse.Namespace) -> None:
    """Handle prepare command"""
    # This is essentially the same as clone for now
    handle_clone(args)


def main() -> None:
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands
    if args.command == "parse":
        handle_parse(args)
    elif args.command == "clone":
        handle_clone(args)
    elif args.command == "prepare":
        handle_prepare(args)


if __name__ == "__main__":
    main()
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/launch_gradio.py

```python
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
import sys
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
    
    # Base URL and model name
    parser.add_argument(
        '--base_url',
        type=str,
        default='http://localhost:8000',
        help='Base URL for the API endpoint'
    )
    parser.add_argument(
        '--base_model_name', 
        type=str,
        default='gpt-4o',
        help='Base model name to use for the LLM'
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
    
    # Update LLM configuration with base model name
    if args.base_model_name:
        config.llm.model_name = args.base_model_name
    
    if args.base_url:
        config.llm.base_url = args.base_url
    
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
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", config.llm.base_url),
        },
        "retrieval": {
            "max_results": args.max_results or config.rag.max_results,
            "max_tokens_per_doc": getattr(config.llm, "max_tokens_per_doc", 8000),  # Default to 8000 if not found
            "filter_threshold": float(os.environ.get("FILTER_THRESHOLD", "0.7")),  # Default to 0.7
        },
        "ui": {
            "title": args.title or "RAG Documentation Assistant",
            "description": args.description,
            "suggested_questions": args.suggested_questions or [
                "How do I install this project?",
                "What are the main features?",
                "How do I configure the system?"
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
        base_url=llm_config["openai_base_url"],
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
    # Check if interface is None
    if interface is None:
        logger.warning("Interface is None, cannot customize")
        return interface
        
    # Extract UI configuration
    ui_config = config.get("ui", {})
    if not ui_config:
        logger.warning("UI configuration is empty or missing")
        return interface
    
    try:
        # Set title if available
        if hasattr(interface, "title") and "title" in ui_config:
            interface.title = ui_config["title"]
            logger.debug(f"Set interface title to: {ui_config['title']}")
        
        # Instead of trying to modify the interface directly,
        # we'll store the configuration so it can be accessed later
        # during the interface rendering
        
        # Store the configuration in the interface for later use
        if not hasattr(interface, "_custom_config"):
            interface._custom_config = {}
        
        # Add UI configuration
        interface._custom_config["title"] = ui_config.get("title", "RAG Documentation Assistant")
        interface._custom_config["description"] = ui_config.get("description", "Documentation search with vector database")
        interface._custom_config["suggested_questions"] = ui_config.get("suggested_questions", [
            "How do I install this project?",
            "What are the main features?",
            "How do I configure the system?"
        ])
        
        logger.debug(f"Stored custom configuration in interface: {interface._custom_config}")
        
    except Exception as e:
        logger.error(f"Error customizing interface: {str(e)}")
    
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
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/rag_cli.py

```python
#!/usr/bin/env python3
"""
RAG Command-line Interface

This script provides an interactive command-line interface for the RAG system.
It allows users to query documentation and get AI-generated responses.

Features:
1. Interactive chat mode with the RAG system
2. Configurable LLM parameters (model, temperature)
3. Configurable retrieval parameters (max results)
4. Support for service-specific paths

Basic Usage:
    python rag_cli.py

With Custom Paths:
    python rag_cli.py --docs-path ./my_docs --indices-path ./my_indices

With Custom Model Parameters:
    python rag_cli.py --model gpt-4o-mini --temperature 0.3 --max-results 10

With Service ID:
    python rag_cli.py --service-id my_service_id

Environment Variables:
    OPENAI_API_KEY - OpenAI API key for LLM access
    OPENAI_BASE_URL - Optional custom OpenAI API endpoint
    OPENAI_MODEL - Default model to use
    TEMPERATURE - Default temperature for text generation
    MAX_RESULTS - Default number of search results to use
    RAG_SERVICE_PATH - Base path for RAG services
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

from core.retrieval import RetrievalEngine
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from config.config import load_config, LLMConfig, RetrievalSettings, PathConfig

logger = structlog.get_logger()

async def interactive_mode(rag_engine: RAGEngine, verbose: bool = False) -> None:
    """Run the interactive chatbot interface."""
    print("\n----- RAG Chatbot Test Interface -----")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Command line chat loop
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        # Process user query and get AI response
        print("\nThinking...")
        
        # Collect response chunks
        response_text = ""
        try:
            async for chunk in rag_engine.process_query(user_input):
                # Print chunk without newline to simulate streaming
                print(chunk, end="", flush=True)
                response_text += chunk
            print()  # Add a newline at the end
            
            if verbose:
                print("\n--- Debug Info ---")
                print(f"Model: {rag_engine.llm_interface.settings.model_name}")
                print(f"Temperature: {rag_engine.llm_interface.settings.temperature}")
                print("------------------")
        except Exception as e:
            print(f"\nError: {str(e)}")

async def main() -> int:
    """
    Main entry point for the RAG CLI application.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load configuration first
    config = load_config()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Command-line Interface")
    parser.add_argument("--docs-path", type=str, help="Path to documentation directory")
    parser.add_argument("--indices-path", type=str, help="Path to vector indices directory")
    parser.add_argument("--model", type=str, help=f"OpenAI model to use (default: {config.llm.model_name})")
    parser.add_argument("--temperature", type=float, help=f"Temperature for text generation (default: {config.llm.temperature})")
    parser.add_argument("--max-results", type=int, help=f"Maximum number of search results to use (default: {config.rag.max_results})")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--service-id", type=str, help="Service ID for service-specific paths")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Set up paths using PathConfig
        path_config = config.paths
        
        # Update service_id if provided
        if args.service_id:
            path_config.service_id = args.service_id
        
        # Resolve paths using config if not explicitly provided
        if args.docs_path:
            docs_path = Path(args.docs_path)
            print(f"Using provided docs path: {docs_path}")
        else:
            docs_path = path_config.get_service_docs_path(path_config.service_id)
            print(f"Using docs path from config: {docs_path}")
        
        if args.indices_path:
            indices_path = Path(args.indices_path)
            print(f"Using provided indices path: {indices_path}")
        else:
            indices_path = path_config.get_service_indices_path(path_config.service_id)
            print(f"Using indices path from config: {indices_path}")
        
        # Ensure paths exist
        if not docs_path.exists():
            print(f"Warning: Documentation directory {docs_path} does not exist.")
        
        # Create indices directory if it doesn't exist
        indices_path.mkdir(exist_ok=True, parents=True)
        
        # Set up LLM settings
        llm_settings = LLMConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY", config.llm.openai_api_key),
            base_url=os.environ.get("OPENAI_BASE_URL", config.llm.base_url),
            model_name=args.model or config.llm.model_name,
            temperature=args.temperature if args.temperature is not None else config.llm.temperature,
            streaming=True,
        )
        
        # Set up retrieval settings
        retrieval_settings = RetrievalSettings(
            max_results=args.max_results or config.rag.max_results,
            max_tokens_per_doc=config.llm.max_tokens_per_doc,
            filter_threshold=float(os.environ.get("FILTER_THRESHOLD", "0.7")),
            docs_path=str(docs_path),
            indices_path=str(indices_path),
        )
        
        # Initialize vector store with configuration
        vector_store = VectorStore(
            docs_root=docs_path,
            indices_path=indices_path,
            llm_config=llm_settings,
            retrieval_settings=retrieval_settings,
            path_config=path_config,
            service_id=args.service_id or path_config.service_id
        )
        
        # Load vector index
        print("Loading vector index...")
        await vector_store.load_index()
        
        # Initialize components
        llm_interface = LLMInterface(llm_settings)
        retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
        rag_engine = RAGEngine(retrieval_engine, llm_interface)
        
        # Run interactive mode
        await interactive_mode(rag_engine, args.verbose)
        
        return 0
        
    except Exception as e:
        logger.error("Error in RAG CLI", error=str(e))
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    import asyncio
    # Set logging level to DEBUG to see all debugging information
    import structlog
    import logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )
    exit_code = asyncio.run(main())
    exit(exit_code)
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/cli_app/vectorstore_cli.py

```python
#!/usr/bin/env python3
"""
Vector Store CLI Tool

This CLI tool provides a command-line interface for:
1. Processing documents and creating vector indices
2. Searching in vector indices
3. Testing and evaluating vector store functionality

Usage:
    # Process documents
    python vectorstore_cli.py process --docs-path ./docs --indices-path ./indices
    
    # Search in vector store
    python vectorstore_cli.py search "How to configure the model?" --indices-path ./indices
    
    # List available indices
    python vectorstore_cli.py list-indices --indices-path ./indices
    
    # Evaluate search performance
    python vectorstore_cli.py evaluate --indices-path ./indices --queries-file ./queries.txt
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from data.vector_store import VectorStore
from core.document_processor import DocumentProcessor
from config.config import load_config, LLMConfig, RetrievalSettings, PathConfig, ChunkingSettings

# Initialize logger and console
logger = structlog.get_logger()
console = Console()

# Create Typer app
app = typer.Typer(
    help="Vector Store CLI Tool for processing and searching documents",
    add_completion=False,
)


@app.command("process")
def process_documents(
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (if not provided, uses config default)",
    ),
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to store vector indices (if not provided, uses config default)",
    ),
    chunk: bool = typer.Option(
        True,
        "--chunk",
        "-c",
        help="Whether to chunk the documents",
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        "-c",
        help="Size of chunks for document splitting",
    ),
    chunk_overlap: int = typer.Option(
        200,
        "--chunk-overlap",
        "-o",
        help="Overlap between chunks",
    ),
    file_pattern: str = typer.Option(
        "*.md",
        "--file-pattern",
        "-p",
        help="File pattern to match (e.g., '*.md')",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """Process documents and create vector indices"""
    console.print(Panel.fit("Processing Documents", style="bold blue"))
    
    async def run():
        try:
            # Load configuration
            config = load_config()
            path_config = config.paths
            
            # Update path_config with service_id if provided
            if service_id:
                path_config.service_id = service_id
            
            # Resolve paths using config if not explicitly provided
            if docs_path is None:
                resolved_docs_path = path_config.get_service_docs_path(service_id)
                console.print(f"Using docs path from config: [bold]{resolved_docs_path}[/bold]")
            else:
                resolved_docs_path = docs_path
                console.print(f"Using provided docs path: [bold]{resolved_docs_path}[/bold]")
            
            if indices_path is None:
                resolved_indices_path = path_config.get_service_indices_path(service_id)
                console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
            else:
                resolved_indices_path = indices_path
                console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
            
            # Initialize vector store with configuration
            vector_store = VectorStore(
                docs_root=resolved_docs_path, 
                indices_path=resolved_indices_path,
                llm_config=config.llm,
                path_config=path_config,
                service_id=service_id
            )
            
            # Initialize document processor with custom chunk settings
            doc_processor = DocumentProcessor(
                docs_root=resolved_docs_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Process documentation
                task = progress.add_task(description="Processing documentation...", total=None)
                
                # Define a file filter based on the pattern
                extension = file_pattern.replace("*.", "")
                file_filter = lambda p: p.is_file() and p.suffix.lstrip(".").lower() == extension
                
                # Collect documents using DocumentProcessor
                documents = await doc_processor.collect_documents(
                    directory=resolved_docs_path,
                    recursive=True,
                    chunk=chunk,
                    file_filter=file_filter
                )
                
                if not documents:
                    console.print("[yellow]No documents found to process[/yellow]")
                    return
                
                progress.update(task, description=f"Creating vector indices for {len(documents)} documents...")
                
                # Create vector indices
                await vector_store.create_indices(documents)
                
                progress.update(task, description=f"Processed {len(documents)} documents")
            
            # Display summary
            console.print("\n[green]✓[/green] Document processing completed!")
            console.print(f"Documents processed: [bold]{len(documents)}[/bold]")
            console.print(f"Vector indices saved to: [bold]{resolved_indices_path}[/bold]")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


@app.command("search")
def search_documents(
    query: str = typer.Argument(..., help="Search query"),
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to vector indices (if not provided, uses config default)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of results to return (if not provided, uses config default)",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (if not provided, uses config default)",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """Search for documents in vector indices"""
    console.print(Panel.fit(f"Searching: {query}", style="bold blue"))
    
    async def run():
        try:
            # Load configuration
            config = load_config()
            path_config = config.paths
            retrieval_settings = config.rag
            
            # Update path_config with service_id if provided
            if service_id:
                path_config.service_id = service_id
            
            # Resolve paths using config if not explicitly provided
            if docs_path is None:
                resolved_docs_path = path_config.get_service_docs_path(service_id)
                console.print(f"Using docs path from config: [bold]{resolved_docs_path}[/bold]")
            else:
                resolved_docs_path = docs_path
                console.print(f"Using provided docs path: [bold]{resolved_docs_path}[/bold]")
            
            if indices_path is None:
                resolved_indices_path = path_config.get_service_indices_path(service_id)
                console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
            else:
                resolved_indices_path = indices_path
                console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
            
            # Initialize vector store with configuration
            vector_store = VectorStore(
                docs_root=resolved_docs_path, 
                indices_path=resolved_indices_path,
                llm_config=config.llm,
                retrieval_settings=retrieval_settings,
                path_config=path_config,
                service_id=service_id
            )
            
            # Load index
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(description="Loading vector index...", total=None)
                await vector_store.load_index()
                
                if not vector_store.index:
                    console.print("[red]Error:[/red] Failed to load vector index")
                    return
                
                progress.update(task, description="Searching documents...")
                results = await vector_store.search_documents(query, limit)
                
            # Display results
            if not results:
                console.print("[yellow]No matching documents found[/yellow]")
                return
                
            console.print(f"\nFound [bold]{len(results)}[/bold] matching documents:")
            
            # Create a table for results
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Score", justify="right", style="cyan", no_wrap=True)
            table.add_column("Source", style="green")
            table.add_column("Content", style="white")
            
            for i, result in enumerate(results):
                # Extract metadata
                metadata = result.get("metadata", {})
                source = metadata.get("relative_path", metadata.get("source_path", "Unknown"))
                
                # Format content (truncate if too long)
                content = result.get("content", "")
                if len(content) > 200:
                    content = content[:197] + "..."
                
                # Add row to table
                score = result.get("similarity_score", 0.0)
                table.add_row(f"{score:.4f}", source, content)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


@app.command("list-indices")
def list_indices(
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to vector indices (if not provided, uses config default)",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """List available vector indices"""
    console.print(Panel.fit("Available Vector Indices", style="bold blue"))
    
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Update path_config with service_id if provided
    if service_id:
        path_config.service_id = service_id
    
    # Resolve indices path using config if not explicitly provided
    if indices_path is None:
        resolved_indices_path = path_config.get_service_indices_path(service_id)
        console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
    else:
        resolved_indices_path = indices_path
        console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
    
    # Check if indices directory exists
    if not resolved_indices_path.exists():
        console.print("[yellow]Indices directory does not exist[/yellow]")
        return
    
    # List all subdirectories
    indices = [d for d in resolved_indices_path.iterdir() if d.is_dir()]
    
    if not indices:
        console.print("[yellow]No vector indices found[/yellow]")
        return
    
    # Create a table for indices
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Index", style="green")
    table.add_column("Size", justify="right", style="cyan")
    table.add_column("Last Modified", style="magenta")
    
    for index_dir in indices:
        # Get directory size
        size = sum(f.stat().st_size for f in index_dir.glob('**/*') if f.is_file())
        size_str = f"{size / 1024 / 1024:.2f} MB"
        
        # Get last modified time
        mtime = index_dir.stat().st_mtime
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        
        # Add row to table
        table.add_row(index_dir.name, size_str, mtime_str)
    
    console.print(table)


@app.command("evaluate")
def evaluate_search(
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to vector indices (if not provided, uses config default)",
    ),
    queries_file: Path = typer.Option(
        ...,
        "--queries-file",
        "-q",
        help="Path to file containing evaluation queries (one per line)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of results per query (if not provided, uses config default)",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (if not provided, uses config default)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path to save evaluation results (JSON format)",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """Evaluate search performance using a set of queries"""
    console.print(Panel.fit("Evaluating Search Performance", style="bold blue"))
    
    async def run():
        try:
            # Load configuration
            config = load_config()
            path_config = config.paths
            retrieval_settings = config.rag
            
            # Update path_config with service_id if provided
            if service_id:
                path_config.service_id = service_id
            
            # Resolve paths using config if not explicitly provided
            if docs_path is None:
                resolved_docs_path = path_config.get_service_docs_path(service_id)
                console.print(f"Using docs path from config: [bold]{resolved_docs_path}[/bold]")
            else:
                resolved_docs_path = docs_path
                console.print(f"Using provided docs path: [bold]{resolved_docs_path}[/bold]")
            
            if indices_path is None:
                resolved_indices_path = path_config.get_service_indices_path(service_id)
                console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
            else:
                resolved_indices_path = indices_path
                console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
            
            # Initialize vector store with configuration
            vector_store = VectorStore(
                docs_root=resolved_docs_path, 
                indices_path=resolved_indices_path,
                llm_config=config.llm,
                retrieval_settings=retrieval_settings,
                path_config=path_config,
                service_id=service_id
            )
            
            # Load queries from file
            with open(queries_file, "r") as f:
                queries = [line.strip() for line in f if line.strip()]
            
            console.print(f"Loaded [bold]{len(queries)}[/bold] queries from {queries_file}")
            
            # Load index
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(description="Loading vector index...", total=None)
                await vector_store.load_index()
                
                if not vector_store.index:
                    console.print("[red]Error:[/red] Failed to load vector index")
                    return
                
                # Process each query
                results = []
                task = progress.add_task(description="Evaluating queries...", total=len(queries))
                
                for query in queries:
                    start_time = time.time()
                    search_results = await vector_store.search_documents(query, limit)
                    end_time = time.time()
                    
                    query_result = {
                        "query": query,
                        "time_seconds": end_time - start_time,
                        "num_results": len(search_results),
                        "results": search_results,
                    }
                    
                    results.append(query_result)
                    progress.update(task, advance=1)
            
            # Calculate statistics
            times = [r["time_seconds"] for r in results]
            avg_time = sum(times) / len(times) if times else 0
            result_counts = [r["num_results"] for r in results]
            avg_results = sum(result_counts) / len(result_counts) if result_counts else 0
            
            # Display summary
            console.print("\n[bold]Evaluation Summary:[/bold]")
            console.print(f"Queries processed: [bold]{len(queries)}[/bold]")
            console.print(f"Average query time: [bold]{avg_time:.4f}[/bold] seconds")
            console.print(f"Average results per query: [bold]{avg_results:.2f}[/bold]")
            
            # Save results if output file is specified
            if output_file:
                output_data = {
                    "summary": {
                        "num_queries": len(queries),
                        "avg_time": avg_time,
                        "avg_results": avg_results,
                    },
                    "queries": results,
                }
                
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                
                console.print(f"\nResults saved to: [bold]{output_file}[/bold]")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


if __name__ == "__main__":
    app()
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/fastapi_app/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/fastapi_app/fastapi_server.py

```python
#!/usr/bin/env python3
"""
FastAPI Server for RAG Model Service
====================================

This module provides an OpenAI-compatible API for the RAG Model Service.
It implements a FastAPI server that exposes endpoints for chat completions
with retrieval-augmented generation capabilities.

Features:
- OpenAI-compatible API endpoints (/v1/chat/completions)
- Support for streaming responses
- Integration with vector store for document retrieval
- Authentication via API key (optional)
- Configurable service paths and model parameters

Usage Examples:
--------------

1. Start the server:
   ```bash
   # Basic usage
   python interfaces/fastapi_app/fastapi_server.py

   # With custom port and host
   python interfaces/fastapi_app/fastapi_server.py --port 8080 --host 127.0.0.1

   # With specific service ID (for path resolution)
   python interfaces/fastapi_app/fastapi_server.py --service-id my_service_id

   # With custom port, host, and service paths
   python interfaces/fastapi_app/fastapi_server.py --port 8080 --host 127.0.0.1 \
       --docs-path rag_services/ede16665/docs \
       --indices-path rag_services/ede16665/indices
   ```

2. Make API requests:
   ```python
   # Example Python client using requests
   import requests
   import json

   url = "http://localhost:8000/v1/chat/completions"
   headers = {
       "Content-Type": "application/json",
       "Authorization": "Bearer YOUR_API_KEY"  # Optional if VALIDATE_API_KEY=false
   }
   data = {
       "model": "gpt-4o",
       "messages": [
           {"role": "system", "content": "You are a helpful assistant with access to documentation."},
           {"role": "user", "content": "What is TensorRT-LLM used for?"}
       ],
       "temperature": 0.2,
       "stream": False
   }

   response = requests.post(url, headers=headers, json=data)
   print(json.dumps(response.json(), indent=2))
   ```

3. Streaming example:
   ```python
   # Example Python client for streaming responses
   import requests
   import json

   url = "http://localhost:8000/v1/chat/completions"
   headers = {
       "Content-Type": "application/json",
       "Authorization": "Bearer YOUR_API_KEY",  # Optional if VALIDATE_API_KEY=false
       "Accept": "text/event-stream"
   }
   data = {
       "model": "gpt-4o",
       "messages": [
           {"role": "system", "content": "You are a helpful assistant with access to documentation."},
           {"role": "user", "content": "Explain how to install TensorRT-LLM"}
       ],
       "temperature": 0.2,
       "stream": True
   }

   with requests.post(url, headers=headers, json=data, stream=True) as response:
       for line in response.iter_lines():
           if line:
               line = line.decode('utf-8')
               if line.startswith('data: '):
                   content = line[6:]
                   if content == '[DONE]':
                       break
                   try:
                       chunk = json.loads(content)
                       if chunk['choices'][0]['delta'].get('content'):
                           print(chunk['choices'][0]['delta']['content'], end='')
                   except json.JSONDecodeError:
                       pass
   ```

4. Using with curl:
   ```bash
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -d '{
       "model": "gpt-4o",
       "messages": [
         {"role": "system", "content": "You are a helpful assistant with access to documentation."},
         {"role": "user", "content": "How to install it?"}
       ],
       "temperature": 0.2
     }'
   ```

Environment Variables:
---------------------
- OPENAI_API_KEY: API key for OpenAI (used for embeddings and LLM)
- VALIDATE_API_KEY: Whether to validate API keys (default: "false")
- SERVICE_ID: Service ID for path resolution (default: "fastapi")
- OPENAI_MODEL: Model to use for chat completions (default from config)
- TEMPERATURE: Temperature for generation (default from config)
- MAX_RESULTS: Maximum number of results to return (default from config)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from core.llm import LLMInterface
from core.retrieval import RetrievalEngine
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from config.config import load_config, LLMConfig, PathConfig, RetrievalSettings

# Configuration
load_dotenv()
VALIDATE_KEY = os.getenv("VALIDATE_API_KEY", "false").lower() == "true"

# --- Pydantic Models ---


class Message(BaseModel):
    """OpenAI-compatible chat message"""

    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0
    top_p: Optional[float] = 1
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class Choice(BaseModel):
    """OpenAI-compatible chat completion choice"""

    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """OpenAI-compatible token usage info"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Usage


class ModelPermission(BaseModel):
    """OpenAI-compatible model permission"""

    id: str = Field(default_factory=lambda: f"modelperm-{uuid4()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class Model(BaseModel):
    """OpenAI-compatible model"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "organization-owner"
    permission: List[ModelPermission] = Field(default_factory=list)
    root: str
    parent: Optional[str] = None


class ModelList(BaseModel):
    """OpenAI-compatible model list"""

    object: str = "list"
    data: List[Model]


class APIKeyValidator:
    """Simple API key validator"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ):
        if not VALIDATE_KEY:
            return credentials.credentials
        if credentials.credentials != self.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return credentials.credentials


# --- FastAPI Application ---

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Initialize components on startup
    global vector_store, rag_engine

    # Get paths from configuration
    custom_docs_path = os.environ.get("CUSTOM_DOCS_PATH")
    if custom_docs_path:
        docs_root = Path(custom_docs_path)
    else:
        docs_root = path_config.get_service_docs_path(service_id)
    
    indices_root = path_config.get_service_indices_path(service_id)

    print(f"Starting up FastAPI server with configuration:")
    print(f"Service ID: {service_id}")
    print(f"Docs path: {docs_root}")
    print(f"Indices path: {indices_root}")
    print(f"LLM model: {llm_config.model_name}")
    print(f"Temperature: {llm_config.temperature}")
    print(f"Max results: {retrieval_settings.max_results}")

    # Create necessary directories if they don't exist
    try:
        docs_root.mkdir(exist_ok=True, parents=True)
        indices_root.mkdir(exist_ok=True, parents=True)
        print(f"Created directories successfully")
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create required directories: {str(e)}"
        )

    # Initialize vector store with configuration
    vector_store = VectorStore(
        docs_root=docs_root, 
        indices_path=indices_root,
        llm_config=llm_config,
        retrieval_settings=retrieval_settings,
        path_config=path_config,
        service_id=service_id
    )

    # Only load existing indices, don't recreate them
    await vector_store.load_index()
    
    # Initialize components
    llm_interface = LLMInterface(llm_config)
    retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
    rag_engine = RAGEngine(retrieval_engine, llm_interface)

    print("Startup complete - Ready to handle requests")
    
    # Yield control back to FastAPI
    yield
    
    # Cleanup on shutdown (if needed)
    print("Shutting down FastAPI server...")


# Create FastAPI app with lifespan
app = FastAPI(title="RAG OpenAI Compatible API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = load_config()
path_config = config.paths
llm_config = config.llm

# Get service ID from environment variable or use default
service_id = os.getenv("SERVICE_ID", "fastapi")

# Create retrieval settings from RAG config
retrieval_settings = RetrievalSettings(
    max_results=config.rag.max_results,
    max_tokens_per_doc=llm_config.max_tokens_per_doc,
    service_id=service_id
)

# Initialize with None, will be set during startup
vector_store = None
rag_engine = None

# API key validator (using environment variable or config)
api_key_validator = APIKeyValidator(os.getenv("OPENAI_API_KEY", llm_config.openai_api_key))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models"""
    default_model = Model(
        id="rag_service", root="rag_service", permission=[ModelPermission()]
    )
    return ModelList(data=[default_model])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
):
    """Create a chat completion using RAG-enhanced responses"""
    try:
        if request.stream:
            return await stream_chat_completion(request)

        # Get the last user message
        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(
                status_code=400, detail="Last message must be from user"
            )

        # Collect response chunks
        response_content = ""
        async for chunk in rag_engine.process_query(
            query=last_message.content
        ):
            response_content += chunk

        # Format response in OpenAI-compatible format
        choice = Choice(
            index=0,
            message=Message(role="assistant", content=response_content),
            finish_reason="stop",
        )

        # Estimate token usage (this is approximate)
        prompt_tokens = len(str(request.messages)) // 4
        completion_tokens = len(response_content) // 4
        return ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion responses"""
    try:
        # Get the last user message
        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(
                status_code=400, detail="Last message must be from user"
            )

        async def generate():
            # Send the first chunk with role
            first_chunk = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield json.dumps(first_chunk)

            # Stream the content chunks
            async for chunk in rag_engine.process_query(
                query=last_message.content
            ):
                response_chunk = {
                    "id": f"chatcmpl-{uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [
                        {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                    ],
                }
                yield json.dumps(response_chunk)

            # Send the final chunk
            final_chunk = {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield json.dumps(final_chunk)
            yield "[DONE]"

        return EventSourceResponse(generate())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import uvicorn
import typer
from typing import Optional

cli = typer.Typer()


@cli.command()
def main(
    service_id: Optional[str] = typer.Option(None, "--service-id", help="Service ID for path resolution"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to the config file"),
    env_file: Optional[str] = typer.Option(None, "--env-file", help="Path to the .env file"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host for the server"),
    port: int = typer.Option(8000, "--port", help="Port for the server"),
    indices_path: Optional[str] = typer.Option(None, "--indices-path", help="Path to the indices directory"),
    docs_path: Optional[str] = typer.Option(None, "--docs-path", help="Path to the documents directory"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum number of results to return"),
):
    """FastAPI server for RAG Model Service"""
    print(f"Starting server with service_id: {service_id}, config: {config}, env_file: {env_file}, indices_path: {indices_path}, docs_path: {docs_path}, max_results: {max_results}")

    # Load environment variables from .env file if provided
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()  # Load default .env file

    # Set service_id in environment if provided
    if service_id:
        os.environ["SERVICE_ID"] = service_id
        
    # Update path configuration with command-line arguments
    if docs_path or indices_path or service_id or max_results:
        # Create args dictionary for PathConfig.update_from_args
        args = {}
        if service_id:
            args["service_id"] = service_id
        if indices_path:
            args["indices_path"] = indices_path
        if docs_path:
            # PathConfig doesn't have docs_path field, but we can set it as a custom arg
            args["docs_path"] = docs_path
        
        # Update the path_config with the args
        path_config.update_from_args(args)
        
        # If docs_path is provided, we need to handle it specially
        if docs_path:
            # Store the docs_path for use in startup_event
            os.environ["CUSTOM_DOCS_PATH"] = docs_path
        
        # Update retrieval settings if max_results is provided
        if max_results:
            retrieval_settings.max_results = max_results

    # No need to call startup_event() directly anymore as it's handled by the lifespan context manager
    # startup_event()

    # Run the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/gradio_app/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/gradio_app/gradio_app.py

```python
import asyncio
import aiofiles
import base64
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import logging
from dotenv import load_dotenv

import gradio as gr
from pydantic import BaseModel, Field
from rich.console import Console

# Load environment variables
load_dotenv()

# Import components from the refactored structure
from config.config import load_config, LLMConfig, PathConfig, RetrievalSettings
from core.llm import LLMInterface
from core.retrieval import RetrievalEngine
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore

# Initialize logger and console
logger = logging.getLogger(__name__)
console = Console()

# Load configuration
config = load_config()
llm_config = config.llm
path_config = config.paths
# Create retrieval settings from RAG config
from config.config import RetrievalSettings
retrieval_settings = RetrievalSettings(
    max_results=config.rag.max_results,
    max_tokens_per_doc=getattr(config.llm, "max_tokens_per_doc", 8000),  # Default to 8000 if not found
    filter_threshold=float(os.environ.get("FILTER_THRESHOLD", "0.7"))
)

# Service name
DOC_NAME = os.getenv("RAG_SERVICE_NAME", llm_config.model_name)


class ChatMessage(BaseModel):
    """Single chat message model"""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatState(BaseModel):
    """State management for chat interface"""

    messages: List[ChatMessage] = Field(default_factory=list)
    current_docs: Dict[str, str] = Field(default_factory=dict)
    selected_doc: Optional[str] = Field(None)

    class Config:
        arbitrary_types_allowed = True


async def read_markdown_file(file_path: str | Path) -> str:
    """Read markdown file content asynchronously and fix image paths"""
    try:
        file_path = Path(file_path)
        logger.info(f"Attempting to read file: {file_path}")
        logger.info(f"File exists: {file_path.exists()}")
        logger.info(f"File is file: {file_path.is_file()}")
        logger.info(f"File absolute path: {file_path.absolute()}")

        # If file doesn't exist, try alternative paths
        if not file_path.exists():
            original_path = file_path
            
            # Try path with RAGModelService if it's not in the path
            if "RAGModelService" not in str(file_path):
                path_str = str(file_path)
                path_parts = path_str.split("/")
                for i in range(len(path_parts)):
                    if path_parts[i] == "rag_services" and i > 0:
                        rag_path_parts = path_parts.copy()
                        rag_path_parts.insert(i, "RAGModelService")
                        alt_path = Path("/".join(rag_path_parts))
                        logger.info(f"Trying alternative path with RAGModelService: {alt_path}")
                        if alt_path.exists() and alt_path.is_file():
                            logger.info(f"Found file at alternative path with RAGModelService: {alt_path}")
                            file_path = alt_path
                            break
            
            # Try path without RAGModelService if it's in the path
            if not file_path.exists() and "RAGModelService" in str(original_path):
                alt_path = Path(str(original_path).replace("/RAGModelService", ""))
                logger.info(f"Trying alternative path without RAGModelService: {alt_path}")
                if alt_path.exists() and alt_path.is_file():
                    logger.info(f"Found file at alternative path without RAGModelService: {alt_path}")
                    file_path = alt_path
            
            # Try replacing GitHub owner/repo with service ID format
            # This handles the case where paths are created with UUIDs instead of GitHub owner/repo
            if not file_path.exists():
                path_str = str(original_path)
                # Look for pattern like /models/RAGModelService/rag_services/NVIDIA/TensorRT-LLM/
                github_pattern = r'/models/RAGModelService/rag_services/([^/]+)/([^/]+)/'
                match = re.search(github_pattern, path_str)
                if match:
                    # Get the service ID from environment variable or try common UUIDs
                    service_id = os.environ.get('RAG_SERVICE_PATH', '')
                    
                    # If service ID is not set, try some common service ID formats
                    if not service_id:
                        # Try to find service ID in parent directories
                        parent_dir = original_path.parent
                        while parent_dir and str(parent_dir) != '/':
                            if parent_dir.name and len(parent_dir.name) == 8 and re.match(r'^[0-9a-f]+$', parent_dir.name):
                                service_id = parent_dir.name
                                break
                            parent_dir = parent_dir.parent
                    
                    if service_id:
                        # Replace GitHub owner/repo with service ID
                        owner_repo = f"{match.group(1)}/{match.group(2)}"
                        alt_path_str = path_str.replace(owner_repo, service_id)
                        alt_path = Path(alt_path_str)
                        logger.info(f"Trying path with service ID: {alt_path}")
                        if alt_path.exists() and alt_path.is_file():
                            logger.info(f"Found file at path with service ID: {alt_path}")
                            file_path = alt_path
            
            # If still not found, report error
            if not file_path.exists():
                error_msg = f"File not found in any location: {original_path}"
                logger.error(error_msg)
                logger.info(f"Parent directory exists: {original_path.parent.exists()} ")
                
                # List contents of parent directory if it exists
                if original_path.parent.exists():
                    logger.info(f"Contents of {original_path.parent}: {list(original_path.parent.iterdir())}")
                
                return f"Error: {error_msg}"

        logger.info(f"File exists, reading content...")
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Process Sphinx-style directives
        import re
        
        # Convert :::{image} directives to standard markdown
        image_directive_pattern = r":::\s*{image}\s*([^\s]+)(?:\s*:alt:\s*([^\n]+))?\s*:::"
        
        def replace_image_directive(match):
            img_path = match.group(1).strip()
            alt_text = match.group(2).strip() if match.group(2) else "Image"
            logger.info(f"Converting Sphinx image directive: {img_path} with alt text: {alt_text}")
            return f"![{alt_text}]({img_path})"
        
        content = re.sub(image_directive_pattern, replace_image_directive, content)
        
        # Convert :::{figure} directives to standard markdown
        figure_directive_pattern = r":::\s*{figure}\s*([^\s]+)(?:\s*:align:\s*[^\n]+)?(?:\s*:alt:\s*([^\n]+))?\s*:::"
        
        def replace_figure_directive(match):
            img_path = match.group(1).strip()
            alt_text = match.group(2).strip() if match.group(2) else "Figure"
            logger.info(f"Converting Sphinx figure directive: {img_path} with alt text: {alt_text}")
            return f"![{alt_text}]({img_path})"
        
        content = re.sub(figure_directive_pattern, replace_figure_directive, content)
        
        # Convert :::{contents} to a simple heading
        contents_directive_pattern = r":::\s*{contents}[^:]*:::"
        content = re.sub(contents_directive_pattern, "**Table of Contents**", content)

        # Fix image paths
        # Replace relative image paths with base64 encoded images
        file_dir = file_path.parent
        import base64
        import mimetypes
        import re

        # Define the base paths to search for assets
        base_paths = [
            Path(path_config.docs_path) / "docs" / "source",        # github_docs/docs/source
            Path(path_config.docs_path) / "source",                 # github_docs/source
            Path(path_config.docs_path),                            # github_docs
            Path(path_config.docs_path).parent / "docs" / "source", # parent/docs/source
            Path(path_config.docs_path) / "docs",                   # Added: github_docs/docs
            # The following paths handle the actual structure with "installation" subdirectory
            Path(path_config.docs_path) / "docs" / "source" / "installation",  # Added: docs/docs/source/installation
            Path(path_config.docs_path) / "source" / "installation",           # Added: docs/source/installation
            Path(path_config.docs_path) / "docs" / "source" / "reference",     # Added: docs/docs/source/reference 
        ]

        def find_asset_file(asset_path):
            """Find an asset file by trying different base paths"""
            # If it's an absolute path starting with /assets/
            if asset_path.startswith("/assets/"):
                asset_rel_path = asset_path.lstrip("/")
                
                # Try each base path
                for base_path in base_paths:
                    full_path = base_path / asset_rel_path
                    logger.info(f"Trying path: {full_path}")
                    if full_path.exists():
                        logger.info(f"Found asset at: {full_path}")
                        return full_path
                
                # If not found in standard locations, try a broader search
                asset_filename = Path(asset_path).name
                logger.info(f"Asset not found in standard locations. Searching for filename: {asset_filename}")
                
                # Search in the repository
                for root, dirs, files in os.walk(Path(path_config.docs_path).parent):
                    if asset_filename in files:
                        full_path = Path(root) / asset_filename
                        logger.info(f"Found asset by filename at: {full_path}")
                        return full_path
            
            # If it's a relative path
            else:
                # First try relative to the current file
                rel_path = file_dir / asset_path
                logger.info(f"Trying relative path: {rel_path}")
                if rel_path.exists():
                    logger.info(f"Found asset at: {rel_path}")
                    return rel_path
            
            # Asset not found
            logger.warning(f"Asset not found: {asset_path}")
            return None

        # Process HTML img tags
        def process_html_img(match):
            img_tag = match.group(0)
            src_pattern = r'src="([^"]+)"'
            src_match = re.search(src_pattern, img_tag)
            
            if src_match:
                img_path = src_match.group(1)
                
                # Skip external URLs and data URIs
                if img_path.startswith(("http://", "https://", "data:")):
                    return img_tag
                
                # Find the asset file
                img_file = find_asset_file(img_path)
                
                # If found, encode it
                if img_file:
                    mime_type = mimetypes.guess_type(img_file)[0]
                    with open(img_file, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    return img_tag.replace(
                        f'src="{img_path}"',
                        f'src="data:{mime_type};base64,{img_data}"',
                    )
            
            return img_tag

        # Process Markdown image syntax
        def process_markdown_img(match):
            img_tag = match.group(0)
            md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
            md_match = re.match(md_pattern, img_tag)
            
            if md_match:
                alt_text = md_match.group(1)
                img_path = md_match.group(2)
                
                # Skip external URLs and data URIs
                if img_path.startswith(("http://", "https://", "data:")):
                    return img_tag
                
                # Find the asset file
                img_file = find_asset_file(img_path)
                
                # If found, encode it
                if img_file:
                    mime_type = mimetypes.guess_type(img_file)[0]
                    with open(img_file, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    return f"![{alt_text}](data:{mime_type};base64,{img_data})"
            
            return img_tag

        # Process HTML img tags
        html_img_pattern = r'<img[^>]+>'
        content = re.sub(html_img_pattern, process_html_img, content)
        
        # Process markdown image syntax
        md_img_pattern = r'!\[[^\]]*\]\([^)]+\)'
        content = re.sub(md_img_pattern, process_markdown_img, content)

        logger.info(f"Successfully read {len(content)} characters")
        logger.info(f"Content preview: {content[:200]}")
        return content
    except Exception as e:
        error_msg = f"Error reading markdown file: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


async def process_query_with_rag(query: str, rag_engine: RAGEngine) -> str:
    """Process a query using the RAG engine and return the full response"""
    full_response = ""
    async for chunk in rag_engine.process_query(query):
        full_response += chunk
    return full_response


def create_gradio_interface(
    rag_engine: RAGEngine,
    docs_path: Optional[Path] = None,
    indices_path: Optional[Path] = None,
    service_id: Optional[str] = None
) -> gr.Blocks:
    """Create Gradio interface for the RAG chat application
    
    Args:
        rag_engine: RAG engine instance
        docs_path: Optional custom path to documentation files. If None, uses default from config.
        indices_path: Optional custom path to vector indices. If None, uses default from config.
        service_id: Optional service ID for service-specific paths. If None, uses default from config.
    """
    
    # IMMEDIATELY TEST read_markdown_file with a known path to verify our logs are working
    logger.error("=================== BEGINNING MANUAL TEST OF MARKDOWN READER ===================")
    try:
        actual_docs_path = Path(docs_path or path_config.docs_path)
        logger.error(f"Manual test - Docs path: {actual_docs_path}")
        # Test with a specific file we know is causing problems
        test_path = "source/installation/linux.md"
        logger.error(f"Manual test - Testing with path: {test_path}")
        
        # Try to find this file directly to ensure our lookup is working
        test_direct_path = actual_docs_path / "docs" / test_path
        logger.error(f"Manual direct path: {test_direct_path}")
        logger.error(f"Manual direct path exists: {test_direct_path.exists()}")
        
        # Try alternative path with RAGModelService if it's not in the path
        if "RAGModelService" not in str(actual_docs_path):
            path_str = str(actual_docs_path)
            path_parts = path_str.split("/")
            for i in range(len(path_parts)):
                if path_parts[i] == "rag_services" and i > 0:
                    rag_path_parts = path_parts.copy()
                    rag_path_parts.insert(i, "RAGModelService")
                    alt_path = Path("/".join(rag_path_parts))
                    logger.error(f"Trying alternative path with RAGModelService: {alt_path}")
                    if alt_path.exists() and alt_path.is_file():
                        logger.error(f"Found file at alternative path with RAGModelService: {alt_path}")
                        break
        
        # Try alternative path without RAGModelService if it's in the path
        if "RAGModelService" in str(actual_docs_path):
            alt_path = Path(str(actual_docs_path).replace("/RAGModelService", ""))
            logger.error(f"Trying alternative path without RAGModelService: {alt_path}")
            if alt_path.exists() and alt_path.is_file():
                logger.error(f"Found file at alternative path without RAGModelService: {alt_path}")
        
        # Recursively look for linux.md to confirm if it exists anywhere
        logger.error("Looking for linux.md anywhere in the docs path...")
        search_paths = [actual_docs_path]
        
        # Add alternative search paths
        if "RAGModelService" not in str(actual_docs_path):
            path_str = str(actual_docs_path)
            path_parts = path_str.split("/")
            for i in range(len(path_parts)):
                if path_parts[i] == "rag_services" and i > 0:
                    # Create path with RAGModelService inserted
                    rag_path_parts = path_parts.copy()
                    rag_path_parts.insert(i, "RAGModelService")
                    search_paths.append(Path("/".join(rag_path_parts)))
                    break
        
        if "RAGModelService" in str(actual_docs_path):
            search_paths.append(Path(str(actual_docs_path).replace("/RAGModelService", "")))
        
        # Search in all paths
        for search_path in search_paths:
            logger.error(f"Searching in: {search_path}")
            if search_path.exists():
                for root, dirs, files in os.walk(search_path):
                    if "linux.md" in files:
                        logger.error(f"FOUND linux.md at: {Path(root) / 'linux.md'}")
    except Exception as e:
        logger.error(f"Manual test FAILED with error: {e}")
    logger.error("=================== END MANUAL TEST ===================")
    
    # Use custom docs_path if provided, otherwise use default from config
    actual_docs_path = docs_path if docs_path is not None else Path(path_config.docs_path)
    logger.info(f"Using documentation path: {actual_docs_path}")
    
    # Use custom indices_path if provided, otherwise use default from config
    actual_indices_path = indices_path if indices_path is not None else Path(path_config.indices_path)
    logger.info(f"Using indices path: {actual_indices_path}")
    
    # Create a custom read_markdown_file function that uses the actual_docs_path
    async def custom_read_markdown_file(file_path: str | Path) -> str:
        """Read markdown file with custom docs path"""
        try:
            # Convert to Path if it's a string
            file_path = Path(file_path)
            logger.info(f"Attempting to read file: {file_path}")
            
            # First check if the file exists as provided
            if not file_path.exists():
                # Try to resolve the path using the actual_docs_path
                logger.info(f"File not found at direct path, trying to resolve with docs path")
                
                # Build a list of possible paths to try
                possible_paths = []
                
                # Try direct path combinations
                possible_paths.extend([
                    actual_docs_path / file_path,                  # direct path
                    actual_docs_path / "docs" / file_path,         # with one docs directory
                    actual_docs_path / "docs" / "docs" / file_path, # with double docs directory
                ])
                
                # Check for rag_services directory structure using the standard patterns
                rag_service_path = os.environ.get("RAG_SERVICE_PATH", str(Path(os.environ.get("PROJECT_PATH", "/Users/sergeyleksikov/Documents/GitHub/RAGModelService")) / "rag_services"))
                
                # Extract service ID from the docs path if possible
                service_id = None
                if "rag_services" in str(actual_docs_path):
                    path_parts = str(actual_docs_path).split("/")
                    for i, part in enumerate(path_parts):
                        if part == "rag_services" and i+1 < len(path_parts):
                            service_id = path_parts[i+1]
                            break
                
                # If we have a service ID, try the standard RAG service paths
                if service_id:
                    logger.info(f"Detected service ID: {service_id}, trying standard RAG service paths")
                    possible_paths.extend([
                        Path(f"{rag_service_path}/{service_id}/docs/{file_path}"),
                        Path(f"{rag_service_path}/{service_id}/docs/docs/{file_path}"),
                        Path(f"{rag_service_path}/{service_id}/docs/docs/source/{file_path}")
                    ])
                    
                    # Add direct path from metadata if it exists
                    if isinstance(file_path, str) and "metadata" in file_path:
                        try:
                            # This might be a JSON string from the metadata
                            import json
                            metadata = json.loads(file_path)
                            if "source_path" in metadata:
                                source_path = Path(metadata["source_path"])
                                logger.info(f"Found source_path in metadata: {source_path}")
                                possible_paths.append(source_path)
                        except:
                            pass
                
                # For paths like 'source/installation/linux.md', try both with and without prepending 'docs/'
                if str(file_path).startswith(("source/", "source\\")):
                    source_path = file_path
                    # Remove 'source/' prefix to try direct path to file
                    non_source_path = Path(str(file_path).replace("source/", "").replace("source\\", ""))
                    
                    # Add additional path combinations
                    possible_paths.extend([
                        actual_docs_path / "docs" / "source" / non_source_path,
                        actual_docs_path / "source" / non_source_path,
                        # Try the full path including 'source' but with different base directories
                        actual_docs_path / "docs" / source_path,
                    ])
                
                # Try each possible path
                resolved_path = None
                for possible_path in possible_paths:
                    logger.info(f"Trying path: {possible_path}")
                    if possible_path.exists():
                        logger.info(f"Found file at: {possible_path}")
                        resolved_path = possible_path
                        break
                
                # If still not found, try a last resort recursive search
                if not resolved_path:
                    # Convert string to Path object before accessing name attribute
                    file_name = Path(relative_path).name
                    logger.info(f"File not found in standard locations. Searching recursively for: {file_name}")
                    
                    # Recursively search for the file by name
                    for root, dirs, files in os.walk(actual_docs_path):
                        if file_name in files:
                            found_path = Path(root) / file_name
                            logger.info(f"Found file by name at: {found_path}")
                            resolved_path = found_path
                            break
            else:
                resolved_path = file_path
            
            # Final check - if still not found
            if not resolved_path or not resolved_path.exists():
                error_msg = f"File not found in any location: {file_path}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

            # Read the file
            logger.info(f"Reading file: {resolved_path}")
            async with aiofiles.open(resolved_path, "r", encoding="utf-8") as f:
                content = await f.read()

            # Process Sphinx-style directives
            import re
            
            # Convert :::{image} directives to standard markdown
            image_directive_pattern = r":::\s*{image}\s*([^\s]+)(?:\s*:alt:\s*([^\n]+))?\s*:::"
            
            def replace_image_directive(match):
                img_path = match.group(1).strip()
                alt_text = match.group(2).strip() if match.group(2) else "Image"
                logger.info(f"Converting Sphinx image directive: {img_path} with alt text: {alt_text}")
                return f"![{alt_text}]({img_path})"
            
            content = re.sub(image_directive_pattern, replace_image_directive, content)
            
            # Convert :::{figure} directives to standard markdown
            figure_directive_pattern = r":::\s*{figure}\s*([^\s]+)(?:\s*:align:\s*[^\n]+)?(?:\s*:alt:\s*([^\n]+))?\s*:::"
            
            def replace_figure_directive(match):
                img_path = match.group(1).strip()
                alt_text = match.group(2).strip() if match.group(2) else "Figure"
                logger.info(f"Converting Sphinx figure directive: {img_path} with alt text: {alt_text}")
                return f"![{alt_text}]({img_path})"
            
            content = re.sub(figure_directive_pattern, replace_figure_directive, content)
            
            # Convert :::{contents} to a simple heading
            contents_directive_pattern = r":::\s*{contents}[^:]*:::"
            content = re.sub(contents_directive_pattern, "**Table of Contents**", content)

            # Fix image paths
            # Replace relative image paths with base64 encoded images
            file_dir = resolved_path.parent
            import base64
            import mimetypes
            import re

            # Define the base paths to search for assets
            base_paths = [
                actual_docs_path / "docs" / "source",        # github_docs/docs/source
                actual_docs_path / "source",                 # github_docs/source
                actual_docs_path,                            # github_docs
                actual_docs_path.parent / "docs" / "source", # parent/docs/source
                actual_docs_path / "docs",                   # Added: github_docs/docs
                # The following paths handle the actual structure with "installation" subdirectory
                actual_docs_path / "docs" / "source" / "installation",  # Added: docs/docs/source/installation
                actual_docs_path / "source" / "installation",           # Added: docs/source/installation
                actual_docs_path / "docs" / "source" / "reference",     # Added: docs/docs/source/reference 
            ]

            def find_asset_file(asset_path):
                """Find an asset file by trying different base paths"""
                # If it's an absolute path starting with /assets/
                if asset_path.startswith("/assets/"):
                    asset_rel_path = asset_path.lstrip("/")
                    
                    # Try each base path
                    for base_path in base_paths:
                        full_path = base_path / asset_rel_path
                        logger.info(f"Trying path: {full_path}")
                        if full_path.exists():
                            logger.info(f"Found asset at: {full_path}")
                            return full_path
                    
                    # If not found in standard locations, try a broader search
                    asset_filename = Path(asset_path).name
                    logger.info(f"Asset not found in standard locations. Searching for filename: {asset_filename}")
                    
                    # Search in the repository
                    for root, dirs, files in os.walk(actual_docs_path.parent):
                        if asset_filename in files:
                            full_path = Path(root) / asset_filename
                            logger.info(f"Found asset by filename at: {full_path}")
                            return full_path
            
                # If it's a relative path
                else:
                    # First try relative to the current file
                    rel_path = file_dir / asset_path
                    logger.info(f"Trying relative path: {rel_path}")
                    if rel_path.exists():
                        logger.info(f"Found asset at: {rel_path}")
                        return rel_path
            
                # Asset not found
                logger.warning(f"Asset not found: {asset_path}")
                return None

            # Process HTML img tags
            def process_html_img(match):
                img_tag = match.group(0)
                src_pattern = r'src="([^"]+)"'
                src_match = re.search(src_pattern, img_tag)
                
                if src_match:
                    img_path = src_match.group(1)
                    
                    # Skip external URLs and data URIs
                    if img_path.startswith(("http://", "https://", "data:")):
                        return img_tag
                    
                    # Find the asset file
                    img_file = find_asset_file(img_path)
                    
                    # If found, encode it
                    if img_file:
                        mime_type = mimetypes.guess_type(img_file)[0]
                        with open(img_file, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()
                        return img_tag.replace(
                            f'src="{img_path}"',
                            f'src="data:{mime_type};base64,{img_data}"',
                        )
                
                return img_tag

            # Process Markdown image syntax
            def process_markdown_img(match):
                img_tag = match.group(0)
                md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
                md_match = re.match(md_pattern, img_tag)
                
                if md_match:
                    alt_text = md_match.group(1)
                    img_path = md_match.group(2)
                    
                    # Skip external URLs and data URIs
                    if img_path.startswith(("http://", "https://", "data:")):
                        return img_tag
                    
                    # Find the asset file
                    img_file = find_asset_file(img_path)
                    
                    # If found, encode it
                    if img_file:
                        mime_type = mimetypes.guess_type(img_file)[0]
                        with open(img_file, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()
                        return f"![{alt_text}](data:{mime_type};base64,{img_data})"
                
                return img_tag

            # Process HTML img tags
            html_img_pattern = r'<img[^>]+>'
            content = re.sub(html_img_pattern, process_html_img, content)
            
            # Process markdown image syntax
            md_img_pattern = r'!\[[^\]]*\]\([^)]+\)'
            content = re.sub(md_img_pattern, process_markdown_img, content)

            logger.info(f"Successfully read {len(content)} characters")
            logger.info(f"Content preview: {content[:200]}")
            return content
        except Exception as e:
            error_msg = f"Error reading markdown file: {e}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    async def process_message(
        message: str,
        state: Any,
    ) -> Tuple[List[Tuple[str, str]], str, str, List[List[str]], str, ChatState]:
        """Process user message and update interface components"""
        if not message.strip():
            if not isinstance(state, ChatState):
                state = ChatState()
            return (
                [],
                "",
                "",
                [],
                "",
                state,
            )

        try:
            # Initialize state if needed
            if not isinstance(state, ChatState):
                state = ChatState()

            # Add user message to state
            state.messages.append(ChatMessage(role="user", content=message))

            # Get context and retrieved results
            logger.info("Getting relevant context...")
            try:
                context, retrieved_results = await rag_engine.retrieval_engine.get_relevant_context(message)
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                return (
                    [],
                    message,
                    "",
                    [],
                    "",
                    state,
                )

            # Clear existing documents in state
            state.current_docs.clear()

            # Process retrieved results
            total_docs = 0
            for result in retrieved_results:
                # Get metadata fields safely with defaults
                metadata = result.get("metadata", {})
                score = result.get("similarity_score", 0.0)
                content = result.get("content", "")

                # Skip if no metadata or relative_path
                if not metadata or "relative_path" not in metadata:
                    logger.warning("Missing metadata for result")
                    continue
                
                relative_path = metadata["relative_path"]
                doc_name = f"{relative_path} (score: {score:.2f})"
                
                # Resolve the path using PathConfig
                resolved_path = None
                
                # First try the direct source_path from metadata if available
                if "source_path" in metadata:
                    source_path = Path(metadata["source_path"])
                    logger.info(f"Trying direct source_path from metadata: {source_path}")
                    if source_path.exists():
                        logger.info(f"Found file at direct source_path: {source_path}")
                        resolved_path = source_path
                
                # If not found via source_path, try standard paths
                if not resolved_path:
                    possible_paths = [
                        actual_docs_path / relative_path,
                        actual_docs_path / "docs" / relative_path,
                        actual_docs_path / "docs" / "docs" / relative_path,
                    ]
                    if str(relative_path).startswith(("source/", "source\\")):
                        source_path = relative_path
                        non_source_path = Path(str(relative_path).replace("source/", "").replace("source\\", ""))
                        possible_paths.extend([
                            actual_docs_path / "docs" / "source" / non_source_path,
                            actual_docs_path / "source" / non_source_path,
                            actual_docs_path / "docs" / source_path,
                        ])
                    for possible_path in possible_paths:
                        logger.info(f"Trying path: {possible_path}")
                        if possible_path.exists():
                            logger.info(f"Found file at: {possible_path}")
                            resolved_path = possible_path
                            break
                    if not resolved_path:
                        # Convert string to Path object before accessing name attribute
                        file_name = Path(relative_path).name
                        logger.info(f"File not found in standard locations. Searching recursively for: {file_name}")
                        for root, dirs, files in os.walk(actual_docs_path):
                            if file_name in files:
                                found_path = Path(root) / file_name
                                logger.info(f"Found file by name at: {found_path}")
                                resolved_path = found_path
                                break
                if not resolved_path or not resolved_path.exists():
                    logger.warning(f"File not found in any location: {relative_path}")
                    continue

                state.current_docs[doc_name] = str(resolved_path)
                total_docs += 1

            logger.info(f"Total documents in state after processing: {total_docs}")
            logger.info(f"Documents in state: {list(state.current_docs.keys())}")

            # Generate response
            full_response = await process_query_with_rag(message, rag_engine)

            # Add assistant response to state
            state.messages.append(ChatMessage(role="assistant", content=full_response))

            # Convert messages to the format expected by Gradio chatbot (list of tuples)
            chat_messages = [
                (msg.content if msg.role == "user" else None, 
                 msg.content if msg.role == "assistant" else None) 
                for msg in state.messages
            ]

            # Return updated UI components
            return (
                chat_messages,
                "",  # Clear input
                f"Total documents: {len(retrieved_results)}\nRetrieved documents: {len(state.current_docs)}",
                [
                    [name] for name in state.current_docs.keys()
                ],  # Correct format for gr.List
                (
                    ""
                    if not state.current_docs
                    else await custom_read_markdown_file(
                        state.current_docs[list(state.current_docs.keys())[0]]
                    )
                ),
                state,  # Return the state
            )
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            if not isinstance(state, ChatState):
                state = ChatState()
            # Add error message to chat
            error_message = f"Error: {str(e)}"
            state.messages.append(ChatMessage(role="assistant", content=error_message))
            chat_messages = [
                (msg.content if msg.role == "user" else None, 
                 msg.content if msg.role == "assistant" else None) 
                for msg in state.messages
            ]
            return chat_messages, "", error_message, [], "", state

    # Create Gradio interface
    with gr.Blocks(
        title=f"{DOC_NAME} Documentation Assistant",
        theme=gr.themes.Base(),
        css="""
        /* Hide New Row and New Column buttons in document viewer */
        .doc-column .controls-wrap {
            display: none !important;
        }
        
        /* Additional styling for better UI */
        .container {
            gap: 20px;
        }
        .chat-column {
            min-width: 400px;
        }
        .doc-column {
            min-width: 500px;
        }
        .wide-input {
            width: 100%;
        }
        .markdown-body pre {
            background-color: #2d2d2d !important;
            color: #f8f8f2 !important;
            padding: 16px !important;
            border-radius: 5px !important;
        }
        .markdown-body code {
            background-color: #2d2d2d !important;
            color: #f8f8f2 !important;
            padding: 0.2em 0.4em !important;
            border-radius: 3px !important;
        }
        """,
    ) as interface:
        # Initialize chat state with default ChatState instance
        initial_state = ChatState()
        state = gr.State(initial_state)

        gr.Markdown(f"# {DOC_NAME} Documentation Assistant")
        gr.Markdown(
            "Documentation search with vector database"
        )
        gr.Markdown(
            "Best usage: Ask specific questions about the documentation content"
        )  # Added hint

        with gr.Row(elem_classes="container"):
            # Left column - Chat and logs
            with gr.Column(elem_classes="chat-column"):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    container=True,
                    elem_classes="chat-history",
                )

                # Add suggestion questions (MATCHED WITH AGENT)
                with gr.Row():
                    gr.Markdown("### Suggested Questions")
                suggestion_questions = [
                    "How to install TensorRT-LLM on Ubuntu Linux?",
                    "What is AWQ?",
                    "What is prefix caching?",
                    "How to quantize Qwen-72b to 8-bit?",
                    "How to do Performance analysis of a model?",
                ]
                suggestion_buttons = []
                with gr.Row():
                    for question in suggestion_questions:
                        btn = gr.Button(question, variant="secondary", size="sm")
                        suggestion_buttons.append(btn)

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your question here...",
                        lines=2,
                        show_label=True,
                        scale=12,  # MATCHED SCALE
                        container=True,  # Add container for better layout
                        elem_classes="wide-input",  # Add custom class
                    )
                    submit_btn = gr.Button(
                        "Submit", variant="primary", scale=1
                    )  # Keep button narrow
                with gr.Accordion(
                    "Logs", open=False
                ):  # Renamed from Classification Results to Logs
                    logs_output = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        container=True,
                        lines=10,  # Increase height
                        scale=20,  # MATCHED SCALE
                        elem_classes=["full-width"],  # Custom class for full width
                    )
                with gr.Accordion("Retrieved Documents", open=True):
                    docs_list = gr.List(
                        label="Retrieved Documents",
                        show_label=True,
                        interactive=True,
                    )

            # Right column - Document content display
            with gr.Column(elem_classes="doc-column"):
                gr.Markdown("# Context document viewer")  # ADDED TITLE
                doc_content = gr.Markdown(
                    label="Document Content",
                    value="Select a document to view its contents",
                )

        # Event handlers
        # Submit via enter key
        msg_input.submit(
            fn=process_message,
            inputs=[msg_input, state],
            outputs=[
                chatbot,
                msg_input,
                logs_output,
                docs_list,
                doc_content,
                state,
            ],
        )
        # Submit via button click
        submit_btn.click(
            fn=process_message,
            inputs=[msg_input, state],
            outputs=[
                chatbot,
                msg_input,
                logs_output,
                docs_list,
                doc_content,
                state,
            ],
        )

        # Add click handlers for suggestion buttons
        def set_input_text(question):
            return question

        for btn, question in zip(suggestion_buttons, suggestion_questions):
            btn.click(
                fn=lambda x=question: set_input_text(x), inputs=[], outputs=[msg_input]
            )

        # Handle document selection with click event
        async def handle_doc_selection(
            evt: gr.SelectData, state: ChatState
        ) -> Tuple[str, ChatState]:
            """Handle document selection from the list"""
            try:
                logger.info(f"Selection event received: {evt}")
                logger.info(f"Selection event value: {evt.value}")
                logger.info(f"Selection event index: {evt.index}")
                logger.info(f"Current docs in state: {state.current_docs}")

                selected_doc_name = evt.value  # The value is already a string
                logger.info(f"Selected doc name: {selected_doc_name}")

                if selected_doc_name in state.current_docs:
                    file_path = state.current_docs[selected_doc_name]
                    logger.info(f"Found file path: {file_path}")
                    content = await custom_read_markdown_file(file_path)
                    state.selected_doc = selected_doc_name
                    return content, state

                logger.error(
                    f"Document not found. Available documents: {list(state.current_docs.keys())}"
                )
                return (
                    f"Error: Document {selected_doc_name} not found in current documents",
                    state,
                )
            except Exception as e:
                logger.error(f"Error handling document selection: {e}", exc_info=True)
                return f"Error: {str(e)}", state

        docs_list.select(
            fn=handle_doc_selection, inputs=[state], outputs=[doc_content, state]
        )

    return interface


async def main(
    docs_path: Optional[Path] = None, 
    indices_path: Optional[Path] = None,
    service_id: Optional[str] = None
):
    """Main entry point
    
    Args:
        docs_path: Optional custom path to documentation files. If None, uses default from config.
        indices_path: Optional custom path to vector indices. If None, uses default from config.
        service_id: Optional service ID for service-specific paths. If None, uses default from config.
    """
    try:
        # Update path_config with service_id if provided
        if service_id:
            path_config.service_id = service_id
            logger.info(f"Using service ID: {service_id}")
        
        # Use custom docs_path if provided, otherwise use service-specific path from config
        actual_docs_path = docs_path if docs_path is not None else path_config.get_service_docs_path()
        logger.info(f"Using documentation path: {actual_docs_path}")
        
        # Use custom indices_path if provided, otherwise use service-specific path from config
        actual_indices_path = indices_path if indices_path is not None else path_config.get_service_indices_path()
        logger.info(f"Using indices path: {actual_indices_path}")
        
        # Create necessary directories
        actual_docs_path.mkdir(exist_ok=True, parents=True)
        actual_indices_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize vector store with configuration
        vector_store = VectorStore(
            docs_root=actual_docs_path, 
            indices_path=actual_indices_path,
            llm_config=llm_config,
            retrieval_settings=retrieval_settings,
            path_config=path_config,
            service_id=service_id
        )
        
        # Log the absolute paths for debugging
        logger.info(f"Vector store initialized with:")
        logger.info(f"  - docs_root absolute path: {actual_docs_path.absolute()}")
        logger.info(f"  - indices_path absolute path: {actual_indices_path.absolute()}")

        # Load vector indices
        await vector_store.load_index()

        # Initialize components
        llm_interface = LLMInterface(llm_config)
        retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
        rag_engine = RAGEngine(retrieval_engine, llm_interface)

        # Create and launch Gradio interface
        interface = create_gradio_interface(
            rag_engine=rag_engine, 
            docs_path=actual_docs_path,
            indices_path=actual_indices_path,
            service_id=service_id
        )
        interface.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=True,
            debug=True,
        )
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        console.print(f"[red]Error initializing application: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/portal/app.py

```python
#!/usr/bin/env python3
"""
RAG Service Portal

A Gradio interface that allows users to:
1. Enter a GitHub URL with documentation
2. Process it in the background to create a RAG service
3. Provide a link to the resulting RAG Gradio service
4. Generate a model definition YAML file for Backend.AI model service creation

Usage:
    python app.py
"""

import os
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import gradio as gr
from dotenv import load_dotenv
import structlog
import asyncio
import traceback

# Filter specific Gradio warnings about documentation groups
warnings.filterwarnings("ignore", category=UserWarning, module="gradio_client.documentation")

# Import utility modules
from utils.github_utils import validate_github_url, parse_github_url, GitHubInfo
from utils.service_utils import (
    setup_environment, 
    get_unique_service_id,
    ServiceStatus,
    ServiceConfig,
    ServerConfig,
    save_service_info,
)
from core.document_processor import DocumentProcessor
from data.vector_store import VectorStore
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from config.config import load_config, LLMConfig, RetrievalSettings, PathConfig, ChunkingSettings
from interfaces.portal.generate_model_definition import generate_model_definition as gen_model_def
from interfaces.portal.generate_model_definition import write_model_definition

# Initialize logger
logger = structlog.get_logger()

# Global state to track running services
SERVICES = {}


def find_available_port(start_port: int = 8000, end_port: int = 9000) -> int:
    """Find an available port in the specified range"""
    import socket
    
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # If the connect fails, the port is available
            result = sock.connect_ex(('localhost', port))
            if result != 0:
                return port
                
    # If no port is available, return the start port and hope for the best
    return start_port


def create_service_directory(service_id: str) -> Path:
    """
    Create a directory for the service.
    
    Args:
        service_id: Service ID
        
    Returns:
        Path to the service directory
    """
    service_dir = Path("rag_services") / service_id
    service_dir.mkdir(parents=True, exist_ok=True)
    
    docs_dir = service_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    return service_dir


def generate_model_definition(github_url: str, service_dir: Path, service_type: str = "Gradio UI") -> Optional[Path]:
    """
    Generate a model definition YAML file for the RAG service.
    
    Args:
        github_url: GitHub URL for documentation
        service_dir: Path to the service directory
        service_type: Type of RAG service to create (Gradio UI or FastAPI Server)
        
    Returns:
        Path to the generated model definition file, or None if generation failed
    """
    try:
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        # Create a service configuration
        service_config = ServerConfig(
            host=config.server.host,
            port=config.server.port
        )
        
        # Define service type based on the radio button selection
        service_type_value = "gradio" if service_type == "Gradio UI" else "fastapi"
        
        github_info = parse_github_url(github_url)
        service_id = service_dir.name
        
        # Update path configuration with service ID
        path_config.service_id = service_id
        
        # Create model definition file path
        model_def_path = service_dir / f"model-definition-{service_id}.yml"
        
        # Get configuration values
        port = service_config.port
        
        # Get BACKEND_MODEL_PATH from environment variable or use a default
        backend_model_path = os.environ.get("BACKEND_MODEL_PATH", "/models")
        
        # Get RAG_SERVICE_PATH from environment variable or use a default
        rag_service_path = os.environ.get("RAG_SERVICE_PATH", f"{backend_model_path}/RAGModelService/rag_services/")
        
        logger.info("Using configuration for model definition", 
                   backend_model_path=backend_model_path,
                   rag_service_path=rag_service_path,
                   service_id=service_id,
                   port=port,
                   service_type=service_type_value)
        
        # Use the imported function to generate the model definition
        model_definition = gen_model_def(
            github_url=github_url,
            model_name=f"RAG Service for {github_info.repo.replace('-', ' ').replace('_', ' ').title()}",
            port=port,
            service_type=service_type_value,
            service_id=service_id  # Pass the service_id to use for paths
        )
        
        # Write the model definition to file
        write_model_definition(model_definition, model_def_path)
        
        print(f"Generated model definition: {model_def_path}")
        logger.info("Generated model definition", 
                   path=str(model_def_path),
                   service_id=service_id)
        
        return model_def_path
        
    except Exception as e:
        logger.error("Error generating model definition", error=str(e), traceback=traceback.format_exc())
        print(f"Error generating model definition: {str(e)}")
        return None
                

async def process_github_url(
    github_url: str, 
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    enable_chunking: bool = True,
    max_results: int = 5,
    base_model_name: str = 'gpt-4o',
    base_url: str = 'https://api.openai.com/v1',
    service_type: str = 'Gradio UI',
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Process a GitHub URL to create a RAG service using Backend.AI.
    
    Args:
        github_url: GitHub URL for documentation
        chunk_size: Size of chunks for document splitting
        chunk_overlap: Overlap between chunks
        enable_chunking: Whether to enable document chunking
        max_results: Number of chunks to retrieve for each query
        base_model_name: Base model name for the LLM
        base_url: Base URL for the API endpoint
        service_type: Type of RAG service to create (Gradio UI or FastAPI Server)
        progress_callback: Callback function to report progress
        
    Returns:
        Service information dictionary
    """
    try:
        # Load configuration
        config = load_config()
        path_config = config.paths
        llm_config = config.llm
        retrieval_settings = config.rag
        retrieval_settings.max_results = max_results

        # Get max_results from environment variable or use a default
        max_results = os.environ.get("MAX_RESULTS", max_results)
        # Setup environment
        setup_environment()
        
        # Generate unique service ID
        service_id = get_unique_service_id()
        logger.info("Generated service ID", service_id=service_id)
        
        # Create service directory
        service_dir = create_service_directory(service_id)
        docs_dir = service_dir / "docs"
        indices_dir = service_dir / "indices"
        indices_dir.mkdir(exist_ok=True)
        logger.info("Created service directories", 
                   service_dir=str(service_dir), 
                   docs_dir=str(docs_dir), 
                   indices_dir=str(indices_dir))
        
        # Update path configuration with service ID
        path_config.service_id = service_id
        
        # Initialize service info
        service_info = {
            "id": service_id,
            "github_url": github_url,
            "service_dir": service_dir,
            "docs_dir": docs_dir,
            "indices_dir": indices_dir,
            "status": ServiceStatus.PROCESSING,
            "url": None,
            "port": None,
            "pid": None,
            "model_def_path": None,
            "error": None,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "enable_chunking": enable_chunking,
            "max_results": max_results,
        }
        logger.info("Initialized service info", service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
        
        # Store service info in global state
        SERVICES[service_id] = service_info
        
        # Report progress
        if progress_callback:
            progress_callback(0.1, f"Created service directory: {service_dir}")
            
        # Process GitHub URL with custom chunking settings
        document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Report progress
        if progress_callback:
            progress_callback(0.2, "Cloning GitHub repository...")
            
        # Clone GitHub repository
        docs_path, error = await document_processor.clone_github_repository(
            github_url=github_url,
            target_dir=docs_dir
        )
        logger.info("Clone repository result", docs_path=str(docs_path) if docs_path else None, error=error)
        
        if error:
            service_info["status"] = ServiceStatus.ERROR
            service_info["error"] = f"Failed to clone repository: {error}"
            logger.error("Failed to clone repository", error=str(error), service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
            return service_info
            
        # Report progress
        if progress_callback:
            progress_callback(0.4, "Repository cloned successfully")
            
        # Initialize vector store with configuration
        vector_store = VectorStore(
            docs_root=docs_path, 
            indices_path=indices_dir,
            llm_config=llm_config,
            retrieval_settings=retrieval_settings,
            path_config=path_config,
            service_id=service_id,
        )
        logger.info("Initialized vector store", docs_path=str(docs_path), indices_dir=str(indices_dir))
        
        # Report progress
        if progress_callback:
            progress_callback(0.5, "Processing documentation...")
            
        # Process documents and create index
        try:
            # Pass the enable_chunking flag to collect_documents
            docs = await document_processor.collect_documents(
                docs_path, 
                chunk=enable_chunking
            )
            logger.info("Collected documents", 
                       docs_count=len(docs) if docs else 0,
                       chunking_enabled=enable_chunking,
                       chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap)
            
            if not docs:
                service_info["status"] = ServiceStatus.ERROR
                service_info["error"] = "No documents found in repository"
                logger.error("No documents found in repository", service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
                return service_info
                
            # Report progress
            if progress_callback:
                progress_callback(0.7, f"Found {len(docs)} documents, creating vector index...")
                
            # Create index
            await vector_store.create_indices(docs)
            logger.info("Created vector indices successfully")
            
            # Report progress
            if progress_callback:
                progress_callback(0.8, "Vector index created successfully")
                
        except Exception as e:
            service_info["status"] = ServiceStatus.ERROR
            service_info["error"] = f"Failed to process documents: {str(e)}"
            logger.error("Failed to process documents", error=str(e), service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
            return service_info
            
        # Generate model definition
        model_def_path = None
        try:
            model_def_path = generate_model_definition(github_url, service_dir, service_type)
            if model_def_path:
                logger.info("Generated model definition", path=str(model_def_path))
            else:
                logger.warning("Failed to generate model definition")
        except Exception as e:
            logger.error("Error generating model definition", error=str(e))
        
        if model_def_path:
            # Make sure to update the service_info with the model definition path
            service_info["model_def_path"] = str(model_def_path)
            # Save the updated service info to ensure it's persisted
            save_service_info(service_id, service_info)
            logger.info("Updated service info with model definition path", model_def_path=str(model_def_path))
        else:
            logger.warning("Failed to generate model definition")
            
        # Create Backend.AI scripts
        create_backend_scripts(service_id, service_dir)
        logger.info("Created Backend.AI scripts")
            
        # Report progress
        if progress_callback:
            progress_callback(0.9, "Preparing service...")
            
        # Update service status
        service_info["status"] = ServiceStatus.READY
        logger.info("Service ready to start", service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
        
        # Report progress
        if progress_callback:
            progress_callback(1.0, "Service ready to start")
            
        return service_info
        
    except Exception as e:
        logger.error("Error processing GitHub URL", error=str(e), traceback=traceback.format_exc())
        error_info = {
            "id": service_id if 'service_id' in locals() else str(uuid.uuid4()),
            "status": ServiceStatus.ERROR,
            "error": str(e)
        }
        logger.error("Returning error info", error_info=error_info)
        return error_info


async def start_service(service_id: str, service_type: str = "Gradio UI") -> Dict[str, Any]:
    """
    Start a RAG service as a Backend.AI model service.
    
    Args:
        service_id: Service ID
        service_type: Type of RAG service to create (Gradio UI or FastAPI Server)
        
    Returns:
        Updated service information
    """
    try:
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        logger.info("Starting service", service_id=service_id)
        
        if service_id not in SERVICES:
            error_msg = f"Service not found: {service_id}"
            logger.error(error_msg)
            return {
                "status": ServiceStatus.ERROR,
                "error": error_msg,
                "url": "",
                "model_def_path": "",
                "id": service_id
            }
            
        service_info = SERVICES[service_id]
        logger.info("Retrieved service info", 
                   service_id=service_id, 
                   service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
        
        if service_info["status"] != ServiceStatus.READY:
            error_msg = "Service is not ready to start"
            logger.error(error_msg, 
                        service_id=service_id, 
                        current_status=service_info["status"])
            return {
                "status": service_info["status"],
                "error": error_msg,
                "url": "",
                "model_def_path": "",
                "id": service_id
            }

        # Start service using Backend.AI
        def start_backend_service_thread():
            try:
                github_url = service_info["github_url"]
                service_dir = Path(service_info["service_dir"])
                
                # Set service_id in path_config
                path_config.service_id = service_id
                
                # Extract repository organization and name from GitHub URL
                match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', github_url)
                if match:
                    repo_org = match.group(1)
                    repo_name = match.group(2)
                    if "." in repo_name:
                        repo_name = repo_name.split(".")[0]
                        
                    service_name = f"rag_{repo_name}"
                else:
                    # Fallback if URL doesn't match expected pattern
                    repo_name = github_url.split("/")[-1]
                    if "." in repo_name:
                        repo_name = repo_name.split(".")[0]
                    service_name = f"rag_service"
                
                # Get model definition path
                model_def_path = service_info.get("model_def_path")
                if not model_def_path:
                    # Try multiple fallback mechanisms to find the model definition file
                    
                    # 1. Try standard naming convention in the service directory
                    possible_paths = [
                        service_dir / f"model-definition-{service_id}.yml",
                        service_dir / f"model-definition-{service_id}.yaml",
                    ]
                    
                    # 2. Try looking for any model definition file in the service directory
                    for file in service_dir.glob("model-definition-*.y*ml"):
                        possible_paths.append(file)
                    
                    # Check all possible paths
                    for path in possible_paths:
                        if path.exists():
                            model_def_path = str(path)
                            # Update service_info with the found path
                            service_info["model_def_path"] = model_def_path
                            save_service_info(service_id, service_info)
                            logger.info("Found and set model definition path", model_def_path=model_def_path)
                            break
                    
                    # If still not found, try to generate it
                    if not model_def_path:
                        try:
                            logger.info("Attempting to regenerate model definition", github_url=service_info["github_url"])
                            regenerated_path = generate_model_definition(service_info["github_url"], service_dir, service_type)
                            if regenerated_path:
                                model_def_path = str(regenerated_path)
                                service_info["model_def_path"] = model_def_path
                                save_service_info(service_id, service_info)
                                logger.info("Regenerated and set model definition path", model_def_path=model_def_path)
                            else:
                                raise ValueError("Failed to regenerate model definition")
                        except Exception as e:
                            logger.error("Failed to regenerate model definition", error=str(e))
                            raise ValueError("Model definition path not found in service info and could not be constructed")
                
                # Ensure model_def_path is a string
                if isinstance(model_def_path, Path):
                    model_def_path = str(model_def_path)
                
                # Get the model definition path relative to the backend model path
                # This is needed for Backend.AI service creation
                model_def_relative_path = model_def_path
                if isinstance(model_def_path, str) and "rag_services" in model_def_path:
                    # Extract the part of the path after rag_services
                    parts = model_def_path.split("rag_services/")
                    if len(parts) > 1:
                        model_def_relative_path = f"rag_services/{parts[1]}"
                        logger.info("Using relative model definition path", 
                                   original=model_def_path, 
                                   relative=model_def_relative_path)
                
                logger.info("Creating Backend.AI service", 
                           service_name=service_name,
                           model_def_path=model_def_relative_path)
                
                # Get backend model path from environment variable
                backend_model_path = os.environ.get("BACKEND_MODEL_PATH", "/models")
                
                # Create Backend.AI model service with environment variables
                create_service_cmd = [
                    "backend.ai", "service", "create",
                    "cr.backend.ai/testing/ngc-pytorch:24.12-pytorch2.6-py312-cuda12.6",
                    "auto_rag",
                    "1",
                    "--name", service_name,
                    "--tag", "rag_model_service",
                    "--scaling-group", "nvidia-H100",
                    "--model-definition-path", f"RAGModelService/{model_def_relative_path}",
                    "--public",
                    "-e", f"RAG_SERVICE_NAME={service_name}",
                    "-e", f"RAG_SERVICE_PATH={service_id}",
                    "-e", f"BACKEND_MODEL_PATH={backend_model_path}",
                    "-r", "cuda.shares=0",
                    "-r", "mem=4g",
                    "-r", "cpu=2"
                ]
                
                logger.info("Executing Backend.AI command", cmd=create_service_cmd)
                
                # Print the command for debugging
                print(f"Executing command: {' '.join(create_service_cmd)}")
                
                # Run the command
                create_result = subprocess.run(
                    create_service_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=path_config.project_path
                )
                
                # Print raw output for debugging
                print(f"Command stdout: {create_result.stdout}")
                print(f"Command stderr: {create_result.stderr}")
                print(f"Command return code: {create_result.returncode}")
                
                # Log the full output for debugging
                logger.info("Backend.AI command output", 
                           stdout=create_result.stdout,
                           stderr=create_result.stderr,
                           returncode=create_result.returncode)
                
                if create_result.returncode != 0:
                    error_msg = f"Backend.AI service creation failed: {create_result.stderr}"
                    logger.error("Backend.AI service creation failed",
                               stderr=create_result.stderr,
                               stdout=create_result.stdout,
                               returncode=create_result.returncode)
                    service_info["status"] = ServiceStatus.ERROR
                    service_info["error"] = error_msg
                    return
                
                logger.info("Backend.AI service creation output", 
                           stdout=create_result.stdout,
                           stderr=create_result.stderr)
                
                # Extract service endpoint and update service info
                service_url = None
                for line in create_result.stdout.split('\n'):
                    if "Service endpoint" in line:
                        service_url = line.split("Service endpoint:")[1].strip()
                        break
                
                if service_url:
                    service_info["url"] = service_url
                    logger.info(f"Service {service_id} started with URL {service_url}")
                else:
                    # If we couldn't extract the URL, build a default one
                    default_url = f"https://service.backend.ai/services/{service_name}"
                    service_info["url"] = default_url
                    logger.warning(f"Could not extract service URL, using default: {default_url}")
                
                # Update status
                service_info["status"] = ServiceStatus.READY
                logger.info("Service started successfully", service_id=service_id)
            except Exception as e:
                logger.error("Error in Backend.AI service creation thread", 
                            error=str(e), 
                            traceback=traceback.format_exc())
                service_info["status"] = ServiceStatus.ERROR
                service_info["error"] = str(e)
        
        # Start thread
        thread = threading.Thread(target=start_backend_service_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for service to start (briefly, to catch immediate failures)
        for i in range(10):
            logger.info(f"Waiting for Backend.AI service to start (attempt {i+1}/10)", 
                       service_id=service_id, 
                       status=service_info["status"])
            if service_info["status"] == ServiceStatus.ERROR:
                break
            await asyncio.sleep(0.5)
            
        # Set status to PROCESSING if it's not already ERROR
        if service_info["status"] != ServiceStatus.ERROR:
            service_info["status"] = ServiceStatus.PROCESSING
            logger.info("Setting service status to PROCESSING", 
                       service_id=service_id,
                       previous_status=service_info.get("status", "None"),
                       new_status=ServiceStatus.PROCESSING)
            
        logger.info("Returning service info from start_service", 
                   service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
        return service_info
        
    except Exception as e:
        logger.error("Error starting service", 
                    error=str(e), 
                    traceback=traceback.format_exc(), 
                    service_id=service_id)
        error_info = {
            "status": ServiceStatus.ERROR,
            "error": str(e),
            "url": "",
            "model_def_path": "",
            "id": service_id
        }
        logger.error("Returning error info from start_service", error_info=error_info)
        return error_info


def create_backend_scripts(service_id: str, service_dir: Path) -> None:
    """
    Create necessary scripts for Backend.AI service deployment.
    
    Args:
        service_id: Service ID
        service_dir: Path to service directory
    """
    # Load configuration
    config = load_config()
    path_config = config.paths
    max_results = os.environ.get("MAX_RESULTS") or "5"
    base_url = os.environ.get("BASE_URL") or "https://api.openai.com/v1"
    base_model_name = os.environ.get("BASE_MODEL_NAME") or "gpt-4o"
    # Set service_id in path_config
    path_config.service_id = service_id 
    
    # Create a start.sh script that Backend.AI will execute
    start_script = service_dir / "start.sh"
    
    # Get backend model path from environment variable
    backend_model_path = os.environ.get("BACKEND_MODEL_PATH", "/models")
    
    script_content = f"""#!/bin/bash
# Start script for RAG Service {service_id}

# Start the Gradio server with paths configured for Backend.AI
python -m interfaces.cli_app.launch_gradio \\
    --indices-path {backend_model_path}/RAGModelService/rag_services/{service_id}/indices \\
    --docs-path {backend_model_path}/RAGModelService/rag_services/{service_id}/docs \\
    --base_model_name {base_model_name} \\
    --base_url {base_url} \\
    --max-results {max_results} \\
    --service-id {service_id} \\
    --port 8000 \\
    --host 0.0.0.0
"""
    
    # Write the script
    with open(start_script, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(start_script, 0o755)
    
    logger.info("Created Backend.AI start script", script_path=str(start_script))


async def create_rag_service(
    github_url: str, 
    chunking_preset: str,
    chunk_size: int, 
    chunk_overlap: int,
    enable_chunking: bool = True,
    max_results: int = 5,
    base_url: str = "https://api.openai.com/v1",
    base_model_name: str = "gpt-4o",
    service_type: str = "Gradio UI",
    progress=gr.Progress()
) -> Tuple[str, str, str, str]:
    """
    Create a RAG service from a GitHub URL (Gradio interface function).
    
    Args:
        github_url: GitHub URL for documentation
        chunking_preset: Selected chunking preset (for logging only)
        chunk_size: Size of chunks for document splitting
        chunk_overlap: Overlap between chunks
        enable_chunking: Whether to enable document chunking
        progress: Gradio progress tracker
        base_url: Base URL for the API endpoint
        base_model_name: Base model name for the LLM
        service_type: Type of RAG service to create (Gradio UI or FastAPI Server)
        
    Returns:
        Tuple of (status, message, url, model_definition_path) for the Gradio interface
    """
    try:
        # Validate GitHub URL
        if not validate_github_url(github_url):
            error_msg = "Invalid GitHub URL"
            logger.error(error_msg, github_url=github_url, validation_result=False)
            return ("Error", error_msg, "", "")
        
        # Set the MAX_RESULTS environment variable
        os.environ["MAX_RESULTS"] = str(max_results)
        os.environ["BASE_MODEL_NAME"] = base_model_name
        os.environ["BASE_URL"] = base_url
        

        # Process GitHub URL with progress tracking
        def update_progress(progress_value, description):
            progress(progress_value, description)
            
        logger.info(
        "RAG service settings", 
            preset=chunking_preset, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            enabled=enable_chunking,
            max_results=max_results,  # Log max_results
            base_url=base_url,
            base_model_name=base_model_name,
            service_type=service_type
        )
        
        # Process GitHub URL with chunking parameters
        service_info = await process_github_url(
        github_url, 
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_chunking=enable_chunking,
            max_results=max_results, 
            base_model_name=base_model_name,
            base_url=base_url,
            service_type=service_type,
            progress_callback=update_progress
        )

        logger.info("Service info after processing GitHub URL", 
                   service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()},
                   service_status=service_info.get("status"),
                   service_status_type=type(service_info.get("status")).__name__ if service_info.get("status") else None,
                   expected_status=ServiceStatus.READY,
                   status_comparison=service_info.get("status") == ServiceStatus.READY)
        
        if service_info.get("status") == ServiceStatus.ERROR:
            error_msg = f"Failed to create service: {service_info.get('error', 'Unknown error')}"
            logger.error(error_msg, 
                        service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()},
                        error_details=service_info.get('error', 'Unknown error'))
            
            # Log the raw values being returned to the status boxes for service creation error
            logger.error("Raw service creation error values for status boxes", 
                       status_type=type("Error").__name__,
                       status_value="Error",
                       message_type=type(error_msg).__name__,
                       message_value=error_msg,
                       service_url_type=type("").__name__,
                       service_url_value="",
                       model_def_path_type=type("").__name__,
                       model_def_path_value="")
            
            # Create service creation error return tuple and log it
            error_tuple = ("Error", error_msg, "", "")
            logger.error("Service creation error return tuple for Gradio", 
                       return_tuple=error_tuple,
                       tuple_type=type(error_tuple).__name__,
                       tuple_length=len(error_tuple))
            
            return error_tuple
            
        # Start Backend.AI service
        progress(0.95, "Creating Backend.AI service...")
        service_info = await start_service(service_info["id"], service_type=service_type)
        logger.info("Service info after starting Backend.AI service", service_info=service_info)
        
        if service_info.get("status") == ServiceStatus.ERROR:
            error_msg = f"Failed to create Backend.AI service: {service_info.get('error', 'Unknown error')}"
            logger.error(error_msg, 
                        service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()},
                        error_details=service_info.get('error', 'Unknown error'))
            
            # Log the raw values being returned to the status boxes for service start error
            logger.error("Raw service start error values for status boxes", 
                       status_type=type("Error").__name__,
                       status_value="Error",
                       message_type=type(error_msg).__name__,
                       message_value=error_msg,
                       service_url_type=type(service_info['url']).__name__,
                       service_url_value=service_info['url'],
                       model_def_path_type=type(service_info['model_def_path']).__name__,
                       model_def_path_value=service_info['model_def_path'])
            
            # Create service start error return tuple and log it
            error_tuple = ("Error", error_msg, service_info['url'], service_info['model_def_path'])
            logger.error("Service start error return tuple for Gradio", 
                       return_tuple=error_tuple,
                       tuple_type=type(error_tuple).__name__,
                       tuple_length=len(error_tuple))
            
            return error_tuple
            
        # Return success only if service is actually ready
        progress(1.0, "Backend.AI service created successfully!")
        service_url = service_info.get("url", "")
        model_def_path = service_info.get("model_def_path", "")
        service_id = service_info.get("id", "")
        
        # Ensure service_url and model_def_path are strings, not None
        if service_url is None:
            service_url = ""
            logger.warning("Service URL is None, setting to empty string")
            
        if model_def_path is None:
            model_def_path = ""
            logger.warning("Model definition path is None, setting to empty string")
        
        # Check if service is in the correct state
        service_status = service_info.get("status", "")
        logger.info("Checking service status before returning", 
                   service_status=service_status,
                   expected_statuses=[ServiceStatus.READY, ServiceStatus.PROCESSING, ServiceStatus.RUNNING])
        
        # Only return Success if the service is in READY, PROCESSING, or RUNNING state
        if service_status not in [ServiceStatus.READY, ServiceStatus.PROCESSING, ServiceStatus.RUNNING]:
            error_msg = f"Backend.AI service in unexpected state: {service_status}"
            logger.error(error_msg, service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
            
            # Log the raw values being returned to the status boxes for unexpected state
            logger.error("Raw unexpected state error values for status boxes", 
                       status_type=type("Error").__name__,
                       status_value="Error",
                       message_type=type(error_msg).__name__,
                       message_value=error_msg,
                       service_url_type=type(service_url).__name__,
                       service_url_value=service_url,
                       model_def_path_type=type(model_def_path).__name__,
                       model_def_path_value=model_def_path)
            
            # Create unexpected state error return tuple and log it
            error_tuple = ("Error", error_msg, service_url, model_def_path)
            logger.error("Unexpected state error return tuple for Gradio", 
                       return_tuple=error_tuple,
                       tuple_type=type(error_tuple).__name__,
                       tuple_length=len(error_tuple))
            
            return error_tuple
        
        # Log the raw values being returned to the status boxes
        logger.info("Raw values for status boxes", 
                   status_type=type("Success").__name__,
                   status_value="Success",
                   message_type=type(f"RAG Service created with Backend.AI. Service ID: {service_info['id']}").__name__,
                   message_value=f"RAG Service created with Backend.AI. Service ID: {service_info['id']}",
                   service_url_type=type(service_url).__name__,
                   service_url_value=service_url,
                   model_def_path_type=type(model_def_path).__name__,
                   model_def_path_value=model_def_path)
        
        # Log the values being returned to the status boxes
        logger.info("Returning values to status boxes", 
                   status="Success",
                   message=f"RAG Service created with Backend.AI. Service ID: {service_info['id']}",
                   service_url=service_url,
                   model_def_path=model_def_path)
        
        # Create return tuple and log it
        return_tuple = (
            "Success", 
            f"RAG Service created with Backend.AI. Service ID: {service_info['id']}", 
            service_url, 
            model_def_path
        )
        logger.info("Final return tuple for Gradio", 
                   return_tuple=return_tuple,
                   tuple_type=type(return_tuple).__name__,
                   tuple_length=len(return_tuple))
        
        return return_tuple
        
    except Exception as e:
        logger.error("Error creating RAG service", error=str(e), traceback=traceback.format_exc())
        error_tuple = ("Error", f"Error creating RAG service: {str(e)}", "", "")
        return error_tuple


def create_interface() -> gr.Blocks:
    """Create the Gradio interface"""
    # Configure blocks with queue enabled for progress tracking
    blocks = gr.Blocks(
        title="RAG Service Creator", 
        theme=gr.themes.Base(),
        analytics_enabled=False,
    )
    
    # Enable queueing explicitly for the blocks instance
    blocks.queue()
    
    with blocks as interface:
        
        gr.Markdown(
            """
            # DosiRAG
            
            Create a RAG (Retrieval-Augmented Generation) service from a GitHub repository containing documentation.
            
            ## Instructions
            
            1. Enter a GitHub URL containing documentation
            2. Configure chunking settings if needed (expand Advanced Settings)
            3. Click 'Create RAG Service'
            4. Wait for the service to be created
            5. Open the service URL to use the RAG Chatbot
            """
        )
        
        with gr.Row():
            github_url = gr.Textbox(
                label="GitHub URL",
                placeholder="Enter GitHub repository URL (e.g., https://github.com/owner/repo)",
            )
            create_button = gr.Button("Create RAG Service", variant="primary")
            
        # Add advanced settings in a collapsible section
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                enable_chunking = gr.Checkbox(
                    label="Enable Document Chunking", 
                    value=True,
                    info="Split documents into smaller chunks for better retrieval"
                )
            with gr.Row():
                max_results = gr.Slider(
                label="Number of Retrieved Chunks",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    info="Controls how many document chunks are retrieved for each query",
                    interactive=True
                )   
            with gr.Row():
                chunking_preset = gr.Radio(
                    label="Chunking Strategy",
                    choices=["Fine-grained", "Balanced", "Contextual"],
                    value="Balanced",
                    info="Choose how documents should be divided into chunks for retrieval",
                    interactive=True
                )
            
            with gr.Row():
                with gr.Column():
                    chunk_size_slider = gr.Slider(
                        label="Chunk Size",
                        minimum=250,
                        maximum=4000,
                        value=1000,
                        step=250,
                        info="Controls the size of each chunk (characters)",
                        interactive=True
                    )
                
                with gr.Column():
                    chunk_overlap_slider = gr.Slider(
                        label="Context Overlap",
                        minimum=0,
                        maximum=800,
                        value=200,
                        step=50,
                        info="Controls how much context is shared between chunks",
                        interactive=True
                    )
            
            with gr.Row():
                base_url = gr.Textbox(
                    label='Base URL',
                    value='https://qwen_qwq_32b.asia03.app.backend.ai/v1',
                    placeholder='Enter the base URL for the API endpoint'
                )
                base_model_name = gr.Textbox(
                    label='Base Model Name',
                    value='QwQ-32B',
                    placeholder='Enter the base model name for the LLM'
                )
            
            with gr.Row():
                service_type = gr.Radio(
                    label="Service Type",
                    choices=["Gradio UI", "FastAPI Server"],
                    value="Gradio UI",
                    info="Choose the type of RAG service to create",
                    interactive=True
                )
            
            # Helper text explaining the chunking settings
            gr.Markdown(
                """
                ### About Chunking Settings
                
                - **Fine-grained**: Smaller chunks (500 chars, 100 overlap) for precise answers but may miss context
                - **Balanced**: Medium chunks (1000 chars, 200 overlap) for good balance of precision and context
                - **Contextual**: Larger chunks (2000 chars, 400 overlap) prioritizing context at the cost of precision
                
                Adjust the sliders to customize your chunking strategy beyond the presets.
                """
            )
            
            # Add event handler to update sliders based on preset selection
            def update_sliders(preset):
                if preset == "Fine-grained":
                    return 500, 100
                elif preset == "Balanced":
                    return 1000, 200
                elif preset == "Contextual":
                    return 2000, 400
                return 1000, 200  # Default
                
            chunking_preset.change(
                fn=update_sliders,
                inputs=chunking_preset,
                outputs=[chunk_size_slider, chunk_overlap_slider]
            )
        
        # Create a consolidated status box
        with gr.Row():
            consolidated_status = gr.Markdown(
                value="### Status: Ready to create service\n\nEnter a GitHub URL and click 'Create RAG Service'",
                elem_id="consolidated_status"
            )
            
        # Hidden components to store original values
        status_hidden = gr.Textbox(visible=False, value="Ready to create service")
        message_hidden = gr.Textbox(visible=False, value="Enter a GitHub URL and click 'Create RAG Service'")
        service_url_hidden = gr.Textbox(visible=False, value="")
        model_def_path_hidden = gr.Textbox(visible=False, value="")
            
        with gr.Row():
            open_button = gr.Button("Open Service", interactive=False)
            
        # Function to update the consolidated status box
        def update_consolidated_status(status, message, service_url, model_def_path):
            status_icon = "✅" if status == "Success" else "❌" if status == "Error" else "⏳"
            
            consolidated_text = f"### Status: {status} {status_icon}\n\n"
            consolidated_text += f"{message}\n\n"
            
            if service_url:
                consolidated_text += f"**Service URL**: {service_url}\n\n"
                
            if model_def_path:
                consolidated_text += f"**Model Definition**: {model_def_path}"
                
            return consolidated_text, status, message, service_url, model_def_path
        
        # Function to update button interactivity
        def update_button_state(service_url):
            return bool(service_url)
            
        # Button click event
        create_button.click(
            create_rag_service,
            inputs=[github_url, chunking_preset, chunk_size_slider, chunk_overlap_slider, 
                    enable_chunking, max_results, base_url, base_model_name, service_type],
            outputs=[status_hidden, message_hidden, service_url_hidden, model_def_path_hidden],
        ).then(
            update_consolidated_status,
            inputs=[status_hidden, message_hidden, service_url_hidden, model_def_path_hidden],
            outputs=[consolidated_status, status_hidden, message_hidden, service_url_hidden, model_def_path_hidden],
        ).then(
            update_button_state,
            inputs=[service_url_hidden],
            outputs=[open_button],
        )
        
        # Open service button
        def open_service(url):
            import webbrowser
            if url:
                webbrowser.open(url)
                return f"Opening {url}..."
            return "No service URL to open"
            
        open_button.click(
            open_service,
            inputs=[service_url_hidden],
            outputs=[message_hidden],
        ).then(
            update_consolidated_status,
            inputs=[status_hidden, message_hidden, service_url_hidden, model_def_path_hidden],
            outputs=[consolidated_status, status_hidden, message_hidden, service_url_hidden, model_def_path_hidden],
        ).then(
            update_button_state,
            inputs=[service_url_hidden],
            outputs=[open_button],
        )
        
        # Example buttons
        with gr.Row():
            gr.Markdown("### Example Repositories")
            
        with gr.Row():
            example_1 = gr.Button("LangChain")
            example_2 = gr.Button("TensorRT-LLM")
            example_3 = gr.Button("vLLM")
            
        example_1.click(
            lambda: "https://github.com/langchain-ai/langchain/tree/master/docs",
            outputs=[github_url],
        )
        example_2.click(
            lambda: "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs",
            outputs=[github_url],
        )
        example_3.click(
            lambda: "https://github.com/vllm-project/vllm/tree/main/docs",
            outputs=[github_url],
        )
        
    return interface


async def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code
    """
    try:
        # Setup environment
        if not setup_environment():
            logger.error("Failed to set up environment")
            print("Error: Failed to set up environment")
            return 1
            
        interface = create_interface()
        
        print("Launching Gradio server for RAG Service Creator...")
        print("URL: http://localhost:8000")
        
        # Launch server with settings to keep the app running
        interface.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=True,
            debug=True,
            prevent_thread_lock=False  # Keep the main thread locked to prevent exit
        )
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/portal/generate_model_definition.py

```python
#!/usr/bin/env python3
"""
Generate Model Definition

This script generates a model definition YAML file for a RAG service based on a GitHub URL.
It extracts the repository name from the URL to use as the documentation name.

Usage:
    python generate_model_definition.py --github-url https://github.com/owner/repo
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import configuration
from config.config import load_config

# Load configuration
config = load_config()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate model definition YAML for RAG service"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Output file
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the model definition file",
        default=".",
    )
    
    # Model name prefix
    parser.add_argument(
        "--name-prefix",
        type=str,
        help="Prefix for the model name",
        default="RAG Service for",
    )
    
    # Port
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the service",
        default=None,
    )
    
    # Service type
    parser.add_argument(
        "--service-type",
        type=str,
        help="Type of service (gradio or fastapi)",
        choices=["gradio", "fastapi", "Gradio UI", "FastAPI Server"],
        default=None,
    )
    
    return parser.parse_args()


def parse_github_url(github_url: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Parse a GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple of (owner, repo, branch, path)
    """
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Match GitHub URL with optional branch and path (tree format)
    tree_pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    tree_match = re.match(tree_pattern, github_url)
    
    if tree_match:
        owner = tree_match.group(1)
        repo = tree_match.group(2)
        branch = tree_match.group(3)  # This will be None if branch is not specified
        path = tree_match.group(4)    # This will be None if path is not specified
        return owner, repo, branch, path
    
    # Match GitHub URL with blob format
    blob_pattern = r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
    blob_match = re.match(blob_pattern, github_url)
    
    if blob_match:
        owner = blob_match.group(1)
        repo = blob_match.group(2)
        branch = blob_match.group(3)
        file_path = blob_match.group(4)
        
        # For blob URLs, extract the directory path
        # First, check if the file path contains a directory
        if '/' in file_path:
            # Get the directory part of the path (everything before the last slash)
            dir_path = '/'.join(file_path.split('/')[:-1])
            return owner, repo, branch, dir_path
        else:
            # It's a file in the root directory
            return owner, repo, branch, None
    
    # Basic GitHub URL (just owner/repo)
    basic_pattern = r"https?://github\.com/([^/]+)/([^/]+)"
    basic_match = re.match(basic_pattern, github_url)
    
    if basic_match:
        owner = basic_match.group(1)
        repo = basic_match.group(2)
        return owner, repo, None, None
    
    # If URL doesn't match any expected patterns
    raise ValueError(f"Invalid GitHub URL: {github_url}")


def generate_model_name(owner: str, repo: str, path: Optional[str], prefix: str) -> str:
    """
    Generate a model name based on the GitHub URL components.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        path: Path within the repository (if any)
        prefix: Prefix for the model name
        
    Returns:
        Model name
    """
    # Use the full repository name
    repo_name = repo
    
    # Format model name
    if path:
        if path == "docs":
            return f"{prefix} {repo_name} Documentation"
        else:
            return f"{prefix} {repo_name} Documentation ({path})"
    else:
        return f"{prefix} {repo_name}"


def generate_docs_name(owner: str, repo: str, path: Optional[str]) -> str:
    """
    Generate a documentation name for the YAML filename based on the GitHub URL components.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        path: Path within the repository (if any)
        
    Returns:
        Documentation name
    """
    # Use the full repository name (lowercase)
    repo_name = repo.lower()
    
    # Format docs name
    if path:
        # Replace slashes with hyphens and remove any special characters
        path_part = re.sub(r'[^a-zA-Z0-9-]', '', path.replace('/', '-'))
        return f"{repo_name}-{path_part}"
    else:
        return repo_name


def generate_model_definition(github_url: str, model_name: str, port: int = None, service_type: str = None, service_id: str = None) -> Dict:
    """
    Generate a model definition for the RAG service.
    
    Args:
        github_url: GitHub URL
        model_name: Model name
        port: Port number (if None, will use the value from config)
        service_type: Service type (gradio or fastapi) (if None, will use the value from config)
        service_id: Service ID (if None, will be generated from GitHub URL)
        
    Returns:
        Model definition as a dictionary
    """
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Get the current MAX_RESULTS from environment
    max_results = os.environ.get("MAX_RESULTS", "5")
    base_model_name = os.environ.get("BASE_MODEL_NAME", config.llm.model_name)
    base_url = os.environ.get("BASE_URL", config.llm.base_url)

    # Use provided values or defaults from config
    if port is None:
        # Use server config instead of service config
        port = config.server.port if hasattr(config, 'server') else 8000
    
    if service_type is None:
        # Default to gradio if not specified
        service_type = "gradio"
    
    # Parse the GitHub URL
    owner, repo, branch, path = parse_github_url(github_url)
    
    # Determine the docs path argument
    docs_path_arg = path if path else ""
    
    # Use the provided service_id or generate one from the GitHub URL
    if service_id is None:
        # For backward compatibility, but this should not be used
        service_id = f"{owner}/{repo}"
    
    # Update path configuration with service ID
    path_config.service_id = service_id
    
    # Get BACKEND_MODEL_PATH from environment variable or use a default
    backend_model_path = os.environ.get("BACKEND_MODEL_PATH", "/models")
    
    # Get RAG_SERVICE_PATH from environment variable or use a default
    rag_service_path = os.environ.get("RAG_SERVICE_PATH", f"{backend_model_path}/RAGModelService/rag_services/")
    
    # Ensure rag_service_path ends with a slash
    if not rag_service_path.endswith('/'):
        rag_service_path += '/'
    
    # Build the service-specific paths using the configuration
    service_dir_path = f"{rag_service_path}{service_id}"
    indices_path = f"{service_dir_path}/indices"
    docs_path = f"{service_dir_path}/docs"
    
    # Determine the start command based on service type
    if service_type in ['gradio', 'Gradio UI']:
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/cli_app/launch_gradio.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--max-results',
            str(max_results),
            '--base_model_name',
            base_model_name,
            '--base_url',
            base_url,
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    elif service_type in ['fastapi', 'FastAPI Server']:  # fastapi
        start_command = [
            'python3',
            f'{backend_model_path}/RAGModelService/interfaces/fastapi_app/fastapi_server.py',
            '--indices-path',
            indices_path,
            '--docs-path',
            docs_path,
            '--max-results',
            str(max_results),
            '--service-id',
            service_id,
            '--host',
            '0.0.0.0',
            '--port',
            str(port)
        ]
    
    # Create the model definition
    model_definition = {
        'models': [
            {
                'name': model_name,
                'model_path': '/models',
                'service': {
                    'port': port,
                    'pre_start_actions': [
                        {
                            'action': 'run_command',
                            'args': {
                                'command': ['/bin/bash', f'{backend_model_path}/RAGModelService/deployment/scripts/setup_{"gradio" if service_type in ["gradio", "Gradio UI"] else "fastapi"}.sh']
                            }
                        }
                    ],
                    'start_command': start_command
                }
            }
        ]
    }
    
    return model_definition


def write_model_definition(model_definition: Dict, output_path: Path) -> None:
    """
    Write model definition to YAML file.
    
    Args:
        model_definition: Model definition dictionary
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        yaml.dump(model_definition, f, default_flow_style=False)
    
    print(f"Model definition written to {output_path}")


def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Parse GitHub URL
        owner, repo, branch, path = parse_github_url(args.github_url)
        
        # Generate model name
        model_name = generate_model_name(owner, repo, path, args.name_prefix)
        
        # Generate docs name for the filename
        docs_name = generate_docs_name(owner, repo, path)
        
        # Generate model definition
        model_definition = generate_model_definition(
            args.github_url,
            model_name,
            args.port,
            args.service_type
        )
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file name
        output_path = output_dir / f"model-definition-{docs_name}.yaml"
        
        # Write model definition to file
        write_model_definition(model_definition, output_path)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/interfaces/portal/github.py

```python
#!/usr/bin/env python3
"""
GitHub Repository Handler for RAG Service

This module provides functionality for:
1. Parsing GitHub repository URLs
2. Cloning GitHub repositories
3. Preparing documentation for RAG processing

Note: This module now delegates to utils.github_utils for core GitHub functionality.
"""

import os
import asyncio
from pathlib import Path
from typing import Tuple, Optional

import structlog

# Import centralized GitHub utilities
from utils.github_utils import parse_github_url as utils_parse_github_url
from utils.github_utils import clone_github_repo as utils_clone_github_repo

# Initialize logger
logger = structlog.get_logger()


def parse_github_url(github_url: str) -> Tuple[str, str, str, str]:
    """
    Parse GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple containing owner, repo, branch, and path
    """
    # Use the centralized implementation but convert the return format
    github_info = utils_parse_github_url(github_url)
    return github_info.owner, github_info.repo, github_info.branch, github_info.path


async def clone_github_repo(github_url: str, target_dir: Path) -> Path:
    """
    Clone repository from GitHub URL to target directory.
    If a specific path is provided in the GitHub URL, only that directory is cloned.
    
    Args:
        github_url: GitHub URL
        target_dir: Target directory for cloned repository
        
    Returns:
        Path to documentation directory
    """
    try:
        # Parse GitHub URL using centralized utility
        github_info = utils_parse_github_url(github_url)
        
        logger.info(
            "Parsed GitHub URL",
            owner=github_info.owner,
            repo=github_info.repo,
            branch=github_info.branch,
            path=github_info.path,
        )
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Clone repository using centralized utility
        if github_info.path:
            print(f"Sparse cloning {github_info.path} from {github_info.owner}/{github_info.repo} repository...")
        else:
            print(f"Cloning {github_info.owner}/{github_info.repo} repository...")
        
        docs_path, error = utils_clone_github_repo(github_info, target_dir)
        
        if error:
            raise error
            
        # Create docs directory if it doesn't exist
        docs_target = target_dir / "docs"
        docs_target.mkdir(exist_ok=True, parents=True)
        
        # If docs_path is different from docs_target, copy the contents
        if docs_path != docs_target and docs_path.exists():
            print(f"Copying documentation from {docs_path} to {docs_target}...")
            # Use os.system instead of subprocess to simplify the async environment
            os.system(f"cp -r {str(docs_path)}/. {str(docs_target)}")
            print(f"Documentation copied successfully to {docs_target}")
            return docs_target
        else:
            return docs_path
        
    except Exception as e:
        logger.error("Error cloning repository", error=str(e))
        print(f"Error cloning repository: {str(e)}")
        raise


def prepare_for_rag(github_url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Prepare GitHub repository for RAG processing.
    
    Args:
        github_url: GitHub URL
        output_dir: Output directory for cloned repository (default: ./github_docs)
        
    Returns:
        Path to documentation directory
    """
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = Path("./github_docs")
        
        # Ensure output directory is a Path object
        output_dir = Path(output_dir)
        
        # Clone repository
        docs_path = asyncio.run(clone_github_repo(github_url, output_dir))
        
        print(f"Repository prepared for RAG processing. Documentation path: {docs_path}")
        
        return docs_path
        
    except Exception as e:
        logger.error("Error preparing repository for RAG", error=str(e))
        print(f"Error preparing repository for RAG: {str(e)}")
        raise
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/codebase_to_markdown.py

```python
#!/usr/bin/env python3
"""
Codebase to Markdown Converter

This script traverses directories to find code files (Python, YAML, Shell scripts, etc.) 
and merges them into a single Markdown file.
Each file is represented as a code block in the Markdown file with its path as a heading.

Example usage:
python scripts/codebase_to_markdown.py --root /Users/sergeyleksikov/Documents/GitHub/RAGModelService --output codebase_context.md --relative --exclude "*venv*" "*__pycache__*" "*bai_manager*"

"""

import os
import argparse
from pathlib import Path
import fnmatch


def get_file_extension(file_path):
    """Get the file extension for syntax highlighting in markdown."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.py':
        return 'python'
    elif ext in ['.yaml', '.yml']:
        return 'yaml'
    elif ext == '.sh':
        return 'bash'
    elif ext == '.md':
        return 'markdown'
    elif ext in ['.env', '.env.example']:
        return 'ini'
    elif ext == '.json':
        return 'json'
    elif ext == '.js':
        return 'javascript'
    elif ext == '.html':
        return 'html'
    elif ext == '.css':
        return 'css'
    else:
        return ''


def collect_code_files(root_dir, file_extensions, exclude_patterns=None):
    """
    Recursively collect code files with specified extensions from the given directory.
    
    Args:
        root_dir (str): Root directory to start the search from
        file_extensions (list): List of file extensions to include (e.g., ['.py', '.yaml'])
        exclude_patterns (list): List of glob patterns to exclude
        
    Returns:
        list: List of paths to code files
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    code_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs_to_remove = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if any(fnmatch.fnmatch(dir_path, pattern) for pattern in exclude_patterns):
                dirs_to_remove.append(d)
        
        for d in dirs_to_remove:
            dirs.remove(d)
            
        # Collect files with specified extensions
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if file has one of the specified extensions
            if file_ext in file_extensions or file.endswith('.env.example'):
                if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                    code_files.append(file_path)
    
    return code_files


def merge_to_markdown(code_files, output_file, relative_to=None):
    """
    Merge code files into a single Markdown file.
    
    Args:
        code_files (list): List of paths to code files
        output_file (str): Path to the output Markdown file
        relative_to (str): Directory to make paths relative to
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Codebase Collection\n\n")
        md_file.write("This document contains merged code files for LLM context.\n\n")
        
        for file_path in sorted(code_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as code_file:
                    content = code_file.read()
                
                # Make path relative if specified
                display_path = file_path
                if relative_to:
                    try:
                        display_path = os.path.relpath(file_path, relative_to)
                    except ValueError:
                        # Keep absolute path if relpath fails
                        pass
                
                # Get the appropriate language for syntax highlighting
                language = get_file_extension(file_path)
                
                md_file.write(f"## {display_path}\n\n")
                md_file.write(f"```{language}\n")
                md_file.write(content)
                if not content.endswith('\n'):
                    md_file.write('\n')
                md_file.write("```\n\n")
                
                print(f"Added {display_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert code files to a single Markdown file')
    parser.add_argument('--root', '-r', type=str, required=True, 
                        help='Root directory to search for code files')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Output Markdown file path')
    parser.add_argument('--exclude', '-e', type=str, nargs='+', default=[],
                        help='Glob patterns to exclude (e.g. "*venv*" "*/tests/*")')
    parser.add_argument('--relative', action='store_true',
                        help='Make file paths relative to the root directory')
    parser.add_argument('--extensions', type=str, nargs='+', 
                        default=['.py', '.yaml', '.yml', '.sh', '.env', '.env.example'],
                        help='File extensions to include (e.g. .py .yaml .sh)')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root)
    output_file = args.output
    exclude_patterns = args.exclude
    file_extensions = args.extensions
    
    print(f"Searching for files with extensions {file_extensions} in: {root_dir}")
    print(f"Excluding patterns: {exclude_patterns}")
    
    code_files = collect_code_files(root_dir, file_extensions, exclude_patterns)
    print(f"Found {len(code_files)} code files")
    
    relative_to = root_dir if args.relative else None
    merge_to_markdown(code_files, output_file, relative_to)
    
    print(f"Successfully merged code files into: {output_file}")


if __name__ == "__main__":
    main()
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/debug/test_backend_ai.py

```python
#!/usr/bin/env python3
"""
Test script for backend.ai command execution
"""

import os
import subprocess
import sys
import time

def main():
    """
    Test backend.ai command execution
    """
    print("Testing backend.ai command execution")
    
    # Check if backend.ai is in PATH
    which_backend_cmd = ["which", "backend.ai"]
    try:
        which_result = subprocess.run(which_backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"backend.ai path: {which_result.stdout.strip()}")
        if which_result.returncode != 0:
            print(f"Error finding backend.ai: {which_result.stderr}")
            return
    except Exception as e:
        print(f"Error checking backend.ai path: {str(e)}")
        return

    # Check if backend.ai is authenticated
    auth_check_cmd = ["backend.ai", "session", "list", "--limit", "1"]
    try:
        auth_result = subprocess.run(auth_check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        print(f"Auth check return code: {auth_result.returncode}")
        if auth_result.returncode != 0:
            print(f"Auth check error: {auth_result.stderr}")
            print("backend.ai might not be authenticated. Try running 'backend.ai login' manually first.")
            return
        else:
            print("backend.ai appears to be authenticated")
    except Exception as e:
        print(f"Error checking backend.ai authentication: {str(e)}")
        return

    # Simple test command
    test_cmd = ["backend.ai", "session", "list", "--limit", "5"]
    print(f"Running test command: {' '.join(test_cmd)}")
    
    # Method 1: subprocess.run with shell=False
    try:
        print("\nMethod 1: subprocess.run with shell=False")
        start_time = time.time()
        result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        end_time = time.time()
        print(f"Command completed in {end_time - start_time:.2f} seconds with return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running command: {str(e)}")

    # Method 2: subprocess.run with shell=True
    try:
        print("\nMethod 2: subprocess.run with shell=True")
        shell_cmd = ' '.join(test_cmd)
        start_time = time.time()
        result = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        end_time = time.time()
        print(f"Command completed in {end_time - start_time:.2f} seconds with return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running command: {str(e)}")

    # Method 3: os.system
    try:
        print("\nMethod 3: os.system")
        shell_cmd = ' '.join(test_cmd)
        start_time = time.time()
        return_code = os.system(shell_cmd)
        end_time = time.time()
        print(f"Command completed in {end_time - start_time:.2f} seconds with return code: {return_code}")
    except Exception as e:
        print(f"Error running command: {str(e)}")

    print("\nTest completed")

if __name__ == "__main__":
    main()
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/debug/test_github_cli.sh

```bash
#!/bin/bash
# Test script for github_cli.py
# Tests the ability to clone only a specific directory from a GitHub repository

echo "Testing GitHub CLI tool with sparse checkout functionality"
echo "=========================================================="

# Test URL with a specific path (docs directory)
TEST_URL="https://github.com/vllm-project/vllm/tree/main/docs"
TEST_DIR="./test_github_cli_output"

# Clean up any existing test directory
if [ -d "$TEST_DIR" ]; then
    echo "Cleaning up existing test directory: $TEST_DIR"
    rm -rf "$TEST_DIR"
fi

# Create test directory
mkdir -p "$TEST_DIR"

# Test the parse command
echo -e "\n1. Testing 'parse' command..."
python interfaces/cli_app/github_cli.py parse "$TEST_URL"

# Test the clone command
echo -e "\n2. Testing 'clone' command with specific path..."
python interfaces/cli_app/github_cli.py clone "$TEST_URL" --output-dir "$TEST_DIR/clone_test"

# List the contents of the cloned directory
echo -e "\nContents of cloned directory:"
ls -la "$TEST_DIR/clone_test"

# Test the prepare command
echo -e "\n3. Testing 'prepare' command with specific path..."
python interfaces/cli_app/github_cli.py prepare "$TEST_URL" --output-dir "$TEST_DIR/prepare_test"

# List the contents of the prepared directory
echo -e "\nContents of prepared directory:"
ls -la "$TEST_DIR/prepare_test"

echo -e "\nTest completed!"
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/debug/test_model_definition.py

```python
#!/usr/bin/env python3
"""
Test Model Definition Generator

This script tests the model definition generator with various GitHub URL formats.
"""

import os
import sys
from pathlib import Path
import tempfile
import yaml

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the functions to test
from interfaces.portal.generate_model_definition import (
    parse_github_url,
    generate_model_name,
    generate_docs_name
)

# Import our local implementation for model definition generation
from interfaces.cli_app.generate_model_definition_cli import generate_model_definition_local
from config.config import BACKEND_MODEL_PATH

def test_url_parsing():
    """Test URL parsing with various GitHub URL formats."""
    test_urls = [
        "https://github.com/lablup/backend.ai",
        "https://github.com/lablup/backend.ai/tree/main",
        "https://github.com/lablup/backend.ai/tree/main/docs",
        "https://github.com/reflex-dev/reflex-web/tree/main/docs",
        "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs",
        "https://github.com/pytorch/pytorch/tree/master/docs/source",
        # Test URLs with 'blob' instead of 'tree'
        "https://github.com/lablup/backend.ai/blob/main/README.md",
        "https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md",
        # Test URLs with different branch names
        "https://github.com/huggingface/transformers/tree/v4.28.0/docs",
        # Test URLs with trailing slashes
        "https://github.com/lablup/backend.ai/",
        "https://github.com/lablup/backend.ai/tree/main/docs/"
    ]
    
    print("Testing URL parsing...")
    for url in test_urls:
        owner, repo, branch, path = parse_github_url(url)
        model_name = generate_model_name(owner, repo, path, "RAG Service for")
        docs_name = generate_docs_name(owner, repo, path)
        
        print(f"\nURL: {url}")
        print(f"  Owner: {owner}")
        print(f"  Repo: {repo}")
        print(f"  Branch: {branch}")
        print(f"  Path: {path}")
        print(f"  Model Name: {model_name}")
        print(f"  Docs Name: {docs_name}")
        print(f"  YAML Filename: model-definition-{docs_name}.yaml")

def test_blob_url_parsing():
    """Test parsing of GitHub blob URLs specifically."""
    test_urls = [
        "https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md",
        "https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md",
        "https://github.com/huggingface/transformers/blob/main/docs/source/en/quicktour.md"
    ]
    
    print("\nTesting blob URL parsing...")
    for url in test_urls:
        owner, repo, branch, path = parse_github_url(url)
        
        print(f"\nURL: {url}")
        print(f"  Owner: {owner}")
        print(f"  Repo: {repo}")
        print(f"  Branch: {branch}")
        print(f"  Path: {path}")
        
        # Generate a model definition with this URL
        model_name = generate_model_name(owner, repo, path, "RAG Service for")
        service_id = "test123"
        model_def = generate_model_definition_local(url, model_name, 8000, "gradio", service_id)
        
        print(f"  Model Name: {model_name}")
        # Find the docs-path argument in the start command
        docs_path_index = model_def['models'][0]['service']['start_command'].index('--docs-path') + 1
        docs_path = model_def['models'][0]['service']['start_command'][docs_path_index]
        print(f"  Docs Path: {docs_path}")

def test_model_definition_generation():
    """Test model definition generation with various GitHub URL formats."""
    test_urls = [
        "https://github.com/lablup/backend.ai",
        "https://github.com/reflex-dev/reflex-web/tree/main/docs",
        "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs",
        "https://github.com/pytorch/pytorch/tree/master/docs/source",
    ]
    
    print("\nTesting model definition generation...")
    for url in test_urls:
        owner, repo, branch, path = parse_github_url(url)
        
        # Generate a model name
        model_name = generate_model_name(owner, repo, path, "RAG Service for")
        service_id = "test456"
        
        # Generate a model definition
        model_def = generate_model_definition_local(url, model_name, 8000, "gradio", service_id)
        
        print(f"\nURL: {url}")
        print(f"  Model Name: {model_name}")
        print(f"  Path: {path}")
        print("  Model Definition:")
        
        # Print the model definition in a readable format
        model_info = model_def["models"][0]
        service_value = model_info["service"]
        
        print(f"    name: {model_info['name']}")
        print(f"    model_path: {model_info['model_path']}")
        print(f"    service:")
        print(f"      port: {service_value['port']}")
        print(f"      pre_start_actions: {len(service_value['pre_start_actions'])} action(s)")
        
        # Find docs-path in start command
        docs_path_index = service_value['start_command'].index('--docs-path') + 1
        docs_path = service_value['start_command'][docs_path_index]
        print(f"      docs_path: {docs_path}")
        
        # Find service-id in start command
        service_id_index = service_value['start_command'].index('--service-id') + 1
        service_id_value = service_value['start_command'][service_id_index]
        print(f"      service_id: {service_id_value}")

def test_write_model_definition():
    """Test writing model definition to a YAML file."""
    url = "https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs"
    owner, repo, branch, path = parse_github_url(url)
    model_name = generate_model_name(owner, repo, path, "RAG Service for")
    service_id = "test789"
    
    # Generate a model definition
    model_def = generate_model_definition_local(url, model_name, 8000, "gradio", service_id)
    
    # Create a temporary file to write the model definition
    with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Write the model definition to the temporary file
        with open(temp_path, 'w') as f:
            yaml.dump(model_def, f, default_flow_style=False)
        
        # Read the model definition back from the file
        with open(temp_path, 'r') as f:
            loaded_def = yaml.safe_load(f)
        
        # Verify that the loaded definition matches the original
        print("\nTesting model definition writing...")
        print(f"  Original model name: {model_def['models'][0]['name']}")
        print(f"  Loaded model name: {loaded_def['models'][0]['name']}")
        print(f"  Match: {model_def['models'][0]['name'] == loaded_def['models'][0]['name']}")
        
        # Verify that the start command is preserved
        original_start = model_def['models'][0]['service']['start_command']
        loaded_start = loaded_def['models'][0]['service']['start_command']
        print(f"  Start command preserved: {original_start == loaded_start}")
        
        print(f"  Model definition successfully written to and read from: {temp_path}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

def main():
    """Main function."""
    test_url_parsing()
    test_blob_url_parsing()
    test_model_definition_generation()
    test_write_model_definition()
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/debug/test_model_service_create.sh

```bash
bai_manager/backendai/backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  auto_rag \
  1 \
  --name rag_reflex-web2 \
  --tag rag_model_service \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models/RAGModelService/rag_services/dafced48/model-definition-reflex-dev-reflex-web-docs.yaml \
  --public \
  -e RAG_SERVICE_NAME=rag_reflex-web \
  -e RAG_SERVICE_PATH=rag_services/dafced48 \
  -r mem=4g \
  -r cpu=2 \
  --bootstrap-script ./auto_rag_service/setup.sh \
  --startup-command "python3 /models/RAGModelService/auto_rag_service/start.sh"


bai_manager/backendai/backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  auto_rag \
  1 \
  --name rag_reflex-web2 \
  --tag rag_model_service \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models \
  --public \
  -e RAG_SERVICE_NAME=rag_reflex-web \
  -e RAG_SERVICE_PATH=rag_services/dafced48 \
  -r mem=4g \
  -r cpu=2 \
  --model-definition-path ./RAGModelService/rag_services/dafced48/model-definition-reflex-dev-reflex-web-docs.yaml
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/debug/test_sparse_checkout.py

```python
#!/usr/bin/env python3
"""
Test script for sparse checkout functionality.

This script tests the ability to clone only a specific directory from a GitHub repository
instead of the entire repository.
"""

import os
import sys
import shutil
import asyncio
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.github_utils import parse_github_url, clone_github_repo
from interfaces.portal.github import clone_github_repo as portal_clone_github_repo


async def test_sparse_checkout():
    """Test the sparse checkout functionality"""
    # Test URL with a specific path (docs directory)
    test_url = "https://github.com/vllm-project/vllm/tree/main/docs"
    
    # Create a temporary directory for testing
    test_dir = Path("./test_sparse_checkout")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    print(f"Testing sparse checkout with URL: {test_url}")
    print(f"Target directory: {test_dir}")
    
    # Parse the GitHub URL
    github_info = parse_github_url(test_url)
    print(f"Parsed GitHub URL: {github_info}")
    
    # Test the utils.github_utils.clone_github_repo function
    print("\nTesting utils.github_utils.clone_github_repo...")
    utils_test_dir = test_dir / "utils_test"
    if utils_test_dir.exists():
        shutil.rmtree(utils_test_dir)
    utils_test_dir.mkdir(exist_ok=True)
    
    docs_path, error = clone_github_repo(github_info, utils_test_dir)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Success! Docs path: {docs_path}")
        # List the contents of the docs directory
        print("\nContents of docs directory:")
        for item in docs_path.iterdir():
            print(f"  {item.name}")
    
    # Test the interfaces.portal.github.clone_github_repo function
    print("\nTesting interfaces.portal.github.clone_github_repo...")
    portal_test_dir = test_dir / "portal_test"
    if portal_test_dir.exists():
        shutil.rmtree(portal_test_dir)
    portal_test_dir.mkdir(exist_ok=True)
    
    try:
        portal_docs_path = await portal_clone_github_repo(test_url, portal_test_dir)
        print(f"Success! Portal docs path: {portal_docs_path}")
        # List the contents of the portal docs directory
        print("\nContents of portal docs directory:")
        for item in portal_docs_path.iterdir():
            print(f"  {item.name}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_sparse_checkout())
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/scripts/python_to_markdown.py

```python
#!/usr/bin/env python3
"""
Python to Markdown Converter

This script traverses directories to find Python files and merges them into a single Markdown file.
Each Python file is represented as a code block in the Markdown file with its path as a heading.

python scripts/python_to_markdown.py --root /Users/sergeyleksikov/Documents/GitHub/RAGModelService --output rag_context.md --relative --exclude "*venv*" "*__pycache__*" "*docs*"

"""

import os
import argparse
from pathlib import Path
import fnmatch


def collect_python_files(root_dir, exclude_patterns=None):
    """
    Recursively collect all Python files from the given directory.
    
    Args:
        root_dir (str): Root directory to start the search from
        exclude_patterns (list): List of glob patterns to exclude
        
    Returns:
        list: List of paths to Python files
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs_to_remove = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if any(fnmatch.fnmatch(dir_path, pattern) for pattern in exclude_patterns):
                dirs_to_remove.append(d)
        
        for d in dirs_to_remove:
            dirs.remove(d)
            
        # Collect Python files
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                    python_files.append(file_path)
    
    return python_files


def merge_to_markdown(python_files, output_file, relative_to=None):
    """
    Merge Python files into a single Markdown file.
    
    Args:
        python_files (list): List of paths to Python files
        output_file (str): Path to the output Markdown file
        relative_to (str): Directory to make paths relative to
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Python Code Collection\n\n")
        md_file.write("This document contains merged Python code files for LLM context.\n\n")
        
        for file_path in sorted(python_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as py_file:
                    content = py_file.read()
                
                # Make path relative if specified
                display_path = file_path
                if relative_to:
                    try:
                        display_path = os.path.relpath(file_path, relative_to)
                    except ValueError:
                        # Keep absolute path if relpath fails
                        pass
                
                md_file.write(f"## {display_path}\n\n")
                md_file.write("```python\n")
                md_file.write(content)
                if not content.endswith('\n'):
                    md_file.write('\n')
                md_file.write("```\n\n")
                
                print(f"Added {display_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert Python files to a single Markdown file')
    parser.add_argument('--root', '-r', type=str, required=True, 
                        help='Root directory to search for Python files')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Output Markdown file path')
    parser.add_argument('--exclude', '-e', type=str, nargs='+', default=[],
                        help='Glob patterns to exclude (e.g. "*venv*" "*/tests/*")')
    parser.add_argument('--relative', action='store_true',
                        help='Make file paths relative to the root directory')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root)
    output_file = args.output
    exclude_patterns = args.exclude
    
    print(f"Searching for Python files in: {root_dir}")
    print(f"Excluding patterns: {exclude_patterns}")
    
    python_files = collect_python_files(root_dir, exclude_patterns)
    print(f"Found {len(python_files)} Python files")
    
    relative_to = root_dir if args.relative else None
    merge_to_markdown(python_files, output_file, relative_to)
    
    print(f"Successfully merged Python files into: {output_file}")


if __name__ == "__main__":
    main()
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/setup.py

```python
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="rag_model_service",
    version="0.1.0",
    description="A RAG (Retrieval-Augmented Generation) service for document search and generation",
    author="Lablup",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements + [
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/utils/__init__.py

```python

```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/utils/github_utils.py

```python
#!/usr/bin/env python3
"""
GitHub Utilities for RAG Model Service

This module provides functions for:
1. Parsing GitHub URLs into components
2. Cloning GitHub repositories
3. Extracting documentation from repositories
"""

import re
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import structlog
from git import Repo
from pydantic import BaseModel, Field


# Initialize logger
logger = structlog.get_logger()


class GitHubInfo(BaseModel):
    """Model for GitHub repository information"""
    url: str
    owner: str
    repo: str
    branch: str = "main"
    path: str = ""


def parse_github_url(github_url: str) -> GitHubInfo:
    """
    Parse a GitHub URL into its components.
    
    Args:
        github_url: GitHub URL (https://github.com/owner/repo[/tree/branch][/path/to/docs])
        
    Returns:
        GitHubInfo object with parsed components
        
    Raises:
        ValueError: If the URL is invalid
    """
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Basic URL pattern for GitHub
    pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    match = re.match(pattern, github_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    
    owner = match.group(1)
    repo = match.group(2)
    branch = match.group(3) or "main"  # Default to 'main' if branch is not specified
    path = match.group(4) or ""  # Default to empty string if path is not specified
    
    return GitHubInfo(
        url=github_url,
        owner=owner,
        repo=repo,
        branch=branch,
        path=path
    )


def validate_github_url(url: str) -> bool:
    """
    Validate a GitHub URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Basic GitHub URL pattern
    pattern = r"^https?://github\.com/[^/]+/[^/]+(?:/tree/[^/]+(?:/.*)?)?$"
    return bool(re.match(pattern, url))


def clone_github_repo(
    github_info: GitHubInfo, 
    target_dir: Union[str, Path]
) -> Tuple[Path, Optional[Exception]]:
    """
    Clone a GitHub repository to a local directory.
    If a specific path is provided in the GitHub URL, only that directory is cloned.
    
    Args:
        github_info: GitHubInfo object with repository information
        target_dir: Directory to clone the repository to
        
    Returns:
        Tuple of (repository path, exception if any)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Construct repo URL
        repo_url = f"https://github.com/{github_info.owner}/{github_info.repo}.git"
        
        # If a specific path is provided, use sparse checkout to only get that directory
        if github_info.path:
            logger.info(
                "Sparse cloning repository directory", 
                repo=f"{github_info.owner}/{github_info.repo}",
                branch=github_info.branch,
                path=github_info.path
            )
            
            # Create a temporary working directory
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            
            try:
                # Clone only the specific directory using GitHub's SVN interface
                # This is more reliable than git sparse-checkout for single directory cloning
                svn_url = f"https://github.com/{github_info.owner}/{github_info.repo}/trunk/{github_info.path}"
                
                # Check if svn is available
                try:
                    import subprocess
                    subprocess.run(["svn", "--version"], check=True, capture_output=True)
                    
                    # Use SVN to checkout just the docs directory
                    subprocess.run(
                        ["svn", "export", svn_url, str(target_dir / github_info.path)],
                        check=True
                    )
                    
                    # Determine documentation path
                    docs_path = target_dir / github_info.path
                    if not docs_path.exists():
                        raise ValueError(f"Documentation path not found: {docs_path}")
                    
                    return docs_path, None
                    
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("SVN not available, falling back to full repository clone")
                    # Fall back to cloning the entire repository
                    pass
            finally:
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Fall back to cloning the entire repository if SVN approach fails
            logger.info(
                "Falling back to full repository clone", 
                repo=f"{github_info.owner}/{github_info.repo}",
                branch=github_info.branch
            )
            
        # Clone the entire repository
        logger.info(
            "Cloning repository", 
            repo=f"{github_info.owner}/{github_info.repo}",
            branch=github_info.branch
        )
        
        # If a specific path is provided, use a shallow clone to save time and space
        if github_info.path:
            git_repo = Repo.clone_from(
                repo_url, 
                target_dir, 
                branch=github_info.branch,
                depth=1,  # Shallow clone - only get the latest commit
                multi_options=['--single-branch']  # Only clone the specified branch
            )
        else:
            # Regular clone for the entire repository
            git_repo = Repo.clone_from(repo_url, target_dir, branch=github_info.branch)
        
        # Determine documentation path
        docs_path = target_dir
        if github_info.path:
            docs_path = target_dir / github_info.path
            if not docs_path.exists():
                raise ValueError(f"Documentation path not found: {docs_path}")
                
        return docs_path, None
        
    except Exception as e:
        logger.error(
            "Error cloning repository", 
            repo=f"{github_info.owner}/{github_info.repo}",
            error=str(e)
        )
        return target_dir, e
```

## /Users/sergeyleksikov/Documents/GitHub/RAGModelService/utils/service_utils.py

```python
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
    model_config = {"protected_namespaces": ()}
    
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


def save_service_info(service_id: str, service_info: Dict[str, Any]) -> bool:
    """
    Save service information to a file.
    
    Args:
        service_id: Unique service identifier
        service_info: Dictionary containing service information
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import json
        
        # Create services directory if it doesn't exist
        services_dir = Path("rag_services")
        services_dir.mkdir(exist_ok=True)
        
        # Create service directory if it doesn't exist
        service_dir = services_dir / service_id
        service_dir.mkdir(exist_ok=True)
        
        # Create info file path
        info_file = service_dir / "service_info.json"
        
        # Convert Path objects to strings for JSON serialization
        serializable_info = {}
        for key, value in service_info.items():
            if isinstance(value, Path):
                serializable_info[key] = str(value)
            else:
                serializable_info[key] = value
        
        # Write info to file
        with open(info_file, "w") as f:
            json.dump(serializable_info, f, indent=2)
            
        logger.info("Saved service info", service_id=service_id, info_file=str(info_file))
        return True
        
    except Exception as e:
        logger.error("Error saving service info", service_id=service_id, error=str(e))
        return False


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
```

