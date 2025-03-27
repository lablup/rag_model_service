# core/retrieval.py
from typing import List, Dict, Tuple, AsyncGenerator, Optional
from pydantic import BaseModel
import structlog

from rag_model_service.data.vector_store import VectorStore
from rag_model_service.data.filtering import DocumentFilter
from rag_model_service.config.models import RetrievalSettings

class RetrievalEngine:
    """Handles document retrieval and context preparation."""
    
    def __init__(
        self,
        settings: RetrievalSettings,
        vector_store: VectorStore,
        document_filter: Optional[DocumentFilter] = None
    ):
        self.logger = structlog.get_logger().bind(component="RetrievalEngine")
        self.settings = settings
        self.vector_store = vector_store
        self.document_filter = document_filter or DocumentFilter()
        
    async def get_relevant_context(self, query: str) -> Tuple[str, List[Dict]]:
        """Retrieve and format relevant context for a query."""
        # Implementation
        ...

# core/llm.py
from typing import AsyncGenerator, List, Optional
import structlog

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag_model_service.config.models import LLMSettings

class LLMInterface:
    """Interface to large language model."""
    
    def __init__(self, settings: LLMSettings):
        self.logger = structlog.get_logger().bind(component="LLMInterface")
        self.settings = settings
        self.llm = ChatOpenAI(
            openai_api_key=settings.api_key,
            model_name=settings.model_name,
            temperature=settings.temperature,
            streaming=settings.streaming,
            max_tokens=settings.max_tokens
        )
        self.messages = []
        
    async def generate_response(
        self, 
        user_input: str, 
        context: str
    ) -> AsyncGenerator[str, None]:
        """Generate a response using the LLM with given context."""
        # Implementation
        ...



# data/vector_store.py
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog
import aiofiles

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag_model_service.config.models import VectorStoreSettings

class VectorStore:
    """Vector database for document storage and retrieval."""
    
    def __init__(self, settings: VectorStoreSettings):
        self.logger = structlog.get_logger().bind(component="VectorStore")
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.index: Optional[FAISS] = None
        
    async def collect_documents(self) -> List[Document]:
        """Collect documents from the docs directory."""
        # Implementation
        ...
        
    async def create_index(self, documents: List[Document]) -> None:
        """Create vector index from documents."""
        # Implementation
        ...
        
    async def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        # Implementation
        ...



# interfaces/gradio_app/app.py
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import asyncio
import gradio as gr

from rag_model_service.core.retrieval import RetrievalEngine
from rag_model_service.core.llm import LLMInterface
from rag_model_service.data.vector_store import VectorStore
from rag_model_service.config.loader import load_config
from rag_model_service.interfaces.gradio_app.handlers import process_message, handle_doc_selection
from rag_model_service.interfaces.gradio_app.components import create_chat_interface

async def create_gradio_app(
    retrieval_engine: RetrievalEngine,
    llm_interface: LLMInterface,
    title: str = "Documentation Assistant",
    description: str = "Search documentation with natural language queries"
) -> gr.Blocks:
    """Create Gradio interface for RAG chat application."""
    # Implementation
    ...


# interfaces/fastapi_app/server.py
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from rag_model_service.interfaces.fastapi_app.models import ChatCompletionRequest, ChatCompletionResponse
from rag_model_service.core.retrieval import RetrievalEngine
from rag_model_service.core.llm import LLMInterface

def create_fastapi_app(
    retrieval_engine: RetrievalEngine,
    llm_interface: LLMInterface
) -> FastAPI:
    """Create FastAPI application with OpenAI-compatible endpoints."""
    app = FastAPI(title="RAG OpenAI Compatible API")
    
    # Add CORS middleware and routes
    # Implementation
    ...
    
    return app


# config/models.py
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

class LLMSettings(BaseModel):
    """LLM configuration settings."""
    api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field("gpt-4o", description="Model name to use")
    temperature: float = Field(0.2, description="Temperature parameter")
    max_tokens: int = Field(2048, description="Maximum tokens for generation")
    streaming: bool = Field(True, description="Whether to stream responses")

class VectorStoreSettings(BaseModel):
    """Vector store configuration."""
    docs_root: Path
    indices_path: Path
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model")

class RetrievalSettings(BaseModel):
    """Retrieval engine settings."""
    max_results: int = Field(5, description="Maximum results to retrieve")
    max_tokens_per_doc: int = Field(8000, description="Maximum tokens per document")

class ServerSettings(BaseModel):
    """Server configuration."""
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, description="Server port")
    share_enabled: bool = Field(False, description="Enable public sharing")

class ApplicationSettings(BaseModel):
    """Main application settings."""
    llm: LLMSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    server: ServerSettings

    ...






def create_service_directory(service_id: str) -> Path:
    """Create a directory for the service."""
    service_dir = Path("./services") / service_id
    service_dir.mkdir(parents=True, exist_ok=True)
    return service_dir
