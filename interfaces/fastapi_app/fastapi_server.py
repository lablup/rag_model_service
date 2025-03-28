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

app = FastAPI(title="RAG OpenAI Compatible API")

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
retrieval_settings = config.retrieval

# Get service ID from environment variable or use default
service_id = os.getenv("SERVICE_ID", "fastapi")

# Initialize with None, will be set during startup
vector_store = None
rag_engine = None

# API key validator (using environment variable or config)
api_key_validator = APIKeyValidator(os.getenv("OPENAI_API_KEY", llm_config.openai_api_key))


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_store, rag_engine

    # Get paths from configuration
    docs_root = path_config.get_service_docs_path(service_id)
    indices_root = path_config.get_service_indices_path(service_id)

    print(f"Starting up FastAPI server with configuration:")
    print(f"Service ID: {service_id}")
    print(f"Docs path: {docs_root}")
    print(f"Indices path: {indices_root}")
    print(f"LLM model: {llm_config.model_name}")
    print(f"Temperature: {llm_config.temperature}")
    print(f"Max results: {retrieval_settings.top_k}")

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


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--service-id", type=str, default="fastapi", help="Service ID for path resolution")
    
    args = parser.parse_args()
    
    # Update service ID from command line
    if args.service_id:
        service_id = args.service_id
        print(f"Using service ID from command line: {service_id}")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)
