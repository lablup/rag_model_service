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

from app.rag_chatbot import LLMConfig, RAGManager
from vectordb_manager.vectordb_manager import VectorDBManager


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
docs_root = Path("/models/RAGModelService/TensorRT-LLM/docs/source")  # Directory for documentation files
indices_path = Path("./embedding_indices")  # Directory for vector store indices
vector_db = None
rag_manager = None

# API key validator (using environment variable)
api_key_validator = APIKeyValidator(os.getenv("OPENAI_API_KEY", ""))


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_db, rag_manager

    # Create necessary directories if they don't exist
    docs_root.mkdir(exist_ok=True)
    indices_path.mkdir(exist_ok=True)


    # Initialize vector database with existing indices path
    vector_db = VectorDBManager(docs_root=docs_root, indices_path=indices_path)

    # Only load existing indices, don't recreate them
    await vector_db.load_index()

    # Initialize RAG manager
    config = LLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name="gpt-4.1-mini",  # Match the model from RAGManager
        temperature=0.2,  # Match the temperature from RAGManager
        streaming=True,
    )
    rag_manager = RAGManager(config=config, vector_store=vector_db)

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
        async for chunk in rag_manager.generate_response(
            user_input=last_message.content
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
            async for chunk in rag_manager.generate_response(
                user_input=last_message.content
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
