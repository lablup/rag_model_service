# Codebase Collection

This document contains merged code files for LLM context.

## ai_guides/backendai_commands.sh

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

## app/__init__.py

```python

```

## app/document_filter.py

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

## app/gradio_app.py

```python
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in path
project_root = Path(__file__).parent.parent  # Updated to point to the correct project root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import logging

import aiofiles
import gradio as gr
from pydantic import BaseModel, Field
from rich.console import Console

from config.config import load_config
from app.rag_chatbot import LLMConfig, RAGManager
from vectordb_manager.vectordb_manager import VectorDBManager

# Initialize logger and console
logger = logging.getLogger(__name__)
console = Console()

# Load configuration
config = load_config()

DOC_NAME = os.getenv("RAG_SERVICE_NAME", "gpt-4o")

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

        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        logger.info(f"File exists, reading content...")
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Fix image paths
        # Replace relative image paths with base64 encoded images
        file_dir = file_path.parent
        import base64
        import mimetypes
        import re

        def replace_image_path(match):
            img_tag = match.group(0)
            if 'src="' in img_tag:
                # Handle HTML img tags
                src_pattern = r'src="([^"]+)"'
                src_match = re.search(src_pattern, img_tag)
                if src_match:
                    img_path = src_match.group(1)
                    if not img_path.startswith(("http://", "https://", "data:")):
                        # Convert the path to be relative to assets/images
                        rel_path = Path(img_path).name
                        # Use the docs_root from config instead of hardcoded path
                        try:
                            doc_rel_path = file_path.relative_to(config.paths.docs_root)
                            img_file = (
                                project_root
                                / "rag_service"
                                / "docs"
                                / "assets"
                                / "images"
                                / doc_rel_path.parent
                                / rel_path
                            )
                            if img_file.exists():
                                mime_type = mimetypes.guess_type(img_file)[0]
                                with open(img_file, "rb") as f:
                                    img_data = base64.b64encode(f.read()).decode()
                                return img_tag.replace(
                                    f'src="{img_path}"',
                                    f'src="data:{mime_type};base64,{img_data}"',
                                )
                        except ValueError:
                            # If the file is not in the docs_root, try a direct approach
                            logger.warning(f"File {file_path} is not in the docs_root, trying direct path")
                            img_file = file_path.parent / img_path
                            if img_file.exists():
                                mime_type = mimetypes.guess_type(img_file)[0]
                                with open(img_file, "rb") as f:
                                    img_data = base64.b64encode(f.read()).decode()
                                return img_tag.replace(
                                    f'src="{img_path}"',
                                    f'src="data:{mime_type};base64,{img_data}"',
                                )
            elif "![" in img_tag:
                # Handle Markdown image syntax
                md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
                md_match = re.match(md_pattern, img_tag)
                if md_match:
                    alt_text = md_match.group(1)
                    img_path = md_match.group(2)
                    if not img_path.startswith(("http://", "https://", "data:")):
                        # Convert the path to be relative to assets/images
                        rel_path = Path(img_path).name
                        try:
                            doc_rel_path = file_path.relative_to(config.paths.docs_root)
                            img_file = (
                                project_root
                                / "rag_service"
                                / "docs"
                                / "assets"
                                / "images"
                                / doc_rel_path.parent
                                / rel_path
                            )
                            if img_file.exists():
                                mime_type = mimetypes.guess_type(img_file)[0]
                                with open(img_file, "rb") as f:
                                    img_data = base64.b64encode(f.read()).decode()
                                return f"![{alt_text}](data:{mime_type};base64,{img_data})"
                        except ValueError:
                            # If the file is not in the docs_root, try a direct approach
                            logger.warning(f"File {file_path} is not in the docs_root, trying direct path")
                            img_file = file_path.parent / img_path
                            if img_file.exists():
                                mime_type = mimetypes.guess_type(img_file)[0]
                                with open(img_file, "rb") as f:
                                    img_data = base64.b64encode(f.read()).decode()
                                return f"![{alt_text}](data:{mime_type};base64,{img_data})"
            return img_tag

        # Match both HTML img tags and Markdown image syntax
        image_pattern = r"<img[^>]+>|!\[[^\]]*\]\([^)]+\)"
        content = re.sub(image_pattern, replace_image_path, content)

        logger.info(f"Successfully read {len(content)} characters")
        logger.info(f"Content preview: {content[:200]}")
        return content
    except Exception as e:
        error_msg = f"Error reading markdown file: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


def create_gradio_interface(
    rag_manager: RAGManager,
    docs_path: Optional[Path] = None,
) -> gr.Blocks:
    """Create Gradio interface for the RAG chat application
    
    Args:
        rag_manager: RAG manager instance
        docs_path: Optional custom path to documentation files. If None, uses default from config.
    """
    
    # Use custom docs_path if provided, otherwise use default from config
    actual_docs_path = docs_path if docs_path is not None else config.paths.docs_root
    logger.info(f"Using documentation path: {actual_docs_path}")
    
    # Create a custom read_markdown_file function that uses the actual_docs_path
    async def custom_read_markdown_file(file_path: str | Path) -> str:
        """Read markdown file with custom docs path"""
        try:
            file_path = Path(file_path)
            logger.info(f"Attempting to read file: {file_path}")
            logger.info(f"File exists: {file_path.exists()}")
            logger.info(f"File is file: {file_path.is_file()}")
            logger.info(f"File absolute path: {file_path.absolute()}")

            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

            logger.info(f"File exists, reading content...")
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            # Fix image paths
            # Replace relative image paths with base64 encoded images
            file_dir = file_path.parent
            import base64
            import mimetypes
            import re

            def replace_image_path(match):
                img_tag = match.group(0)
                if 'src="' in img_tag:
                    # Handle HTML img tags
                    src_pattern = r'src="([^"]+)"'
                    src_match = re.search(src_pattern, img_tag)
                    if src_match:
                        img_path = src_match.group(1)
                        if not img_path.startswith(("http://", "https://", "data:")):
                            # Convert the path to be relative to assets/images
                            rel_path = Path(img_path).name
                            # Use the actual_docs_path instead of config.paths.docs_root
                            try:
                                doc_rel_path = file_path.relative_to(actual_docs_path)
                                img_file = (
                                    project_root
                                    / "rag_service"
                                    / "docs"
                                    / "assets"
                                    / "images"
                                    / doc_rel_path.parent
                                    / rel_path
                                )
                                if img_file.exists():
                                    mime_type = mimetypes.guess_type(img_file)[0]
                                    with open(img_file, "rb") as f:
                                        img_data = base64.b64encode(f.read()).decode()
                                    return img_tag.replace(
                                        f'src="{img_path}"',
                                        f'src="data:{mime_type};base64,{img_data}"',
                                    )
                            except ValueError:
                                # If the file is not in the docs_root, try a direct approach
                                logger.warning(f"File {file_path} is not in the actual_docs_path, trying direct path")
                                img_file = file_path.parent / img_path
                                if img_file.exists():
                                    mime_type = mimetypes.guess_type(img_file)[0]
                                    with open(img_file, "rb") as f:
                                        img_data = base64.b64encode(f.read()).decode()
                                    return img_tag.replace(
                                        f'src="{img_path}"',
                                        f'src="data:{mime_type};base64,{img_data}"',
                                    )
                elif "![" in img_tag:
                    # Handle Markdown image syntax
                    md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
                    md_match = re.match(md_pattern, img_tag)
                    if md_match:
                        alt_text = md_match.group(1)
                        img_path = md_match.group(2)
                        if not img_path.startswith(("http://", "https://", "data:")):
                            # Convert the path to be relative to assets/images
                            rel_path = Path(img_path).name
                            try:
                                doc_rel_path = file_path.relative_to(actual_docs_path)
                                img_file = (
                                    project_root
                                    / "rag_service"
                                    / "docs"
                                    / "assets"
                                    / "images"
                                    / doc_rel_path.parent
                                    / rel_path
                                )
                                if img_file.exists():
                                    mime_type = mimetypes.guess_type(img_file)[0]
                                    with open(img_file, "rb") as f:
                                        img_data = base64.b64encode(f.read()).decode()
                                    return f"![{alt_text}](data:{mime_type};base64,{img_data})"
                            except ValueError:
                                # If the file is not in the docs_root, try a direct approach
                                logger.warning(f"File {file_path} is not in the actual_docs_path, trying direct path")
                                img_file = file_path.parent / img_path
                                if img_file.exists():
                                    mime_type = mimetypes.guess_type(img_file)[0]
                                    with open(img_file, "rb") as f:
                                        img_data = base64.b64encode(f.read()).decode()
                                    return f"![{alt_text}](data:{mime_type};base64,{img_data})"
                return img_tag

            # Match both HTML img tags and Markdown image syntax
            image_pattern = r"<img[^>]+>|!\[[^\]]*\]\([^)]+\)"
            content = re.sub(image_pattern, replace_image_path, content)

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
    ) -> Tuple[List[Dict[str, str]], str, str, List[List[str]], str, ChatState]:
        """Process user message and update interface components"""
        if not message.strip():
            if not isinstance(state, ChatState):
                state = ChatState()
            return (
                [{"role": msg.role, "content": msg.content} for msg in state.messages],
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
                context, retrieved_results = await rag_manager._get_relevant_context(message)
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                return (
                    "I encountered an error retrieving the context. Please try again."
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
                full_path = actual_docs_path / relative_path

                # Verify file exists before adding
                if not full_path.exists():
                    logger.warning(f"File not found: {full_path}")
                    continue

                state.current_docs[doc_name] = str(full_path)
                total_docs += 1

            logger.info(f"Total documents in state after processing: {total_docs}")
            logger.info(f"Documents in state: {list(state.current_docs.keys())}")

            # Generate response
            full_response = ""
            async for chunk in rag_manager.generate_response_with_context(
                message, context
            ):
                full_response += chunk

            # Add assistant response to state
            state.messages.append(ChatMessage(role="assistant", content=full_response))

            # Convert messages to the new Gradio chatbot format with roles
            chat_messages = [
                {"role": msg.role, "content": msg.content} for msg in state.messages
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
            return [], "", f"Error: {str(e)}", [], "", state

    # Create Gradio interface
    with gr.Blocks(
        title=f"{DOC_NAME} Documentation Assistant",
        theme=gr.themes.Base(),
        css="""

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
                    type="messages",
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


async def main(docs_path: Optional[Path] = None):
    """Main entry point
    
    Args:
        docs_path: Optional custom path to documentation files. If None, uses default from config.
    """
    try:
        # Use custom docs_path if provided, otherwise use default from config
        actual_docs_path = docs_path if docs_path is not None else config.paths.docs_root
        logger.info(f"Using documentation path: {actual_docs_path}")
        
        # Initialize vector store
        vector_store = VectorDBManager(
            docs_root=actual_docs_path, indices_path=config.paths.indices_path
        )

        # Load vector indices
        await vector_store.load_index()

        # Initialize RAG manager
        llm_config = LLMConfig(
            openai_api_key=config.openai.api_key,
            model_name=config.openai.model_name,
            temperature=config.openai.temperature,
            max_results=config.rag.max_results,
            streaming=True,
        )

        rag_manager = RAGManager(llm_config, vector_store)

        # Create and launch Gradio interface
        interface = create_gradio_interface(rag_manager, docs_path=actual_docs_path)
        interface.launch(
            server_name=config.server.host,
            server_port=config.server.port,
            share=config.server.share_enabled,
            debug=True,
        )
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        console.print(f"[red]Error initializing application: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

## app/rag_chatbot.py

```python
import argparse
import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

import structlog
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from vectordb_manager.vectordb_manager import VectorDBManager

logger = structlog.get_logger()


class ChatResponse(BaseModel):
    """Model for chat response"""

    role: str
    content: str


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


class RAGManager:
    """Manager for RAG-enhanced chatbot using LangChain"""

    def __init__(self, config: LLMConfig, vector_store: VectorDBManager):
        """Initialize RAG Manager"""
        self.logger = logger.bind(component="RAGManager")
        self.config = config
        self.vector_store = vector_store

        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            streaming=config.streaming,
            base_url=config.base_url if config.base_url else None,
            max_tokens=4096,
            timeout=120,  # Increased timeout to 5 minutes
            max_retries=3,  # Increased retries
        )

        # Initialize memory as a list of messages
        self.messages = []
        self.memory_k = config.memory_k

        system_prompt = """
        You are a helpful AI Assistant with document search and retrieval capabilities. Answer questions based on the provided context.
        Provide the detailed explanation.
        The provided context is a list of documents from a vector store knowledge base.
        The similarity score for each document is also provided as Euclidean distance where the lower the number the more similar.
        If the context doesn't contain relevant information, use your general knowledge but mention this fact. Keep answers focused and relevant to the query.
        If there is no context provided and you don't know, then answer "I don't know".
        """
        self.system_prompt = SystemMessage(content=system_prompt)

        # Create RAG-specific prompt template
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

    async def _get_relevant_context(self, query: str) -> tuple[str, List[Dict]]:
        """
        Retrieve and filter relevant context from vector store

        Args:
            query: User's question

        Returns:
            Tuple of (formatted context string, filtered results)
        """
        try:
            results = await self.vector_store.search_documents(
                query=query, k=self.config.max_results
            )

            filtered_results = results

            context_parts = []
            
            for doc in filtered_results:
                # Get metadata fields safely with defaults
                metadata = doc.get("metadata", {})
                relative_path = metadata.get("relative_path", "unknown_path")
                similarity_score = doc.get("similarity_score", 0.0)
                content = doc.get("content", "")

                # Truncate content to limit tokens
                if (
                    len(content) > self.config.max_tokens_per_doc * 4
                ):  # Rough estimate of 4 chars per token
                    content = (
                        content[: self.config.max_tokens_per_doc * 4] + "..."
                    )

                context_parts.append(
                    f"[Source: {relative_path} "
                    f"(Similarity: {similarity_score:.2f})]\n"
                    f"{content}\n"
                )

            if not context_parts:
                self.logger.warning("No documents passed filtering", query=query)
                return "No relevant context found after filtering.", []

            return "\n".join(context_parts), filtered_results

        except Exception as e:
            self.logger.error(
                "Error retrieving/filtering context", error=str(e), query=query
            )
            return "Error retrieving context.", []

    async def generate_response(
        self, user_input: str
    ) -> AsyncGenerator[str, None]:
        """Generate RAG-enhanced streaming response"""
        try:
            # Retrieve and filter relevant context
            context, filtered_results = await self._get_relevant_context(user_input)
            context_msg = HumanMessage(content=f"<context>\n{context}\n</context>\n")

            # Get chat history
            history = self.get_chat_history()
            
            async for chunk in self.chain.astream(
                {
                    "input": user_input,
                    "context": [context_msg],
                    "chat_history": history,
                }
            ):
                yield chunk

            # Update memory
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=chunk))

        except Exception as e:
            self.logger.error(
                "Error generating response",
                error=str(e),
                user_input=user_input,
            )
            yield f"Error: {str(e)}"

    async def generate_response_with_context(
        self, user_input: str, context: str
    ) -> AsyncGenerator[str, None]:
        """Generate response using provided context"""
        try:
            context_msg = HumanMessage(content=f"<context>\n{context}\n</context>\n")
            history = self.get_chat_history()

            async for chunk in self.chain.astream(
                {
                    "input": user_input,
                    "context": [context_msg],
                    "chat_history": history,
                }
            ):
                yield chunk

            # Update memory
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=chunk))

        except Exception as e:
            self.logger.error(
                "Error generating response with context",
                error=str(e),
                user_input=user_input,
            )
            yield f"Error: {str(e)}"

    def get_chat_history(self) -> List[Union[HumanMessage, AIMessage]]:
        """Get the chat history for the context window"""
        # Limit to the last k exchanges
        if len(self.messages) > self.memory_k * 2:
            return self.messages[-self.memory_k * 2 :]
        return self.messages


async def interactive_mode(rag_manager: RAGManager, verbose: bool = False) -> None:
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
            async for chunk in rag_manager.generate_response(user_input):
                # Print chunk without newline to simulate streaming
                print(chunk, end="", flush=True)
                response_text += chunk
            print()  # Add a newline at the end
            
            if verbose:
                print("\n--- Debug Info ---")
                print(f"Model: {rag_manager.config.model_name}")
                print(f"Temperature: {rag_manager.config.temperature}")
                print(f"Max Results: {rag_manager.config.max_results}")
                print(f"Messages in History: {len(rag_manager.messages)}")
                print("------------------")
        except Exception as e:
            print(f"\nError: {str(e)}")


async def process_single_query(rag_manager: RAGManager, query: str, show_context: bool = False) -> None:
    """Process a single query and exit."""
    print(f"Query: {query}")
    
    if show_context:
        context, _ = await rag_manager._get_relevant_context(query)
        print("\n----- Retrieved Context -----")
        print(context)
        print("----------------------------\n")
    
    print("\nResponse:")
    
    # Collect response chunks
    response_text = ""
    try:
        async for chunk in rag_manager.generate_response(query):
            # Print chunk without newline to simulate streaming
            print(chunk, end="", flush=True)
            response_text += chunk
        print()  # Add a newline at the end
    except Exception as e:
        print(f"\nError: {str(e)}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Query documents using a RAG-enhanced LLM."
    )
    
    # Path arguments
    parser.add_argument(
        "--docs-path", 
        type=str,
        help="Path to documentation directory"
    )
    parser.add_argument(
        "--indices-path", 
        type=str,
        help="Path to vector indices"
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--query", 
        type=str,
        metavar="QUERY",
        help="Single query mode: Process one query and exit"
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Start in interactive chat mode (default if no mode is specified)"
    )
    
    # LLM configuration
    parser.add_argument(
        "--model", 
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the LLM (default: 0.2)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for the LLM response (default: 2048)"
    )
    
    # Additional options
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context before response (only in query mode)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL for custom model endpoints"
    )
    
    return parser.parse_args()


async def main() -> int:
    """Main function to run the RAG chatbot."""
    # Load environment variables from .env file
    load_dotenv()

    # Check for required environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or in your environment.")
        return 1
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    docs_root = Path(args.docs_path) if args.docs_path else base_dir / "docs"
    indices_path = Path(args.indices_path) if args.indices_path else base_dir / "embedding_indices"

    if args.verbose:
        print(f"Initializing with docs path: {docs_root}")
        print(f"Vector indices path: {indices_path}")

    # Initialize VectorDBManager
    vector_manager = VectorDBManager(docs_root, indices_path)
    
    # Load the vector index
    if args.verbose:
        print("Loading vector index...")
    
    try:
        await vector_manager.load_index()
        if not vector_manager.index:
            print("Error: Failed to load index. Please check that the index exists.")
            return 1
    except Exception as e:
        print(f"Error loading index: {e}")
        return 1
    
    # Set up LLM config from arguments and environment variables
    llm_config = LLMConfig(
        openai_api_key=openai_api_key,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_results=args.max_results,
        streaming=True,
        base_url=args.base_url
    )
    
    # Initialize RAG manager
    rag_manager = RAGManager(llm_config, vector_manager)
    
    # Execute the requested mode
    try:
        if args.query:
            await process_single_query(rag_manager, args.query, args.show_context)
        else:
            # Default to interactive mode
            await interactive_mode(rag_manager, args.verbose)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)
```

## auto_rag_service/create_rag_service.py

```python
#!/usr/bin/env python3
"""
End-to-End RAG Service Creator

This script:
1. Clones a GitHub repository containing documentation
2. Processes the documentation to create vector embeddings
3. Launches a Gradio web interface for the RAG system

Usage:
    python create_rag_service.py --github-url https://github.com/owner/repo
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import structlog
from dotenv import load_dotenv
from git import Repo
import gradio as gr
import re

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from vectordb_manager.vectordb_manager import VectorDBManager
from app.rag_chatbot import LLMConfig, RAGManager
from app.gradio_app import create_gradio_interface

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-End RAG Service Creator"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Docs path (optional, to use existing documentation)
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to existing documentation directory (skips cloning if provided)",
        default=None,
    )
    
    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the RAG service",
        default="./rag_service",
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
        help="OpenAI model to use for RAG",
        default="gpt-4o",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM responses",
        default=0.2,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results to retrieve",
        default=5,
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


def parse_github_url(github_url: str) -> Tuple[str, str, str, str]:
    """
    Parse GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple containing owner, repo, branch, and path
    """
    # Handle different GitHub URL formats
    # https://github.com/owner/repo
    # https://github.com/owner/repo/tree/branch
    # https://github.com/owner/repo/tree/branch/path/to/docs
    
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Basic URL pattern for GitHub
    basic_pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    match = re.match(basic_pattern, github_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    
    owner = match.group(1)
    repo = match.group(2)
    branch = match.group(3) or "main"  # Default to 'main' if branch is not specified
    path = match.group(4) or ""  # Default to empty string if path is not specified
    
    return owner, repo, branch, path


def generate_service_config(args) -> Dict:
    """
    Generate service configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Extract GitHub information
    owner, repo, branch, path = parse_github_url(args.github_url)
    
    # Determine output directory
    output_dir = Path(args.output_dir).resolve()
    
    # Create paths
    docs_dir = output_dir / f"{owner}_{repo}"
    indices_dir = output_dir / f"{owner}_{repo}_indices"
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Create configuration
    config = {
        "github": {
            "url": args.github_url,
            "owner": owner,
            "repo": repo,
            "branch": branch or "main",  # Default to main if branch is not specified
            "path": path,
        },
        "paths": {
            "output_dir": output_dir,
            "docs_dir": docs_dir,
            "indices_dir": indices_dir,
            "existing_docs_path": args.docs_path,  # Add the existing docs path
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "share": args.share,
            "debug": args.debug,
        },
        "llm": {
            "model_name": args.openai_model,
            "temperature": args.temperature,
            "max_results": args.max_results,
            "streaming": True,
            "openai_api_key": openai_api_key,
        },
        "ui": {
            "title": args.title,
            "description": args.description,
            "suggested_questions": args.suggested_questions,
        },
    }
    
    # Generate a default title and description based on repository if not provided
    repo_name = f"{owner}/{repo}"
    default_title = f"{repo_name} Documentation Assistant"
    default_description = f"Search and ask questions about {repo_name} documentation"
    
    if not config["ui"]["title"]:
        config["ui"]["title"] = default_title
    if not config["ui"]["description"]:
        config["ui"]["description"] = default_description
    
    return config


async def create_rag_service(config: Dict) -> Tuple[Path, Path]:
    """
    Create RAG service by cloning repository and processing documentation.
    
    Args:
        config: Service configuration
        
    Returns:
        Tuple of documentation path and vector indices path
    """
    try:
        # Create output directory
        output_dir = config["paths"]["output_dir"]
        docs_dir = config["paths"]["docs_dir"]
        indices_dir = config["paths"]["indices_dir"]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone repository
        print(f"Cloning {config['github']['owner']}/{config['github']['repo']} ({config['github']['branch']})...")
        
        repo_url = f"https://github.com/{config['github']['owner']}/{config['github']['repo']}.git"
        
        # Use GitPython to clone the repository
        git_repo = Repo.clone_from(repo_url, docs_dir, branch=config['github']['branch'])
        
        # Determine documentation path
        if config['github']['path']:
            docs_path = docs_dir / config['github']['path']
            if not docs_path.exists():
                raise ValueError(f"Documentation path not found: {docs_path}")
        else:
            # Default to the root of the repository
            docs_path = docs_dir
        
        print(f"Documentation path: {docs_path}")
        
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_dir)
        
        # Collect documents
        print("Collecting documents...")
        documents = await vector_manager.collect_documents()
        
        if not documents:
            logger.warning("No documents found", docs_path=str(docs_path))
            print(f"Warning: No documents found in {docs_path}")
            print("Please check that the repository contains markdown (.md) files.")
            return docs_path, indices_dir
        
        print(f"Found {len(documents)} documents")
        
        # Create vector indices
        print("Creating vector indices...")
        await vector_manager.create_indices(documents)
        
        print(f"Vector indices created successfully in {indices_dir}")
        
        return docs_path, indices_dir
        
    except Exception as e:
        logger.error("Error creating RAG service", error=str(e))
        print(f"Error creating RAG service: {str(e)}")
        raise


async def launch_service(docs_path: Path, indices_path: Path, config: Dict) -> None:
    """
    Launch RAG service with Gradio interface.
    
    Args:
        docs_path: Path to documentation
        indices_path: Path to vector indices
        config: Service configuration
    """
    try:
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_path)
        
        # Load vector indices
        await vector_manager.load_index()
        
        if not vector_manager.index:
            logger.error("Failed to load vector index", indices_path=str(indices_path))
            print(f"Error: Failed to load vector index from {indices_path}")
            return
        
        # Initialize LLMConfig
        llm_config = LLMConfig(
            openai_api_key=config["llm"]["openai_api_key"],
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_results=config["llm"]["max_results"],
            streaming=config["llm"]["streaming"],
        )
        
        # Initialize RAGManager
        rag_manager = RAGManager(llm_config, vector_manager)
        
        # Create Gradio interface
        interface = create_gradio_interface(rag_manager)
        
        # Customize interface
        interface.title = config["ui"]["title"]
        
        # Launch the interface
        print(f"\nLaunching Gradio server with {config['llm']['model_name']} for RAG...")
        print(f"Documentation path: {docs_path}")
        print(f"Vector indices path: {indices_path}")
        print(f"Server: {config['server']['host']}:{config['server']['port']}")
        
        if config['server']['share']:
            print("Creating shareable link...")
        
        # Launch the server
        interface.launch(
            server_name=config["server"]["host"],
            server_port=config["server"]["port"],
            share=config["server"]["share"],
            debug=config["server"]["debug"],
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
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Generate service configuration
        config = generate_service_config(args)
        
        # Create RAG service
        if args.docs_path:
            docs_path = Path(args.docs_path)
            indices_path = config["paths"]["indices_dir"]
            print(f"Using existing documentation directory: {docs_path}")
        else:
            docs_path, indices_path = await create_rag_service(config)
        
        # Launch service unless skip-launch is specified
        if not args.skip_launch:
            await launch_service(docs_path, indices_path, config)
        else:
            print("\nRAG service setup completed successfully!")
            print("Service is ready but not launched (--skip-launch specified)")
            print(f"\nTo launch the service manually, run:")
            print(f"python launch_gradio.py --docs-path {docs_path} --indices-path {indices_path}")
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

## auto_rag_service/deploy/model-definition.yaml

```yaml
models:
  - name: "RAG Assistant WebUI"
    model_path: "/models"
    service:
      pre_start_actions:
        - action: run_command
          args:
            command: ["/bin/bash", "/models/auto_rag_service/setup.sh"]
      start_command:
        - /bin/bash
        - /models/auto_rag_service/start.sh
        - docs_path
      port: 8000
```

## auto_rag_service/deploy/setup_test.sh

```bash
#!/bin/bash

# make directory
mkdir test_dir/
chmod +x test_dir/
```

## auto_rag_service/generate_model_definition.py

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

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


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


def generate_model_definition(github_url: str, model_name: str, port: int, service_type: str) -> Dict:
    """
    Generate a model definition for the RAG service.
    
    Args:
        github_url: GitHub URL
        model_name: Model name
        port: Port number
        service_type: Service type (gradio or fastapi)
        
    Returns:
        Model definition as a dictionary
    """
    # Parse the GitHub URL
    owner, repo, branch, path = parse_github_url(github_url)
    
    # Determine the docs path argument
    docs_path_arg = path if path else ""
    
    # Create the model definition
    model_definition = {
        "models": [
            {
                "name": model_name,
                "model_path": "/models",
                "service": {
                    "pre_start_actions": [
                        "/bin/bash",
                        "/models/RAGModelService/auto_rag_service/setup.sh",
                        github_url
                    ],
                    "start_command": [
                        "/bin/bash",
                        "/models/RAGModelService/auto_rag_service/start.sh",
                        docs_path_arg
                    ],
                    "docs_path_arg": docs_path_arg,
                    "port": port,
                    "repo_owner": owner,
                    "repo_name": repo
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

## auto_rag_service/launch_gradio.py

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
    python launch_gradio.py --indices-path ./my_indices \
                       --docs-path ./my_docs \
                       --host 127.0.0.1 \
                       --port 8080 \
                       --share \
                       --openai-model gpt-4o \
                       --temperature 0.2 \
                       --max-results 5 \
                       --title "My Custom Documentation Assistant" \
                       --description "Search through project documentation" \
                       --suggested-questions "How do I install?" "What are the features?"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
from dotenv import load_dotenv
import gradio as gr

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from vectordb_manager.vectordb_manager import VectorDBManager
from app.rag_chatbot import LLMConfig, RAGManager
from app.gradio_app import create_gradio_interface

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gradio Server Launcher for RAG Service"
    )
    
    # Paths
    parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to vector indices",
        default="./embedding_indices",
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to documentation directory",
        default="./github_docs",
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
        help="OpenAI model to use for RAG",
        default="gpt-4o",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM responses",
        default=0.2,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results to retrieve",
        default=5,
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
    # Resolve paths
    indices_path = Path(args.indices_path)
    docs_path = Path(args.docs_path)
    
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
    config = {
        "paths": {
            "indices_path": indices_path,
            "docs_path": docs_path,
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "share": args.share,
        },
        "llm": {
            "model_name": args.openai_model,
            "temperature": args.temperature,
            "max_results": args.max_results,
            "streaming": True,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
        "ui": {
            "title": args.title or f"{args.openai_model} Documentation Assistant",
            "description": args.description,
            "suggested_questions": args.suggested_questions or [
                "What are the main features?",
                "How do I install this?",
                "What configuration options are available?",
                "How do I contribute to this project?",
                "What license is this project under?",
            ],
        },
    }
    
    return config


async def initialize_server(config: Dict) -> Tuple[VectorDBManager, RAGManager]:
    """
    Initialize the server components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of VectorDBManager and RAGManager
    """
    try:
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(
            docs_root=config["paths"]["docs_path"],
            indices_path=config["paths"]["indices_path"]
        )
        
        # Load vector indices
        await vector_manager.load_index()
        
        if not vector_manager.index:
            logger.warning(
                "No vector index found, trying to create one",
                docs_path=str(config["paths"]["docs_path"]),
                indices_path=str(config["paths"]["indices_path"])
            )
            print("No vector index found. Attempting to create one from documentation...")
            
            # Collect documents
            documents = await vector_manager.collect_documents()
            
            if not documents:
                logger.error("No documents found for indexing")
                print(f"Error: No documents found in {config['paths']['docs_path']}. Please check the path.")
                return None, None
            
            # Create indices
            await vector_manager.create_indices(documents)
            
            # Load the newly created index
            await vector_manager.load_index()
            
            if not vector_manager.index:
                logger.error("Failed to create vector index")
                print("Error: Failed to create vector index.")
                return None, None
        
        # Initialize LLMConfig
        llm_config = LLMConfig(
            openai_api_key=config["llm"]["openai_api_key"],
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_results=config["llm"]["max_results"],
            streaming=config["llm"]["streaming"],
        )
        
        # Initialize RAGManager
        rag_manager = RAGManager(llm_config, vector_manager)
        
        return vector_manager, rag_manager
        
    except Exception as e:
        logger.error("Error initializing server", error=str(e))
        print(f"Error initializing server: {str(e)}")
        return None, None


def customize_gradio_interface(interface: gr.Blocks, config: Dict) -> gr.Blocks:
    """
    Apply customization to the Gradio interface.
    
    Args:
        interface: Gradio interface
        config: Configuration dictionary
        
    Returns:
        Customized Gradio interface
    """
    interface.title = config["ui"]["title"]
    
    # Change suggested questions if provided
    if "suggested_questions" in config["ui"]:
        # Find Markdown elements containing "Suggested Questions"
        for component in interface.blocks.values():
            if isinstance(component, gr.Markdown) and "Suggested Questions" in component.value:
                component.value = "### Suggested Questions"
    
    return interface


async def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Configure RAG system
        config = configure_rag_system(args)
        
        # Initialize server components
        vector_manager, rag_manager = await initialize_server(config)
        
        if not vector_manager or not rag_manager:
            return 1
        
        # Create Gradio interface
        interface = create_gradio_interface(rag_manager, docs_path=config['paths']['docs_path'])
        
        # Apply customization
        interface = customize_gradio_interface(interface, config)
        
        # Launch the interface
        print(f"\nLaunching Gradio server with {config['llm']['model_name']} for RAG...")
        print(f"Documents path: {config['paths']['docs_path']}")
        print(f"Vector indices path: {config['paths']['indices_path']}")
        
        # Set up suggested questions if provided through arguments
        if args.suggested_questions:
            suggestion_questions = args.suggested_questions
            print(f"Using custom suggested questions: {suggestion_questions}")
        
        # Launch the server
        interface.launch(
            server_name=config["server"]["host"],
            server_port=config["server"]["port"],
            share=config["server"]["share"],
            debug=True,
        )
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

## auto_rag_service/rag_launcher.py

```python
#!/usr/bin/env python3
"""
RAG Launcher

A simple script to launch the complete RAG service system.

Usage:
    python rag_launcher.py [--portal-only]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the RAG service system"
    )
    
    parser.add_argument(
        "--portal-only",
        action="store_true",
        help="Launch only the portal interface (without example repositories)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the portal interface",
    )
    
    return parser.parse_args()

def check_environment():
    """Check if the environment is properly set up."""
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            pass
    
    if not openai_api_key:
        print("Error: OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with the following content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Check for required scripts
    required_scripts = [
        "auto_rag_service/setup_rag.py",
        "auto_rag_service/launch_gradio.py",
        "auto_rag_service/create_rag_service.py",
        "auto_rag_service/rag_service_portal.py"
    ]
    
    for script in required_scripts:
        if not Path(script).exists():
            breakpoint()
            print(f"Error: Required script {script} not found.")
            return False
    
    return True

def launch_rag_portal(port):
    """Launch the RAG service portal."""
    print(f"Launching RAG service portal on port {port}...")
    
    # Run the portal script
    cmd = [
        sys.executable,
        "rag_service_portal.py",
        "--port", str(port),
    ]
    
    # Execute in the current process (blocking)
    os.execv(sys.executable, cmd)

def setup_example_repositories():
    """Set up example repositories."""
    print("Setting up example repositories...")
    
    example_repos = [
        ("fastai/fastdoc", "FastAI Documentation"),
        ("scikit-learn/scikit-learn", "Scikit-Learn"),
        ("huggingface/transformers", "Hugging Face Transformers"),
    ]
    
    for repo, name in example_repos:
        print(f"Setting up {name} repository...")
        
        # Create output directory
        output_dir = Path(f"./examples/{repo.split('/')[1]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run setup_rag.py
        cmd = [
            sys.executable,
            "auto_rag_service/setup_rag.py",
            "--github-url", f"https://github.com/{repo}",
            "--output-dir", str(output_dir),
            "--indices-path", str(output_dir / "indices"),
            "--skip-testing",
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {name} repository setup complete.")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error setting up {name} repository: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    if not check_environment():
        return 1
    
    # Create required directories
    Path("./rag_services").mkdir(exist_ok=True)
    
    if not args.portal_only:
        print("This script will:")
        print("1. Set up example repositories (if requested)")
        print("2. Launch the RAG service portal")
        
        setup_option = input("Would you like to set up example repositories? (y/n): ").lower()
        if setup_option == 'y':
            setup_example_repositories()
    
    # Launch the portal
    launch_rag_portal(args.port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## auto_rag_service/rag_service_portal.py

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
    python rag_service_portal.py
"""

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import gradio as gr
from dotenv import load_dotenv
import structlog

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Initialize logger
logger = structlog.get_logger()

# Global state to track running services
SERVICES = {}

class ServiceStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


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


def get_unique_service_id() -> str:
    """Generate a unique service ID"""
    return str(uuid.uuid4())[:8]


def find_available_port(start_port: int = 8000, end_port: int = 9000) -> int:
    """Find an available port in the specified range"""
    import socket
    
    # Check if port is already in use by a service
    used_ports = [s["port"] for s in SERVICES.values() if "port" in s]
    
    for port in range(start_port, end_port):
        if port in used_ports:
            continue
            
        # Check if port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


def create_service_directory(service_id: str) -> Path:
    """
    Create a directory for the service.
    
    Args:
        service_id: Service ID
        
    Returns:
        Path to the service directory
    """
    service_dir = Path("./rag_services") / service_id
    service_dir.mkdir(parents=True, exist_ok=True)
    return service_dir


def generate_model_definition(github_url: str, service_dir: Path) -> Optional[str]:
    """
    Generate a model definition YAML file for the RAG service.
    
    Args:
        github_url: GitHub URL for documentation
        service_dir: Path to the service directory
        
    Returns:
        Path to the generated model definition file, or None if generation failed
    """
    try:
        # Parse GitHub URL to extract components for the filename
        pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
        match = re.match(pattern, github_url)
        if not match:
            logger.error(f"Could not parse GitHub URL: {github_url}")
            return None
        
        owner, repo = match.group(1), match.group(2)
        path = match.group(4)  # This will be None if path is not specified
        
        # Generate docs name for the filename
        base_name = f"{owner}-{repo}".lower()
        if path:
            # Replace slashes with hyphens and remove any special characters
            path_part = re.sub(r'[^a-zA-Z0-9-]', '', path.replace('/', '-'))
            docs_name = f"{base_name}-{path_part}"
        else:
            docs_name = base_name
        
        # Create the file path
        model_def_path = service_dir / f"model-definition-{docs_name}.yaml"
        
        # Create the model definition content
        model_def_content = f"""models:
  - name: "RAG Service - {repo}"
    model_path: "/models/RAGModelService"
    service:
      pre_start_actions:
        - action: run_command
          args:
            command: ["/bin/bash", "/models/RAGModelService/auto_rag_service/setup.sh"]
      start_command: ["/bin/bash", "/models/RAGModelService/auto_rag_service/start.sh"]
      port: 8000
"""
        # Write the model definition to file
        with open(model_def_path, "w") as f:
            f.write(model_def_content)
        
        logger.info(f"Model definition written to {model_def_path}")
        return str(model_def_path)
        
    except Exception as e:
        logger.error(f"Error generating model definition: {e}")
        return None

def process_github_url(github_url: str, progress_callback: Optional[callable] = None) -> Dict:
    """
    Process a GitHub URL to create a RAG service using Backend.AI.
    
    Args:
        github_url: GitHub URL for documentation
        progress_callback: Callback function to report progress
        
    Returns:
        Service information dictionary
    """
    try:
        # Generate service ID and create unique service name
        service_id = get_unique_service_id()
        service_name = f"rag_service_{service_id}"
        
        # Create service directory
        service_dir = create_service_directory(service_id)
        docs_dir = service_dir / "docs"
        indices_dir = service_dir / "indices"
        
        # Initialize service information
        service_info = {
            "id": service_id,
            "github_url": github_url,
            "service_name": service_name,
            "status": ServiceStatus.PENDING,
            "message": "Initializing service...",
            "service_dir": service_dir,
            "docs_dir": docs_dir,
            "indices_dir": indices_dir,
        }
        
        # Add to services dictionary
        SERVICES[service_id] = service_info
        
        # Update status
        service_info["status"] = ServiceStatus.PROCESSING
        service_info["message"] = "Processing GitHub repository..."
        
        if progress_callback:
            progress_callback(0.2, "Cloning repository...")
        
        # Run setup_rag.py to clone repository and create vector indices
        setup_cmd = [
            sys.executable,
            "auto_rag_service/setup_rag.py",
            "--github-url", github_url,
            "--output-dir", str(docs_dir),
            "--indices-path", str(indices_dir),
            "--skip-testing",  # Skip testing to speed up the process
        ]
        
        # Run the command and capture output
        setup_process = subprocess.run(
            setup_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        if progress_callback:
            progress_callback(0.6, "Creating vector embeddings...")
        
        service_info["message"] = "Vector embeddings created successfully."
        
        # Generate model definition YAML
        if progress_callback:
            progress_callback(0.8, "Generating model definition...")
        
        model_def_path = generate_model_definition(github_url, service_dir)
        if model_def_path:
            service_info["model_definition_path"] = model_def_path
            service_info["message"] = "Model definition generated successfully."
        
        # Update status
        service_info["status"] = ServiceStatus.READY
        service_info["message"] = "Service is ready! Launching Backend.AI model service..."
        
        # Save service info to file for persistence
        with open(service_dir / "service_info.txt", "w") as f:
            for key, value in service_info.items():
                if key not in ["service_dir", "docs_dir", "indices_dir"]:
                    f.write(f"{key}: {value}\n")
        
        # Start the service in a separate thread
        threading.Thread(
            target=start_service,
            args=(service_id,),
            daemon=True
        ).start()
        
        if progress_callback:
            progress_callback(1.0, "Service is ready!")
        
        return service_info
        
    except Exception as e:
        logger.error(f"Error processing GitHub URL: {e}")
        service_info = {
            "id": service_id if 'service_id' in locals() else get_unique_service_id(),
            "github_url": github_url,
            "status": ServiceStatus.ERROR,
            "message": f"Error creating service: {str(e)}",
        }
        SERVICES[service_info["id"]] = service_info
        
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        
        return service_info


def start_service(service_id: str) -> None:
    """
    Start a RAG service as a Backend.AI model service.
    
    Args:
        service_id: Service ID
    """
    if service_id not in SERVICES:
        logger.error(f"Service {service_id} not found")
        return
    
    service = SERVICES[service_id]
    github_url = service["github_url"]
    try:
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
        model_def_path = service.get("model_definition_path")
        if not model_def_path:
            raise ValueError("Model definition path not found in service info")
        
        # Extract repository name for environment variable
        repo_name = github_url.split("/")[-1]
        if "." in repo_name:
            repo_name = repo_name.split(".")[0]
        
        # Create Backend.AI model service with environment variables
        create_service_cmd = [
            "backend.ai", "service", "create",
            "cr.backend.ai/testing/ngc-pytorch:24.12-pytorch2.6-py312-cuda12.6",
            "auto_rag",
            "1",
            "--name", service_name,
            "--tag", "rag_model_service",
            "--scaling-group", "nvidia-H100",
            "--model-definition-path", f"RAGModelService/{model_def_path}",
            "--public",
            "-e", f"RAG_SERVICE_NAME={service_name}",
            "-e", f"RAG_SERVICE_PATH={service['service_dir']}",
            "-r", "cuda.shares=0",
            "-r", "mem=4g",
            "-r", "cpu=2"
        ]
        # Run the command
        create_result = subprocess.run(
            create_service_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Extract service endpoint and update service info
        for line in create_result.stdout.split('\n'):
            if "Service endpoint" in line:
                service_url = line.split("Service endpoint:")[1].strip()
                service["url"] = service_url
                break
        
        # If we couldn't extract the URL, build a default one
        if "url" not in service:
            # This is a placeholder - actual URL format may vary
            service["url"] = f"https://service.backend.ai/services/{service_name}"
        
        logger.info(f"Service {service_id} started with URL {service['url']}")
        
        # Update status
        service["status"] = ServiceStatus.READY
        service["message"] = f"Service is ready at {service['url']}"
        
    except Exception as e:
        logger.error(f"Error starting service {service_id}: {e}")
        service["status"] = ServiceStatus.ERROR
        service["message"] = f"Error starting service: {str(e)}"

def create_rag_service(github_url: str, progress=gr.Progress()) -> Tuple[str, str, str, str]:
    """
    Create a RAG service from a GitHub URL (Gradio interface function).
    
    Args:
        github_url: GitHub URL for documentation
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (status, message, url, model_definition_path) for the Gradio interface
    """
    # Validate GitHub URL
    if not validate_github_url(github_url):
        return (
            ServiceStatus.ERROR,
            "Invalid GitHub URL. Please enter a valid GitHub repository URL.",
            "",
            ""
        )
    
    # Process GitHub URL with progress updates
    service_info = process_github_url(
        github_url,
        lambda fraction, message: progress(fraction, desc=message)
    )
    
    # Return relevant information for the interface as separate values
    return (
        service_info["status"],
        service_info["message"],
        service_info.get("url", ""),
        service_info.get("model_definition_path", "")
    )


def create_interface() -> gr.Blocks:
    """Create the Gradio interface"""
    with gr.Blocks(title="RAG Service Creator") as interface:
        gr.Markdown("# RAG Service Creator")
        gr.Markdown(
            """
            Create a Retrieval-Augmented Generation (RAG) service from any GitHub repository 
            containing documentation. Simply enter the GitHub URL, and we'll create a service 
            that allows you to query the documentation using natural language.
            """
        )
        
        with gr.Row():
            github_url = gr.Textbox(
                label="GitHub Repository URL",
                placeholder="https://github.com/owner/repo",
                info="Enter a GitHub repository URL containing documentation (markdown files)",
            )
        
        with gr.Row():
            create_button = gr.Button("Create RAG Service", variant="primary")
        
        with gr.Row():
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                message = gr.Textbox(label="Message", interactive=False)
                service_url = gr.Textbox(
                    label="Service URL", 
                    interactive=False,
                    info="Click this link to access your RAG service when ready"
                )
                model_definition_path = gr.Textbox(
                    label="Model Definition Path",
                    interactive=False,
                    info="Path to the generated model definition YAML file for Backend.AI"
                )
        
        with gr.Row():
            gr.Markdown("### Example Repositories:")
        
        with gr.Row():
            pytorch_btn = gr.Button("PyTorch")
            typescript_btn = gr.Button("TypeScript")
            pandas_btn = gr.Button("Pandas")
            fastai_btn = gr.Button("FastAI")
        
        # Set up click handlers for example buttons
        pytorch_btn.click(fn=lambda: "https://github.com/pytorch/pytorch", outputs=github_url)
        typescript_btn.click(fn=lambda: "https://github.com/microsoft/TypeScript", outputs=github_url)
        pandas_btn.click(fn=lambda: "https://github.com/pandas-dev/pandas", outputs=github_url)
        fastai_btn.click(fn=lambda: "https://github.com/fastai/fastai", outputs=github_url)
        
        # Handle form submission
        create_button.click(
            fn=create_rag_service,
            inputs=[github_url],
            outputs=[
                status,
                message,
                service_url,
                model_definition_path,
            ],
        )
        
        # Add help information
        with gr.Accordion("Help & Information", open=False):
            gr.Markdown(
                """
                ## How it works
                
                1. Enter a GitHub repository URL containing documentation
                2. We'll clone the repository and create vector embeddings using OpenAI's embeddings API
                3. A Gradio interface will be launched for querying the documentation
                4. A model definition YAML file will be generated for Backend.AI model service creation
                5. You'll receive a link to access the service when it's ready
                
                ## Tips
                
                - The repository should contain markdown (.md) files
                - For better results, specify a path to the documentation directory, e.g., `https://github.com/owner/repo/tree/main/docs`
                - The service will remain active as long as this portal is running
                - Each service runs on a different port to avoid conflicts
                - The generated model definition can be used to create a Backend.AI model service
                """
            )
    
    return interface


def main():
    """Main function"""
    # Setup environment
    load_dotenv()
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with the following content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return 1
    
    # Create and launch the interface
    interface = create_interface()
    interface.queue()  # Enable queuing
    interface.launch(server_name="0.0.0.0", server_port=8000, share=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## auto_rag_service/setup.sh

```bash
#!/bin/bash
# Setup script for the RAG Model Service
# This script installs the necessary dependencies
# and generates the model definition YAML file

# Install the package in development mode
cd /models/RAGModelService
# Source environment variables
if [ -f /models/RAGModelService/.env ]; then
    source /models/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
cat /models/RAGModelService/.env >> ~/.bashrc
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc


pip install -e .


echo "Setup complete!"

```

## auto_rag_service/setup_rag.py

```python
#!/usr/bin/env python3
"""
GitHub Documentation Processor for RAG Service

This script:
1. Clones a GitHub repository containing documentation
2. Processes the documentation to create vector embeddings
3. Tests the RAG system with sample queries

Usage:
    python setup_rag.py --github-url https://github.com/owner/repo --output-dir ./doc

Advanced Usage:
    python setup_rag.py --github-url https://github.com/owner/repo/tree/branch/path/to/docs \
                   --indices-path ./my_indices \
                   --output-dir ./my_docs \
                   --openai-model gpt-4o \
                   --temperature 0.2 \
                   --max-results 5 \
                   --test-queries "What are the main features?" "How do I install this?"

Using Existing Docs:
    python setup_rag.py --docs-path ./path/to/docs

Skipping Usage:
    # Skip cloning (use existing docs)
    python setup_rag.py --github-url https://github.com/owner/repo --skip-clone --docs-path ./existing_docs

    # Skip indexing (use existing indices)
    python setup_rag.py --github-url https://github.com/owner/repo --skip-indexing

    # Skip testing
    python setup_rag.py --github-url https://github.com/owner/repo --skip-testing

Chatting:
    python -m app.rag_chatbot --docs-path github_docs/docs --indices-path embedding_indices
"""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import aiofiles
from dotenv import load_dotenv
import structlog
from git import Repo
import re

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from vectordb_manager.vectordb_manager import VectorDBManager
from app.rag_chatbot import LLMConfig, RAGManager

# Initialize logger
logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GitHub Documentation Processor for RAG Service"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
    )
    
    # Paths
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to documentation directory (if not using GitHub URL)",
    )
    parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to store vector indices",
        default="./embedding_indices",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for cloned repository",
        default="./github_docs",
    )
    
    # OpenAI settings
    parser.add_argument(
        "--openai-model",
        type=str,
        help="OpenAI model to use for RAG",
        default="gpt-4o",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM responses",
        default=0.2,
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of results to retrieve",
        default=5,
    )
    
    # Test queries
    parser.add_argument(
        "--test-queries",
        type=str,
        nargs="+",
        help="Test queries to run against the RAG system",
        default=[
            "What are the main features?",
            "How do I install this?",
            "What configuration options are available?",
        ],
    )

    # Actions
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning the repository (use existing docs path)",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing the documentation (use existing indices)",
    )
    parser.add_argument(
        "--skip-testing",
        action="store_true",
        help="Skip testing the RAG system",
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


def parse_github_url(github_url: str) -> Tuple[str, str, str, str]:
    """
    Parse GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple containing owner, repo, branch, and path
    """
    # Handle different GitHub URL formats
    # https://github.com/owner/repo
    # https://github.com/owner/repo/tree/branch
    # https://github.com/owner/repo/tree/branch/path/to/docs
    
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Basic URL pattern for GitHub
    basic_pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    match = re.match(basic_pattern, github_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    
    owner = match.group(1)
    repo = match.group(2)
    branch = match.group(3) or "main"  # Default to 'main' if branch is not specified
    path = match.group(4) or ""  # Default to empty string if path is not specified
    
    return owner, repo, branch, path


async def clone_github_repo(github_url: str, target_dir: Path) -> Path:
    """
    Clone repository from GitHub URL to target directory.
    
    Args:
        github_url: GitHub URL
        target_dir: Target directory for the cloned repository
        
    Returns:
        Path to documentation directory
    """
    try:
        # Parse GitHub URL
        owner, repo, branch, path = parse_github_url(github_url)
        logger.info(
            "Parsed GitHub URL",
            owner=owner,
            repo=repo,
            branch=branch,
            path=path
        )
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone repository
        print(f"Cloning repository {owner}/{repo} (branch: {branch})...")
        repo_url = f"https://github.com/{owner}/{repo}.git"
        
        # Use GitPython to clone the repository
        git_repo = Repo.clone_from(repo_url, target_dir, branch=branch)
        
        logger.info("Repository cloned successfully", target_dir=str(target_dir))
        print(f"Repository cloned to {target_dir}")
        
        # Determine documentation directory
        if path:
            docs_path = target_dir / path
            if not docs_path.exists():
                raise ValueError(f"Documentation path not found: {docs_path}")
        else:
            # Default to the root of the repository
            docs_path = target_dir
        
        return docs_path
        
    except Exception as e:
        logger.error("Error cloning repository", error=str(e))
        print(f"Error cloning repository: {str(e)}")
        raise


async def process_documentation(docs_path: Path, indices_path: Path) -> None:
    """
    Process documentation to create vector indices.
    
    Args:
        docs_path: Path to documentation directory
        indices_path: Path to store vector indices
    """
    try:
        print(f"Processing documentation in {docs_path}...")
        
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_path)
        
        # Collect documents
        print("Collecting documents...")
        documents = await vector_manager.collect_documents()
        
        if not documents:
            logger.warning("No documents found", docs_path=str(docs_path))
            print(f"Warning: No documents found in {docs_path}")
            return
        
        print(f"Found {len(documents)} documents")
        
        # Create vector indices
        print("Creating vector indices...")
        await vector_manager.create_indices(documents)
        
        print(f"Vector indices created successfully in {indices_path}")
        
    except Exception as e:
        logger.error("Error processing documentation", error=str(e))
        print(f"Error processing documentation: {str(e)}")
        raise


async def test_rag_system(indices_path: Path, docs_path: Path, test_queries: List[str], model_name: str = "gpt-4o", temperature: float = 0.2, max_results: int = 5) -> None:
    """
    Test RAG system with sample queries.
    
    Args:
        indices_path: Path to vector indices
        docs_path: Path to documentation directory
        test_queries: List of test queries
        model_name: OpenAI model name
        temperature: Temperature for LLM responses
        max_results: Maximum number of results to retrieve
    """
    try:
        print("\nTesting RAG system...")
        
        # Initialize VectorDBManager
        vector_manager = VectorDBManager(docs_path, indices_path)
        
        # Load vector index
        await vector_manager.load_index()
        
        if not vector_manager.index:
            logger.error("Failed to load vector index", indices_path=str(indices_path))
            print(f"Error: Failed to load vector index from {indices_path}")
            return
        
        # Initialize RAGManager
        llm_config = LLMConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            model_name=model_name,
            temperature=temperature,
            max_results=max_results,
            streaming=False,  # Disable streaming for testing
        )
        
        rag_manager = RAGManager(llm_config, vector_manager)
        
        # Run test queries
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest Query {i}: \"{query}\"")
            
            # Get context
            context, results = await rag_manager._get_relevant_context(query)
            
            print(f"Found {len(results)} relevant documents")
            
            # Print top 2 documents for reference
            for j, result in enumerate(results[:2], 1):
                metadata = result.get("metadata", {})
                similarity = result.get("similarity_score", 0.0)
                relative_path = metadata.get("relative_path", "unknown")
                
                print(f"  Document {j}: {relative_path} (score: {similarity:.4f})")
            
            # Generate response
            print("Generating response...")
            full_response = ""
            async for chunk in rag_manager.generate_response_with_context(query, context):
                full_response += chunk
            
            print(f"Response: {full_response}")
            print("-" * 80)
        
    except Exception as e:
        logger.error("Error testing RAG system", error=str(e))
        print(f"Error testing RAG system: {str(e)}")
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
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Resolve paths
        output_dir = Path(args.output_dir) if args.output_dir else Path("./github_docs")
        indices_path = Path(args.indices_path) if args.indices_path else Path("./embedding_indices")
        
        # Clone repository if URL provided and not skipped
        docs_path = None
        if args.github_url and not args.skip_clone:
            docs_path = await clone_github_repo(args.github_url, output_dir)
        elif args.docs_path:
            docs_path = Path(args.docs_path)
            if not docs_path.exists():
                logger.error("Documentation path does not exist", path=str(docs_path))
                print(f"Error: Documentation path does not exist: {docs_path}")
                return 1
        else:
            logger.error("No GitHub URL or documentation path provided")
            print("Error: Either --github-url or --docs-path must be provided")
            return 1
        
        # Process documentation if not skipped
        if not args.skip_indexing:
            await process_documentation(docs_path, indices_path)
        
        # Test RAG system if not skipped
        if not args.skip_testing:
            await test_rag_system(
                indices_path,
                docs_path,
                args.test_queries,
                args.openai_model,
                args.temperature,
                args.max_results
            )
        
        print("\nSuccess! RAG system is set up and ready to use.")
        print(f"Documentation: {docs_path}")
        print(f"Vector indices: {indices_path}")
        
        # Provide hint for next steps
        print("\nNext steps:")
        print("1. Run the RAG chat interface:")
        print(f"   python -m app.rag_chatbot --docs-path {docs_path} --indices-path {indices_path}")
        print("2. Or run the Gradio web interface:")
        print(f"   python app/gradio_app.py")
        
        return 0
        
    except Exception as e:
        logger.error("Error in main function", error=str(e))
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

## auto_rag_service/start.sh

```bash
#!/bin/bash
# Start script for the RAG Model Service
# This script starts the RAG service with the specified docs path

# Source environment variables
if [ -f /models/RAGModelService/.env ]; then
    source /models/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

# Get the docs path from the first argument


# Launch the RAG service using create_rag_service.py
cd /models/RAGModelService
python /models/RAGModelService/auto_rag_service/launch_gradio.py \
    --docs-path $BACKEND_MODEL_PATH/${RAG_SERVICE_PATH}/"docs/docs" \
    --indices-path $BACKEND_MODEL_PATH/${RAG_SERVICE_PATH}/"indices" \
    --host "0.0.0.0" \
    --port 8000

# Keep the container running
tail -f /dev/null
```

## auto_rag_service/test_model_definition.py

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

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the function to test
from auto_rag_service.generate_model_definition import (
    parse_github_url,
    generate_model_name,
    generate_docs_name,
    generate_model_definition
)

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
        model_def = generate_model_definition(url, model_name, 8000, "gradio")
        
        print(f"  Model Name: {model_name}")
        print(f"  Start Command Path Arg: {model_def['models'][0]['service']['start_command'][2]}")

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
        
        # Generate a model definition
        model_def = generate_model_definition(url, model_name, 8000, "gradio")
        
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
        print(f"      pre_start_actions: {service_value['pre_start_actions']}")
        print(f"      start_command: {service_value['start_command']}")
        print(f"      docs_path_arg: {service_value['docs_path_arg']}")
        print(f"      port: {service_value['port']}")

def main():
    """Main function."""
    test_url_parsing()
    test_blob_url_parsing()
    test_model_definition_generation()
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## config/__init__.py

```python

```

## config/config.py

```python
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


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o"))
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
```

## debug/test_backend_ai.py

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

## debug/test_custom_path.sh

```bash
#!/bin/bash
# Test script for the RAG Model Service with a custom docs path

# Set the custom path
export RAG_SERVICE_PATH="rag_services/303568ea"

# Source environment variables
if [ -f .env ]; then
    source .env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

echo "Using custom RAG_SERVICE_PATH: ${RAG_SERVICE_PATH}"

# Launch the RAG service using launch_gradio.py
python auto_rag_service/launch_gradio.py \
    --docs-path ./${RAG_SERVICE_PATH}/docs \
    --indices-path ./${RAG_SERVICE_PATH}/indices \
    --host "127.0.0.1" \
    --port 8000
```

## debug/test_default_path.sh

```bash
#!/bin/bash
# Test script for the RAG Model Service with the default docs path

# Unset RAG_SERVICE_PATH to ensure we use the default path
unset RAG_SERVICE_PATH
export RAG_SERVICE_PATH="rag_services/303568ea"
# Source environment variables
if [ -f .env ]; then
    source .env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

echo "Using default docs path from config"

# Launch the RAG service using launch_gradio.py without specifying a docs path
# This will use the default path from config.py
python auto_rag_service/launch_gradio.py \
    --host "127.0.0.1" \
    --port 8001
```

## doc_processor/__init__.py

```python

```

## doc_processor/vectorize.sh

```bash
#!/bin/bash

# Set paths
DOCS_PATH="/Users/lablup/Documents/GitHub/backend-ai-assistant/docs_md"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean up existing files
echo "Cleaning up existing files..."
# rm -rf "$DOCS_PATH"/*

# Convert RST to MD
echo "Converting RST files to Markdown..."
python3 "$SCRIPT_DIR/convert_docs.py"

# Vectorize the converted files
echo "Vectorizing Markdown files..."
python3 "$SCRIPT_DIR/../vectordb_manager/cli_vectorizer.py" process "$DOCS_PATH"

echo "Done! Files have been converted and vectorized."
```

## fastapi_server.py

```python
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
        model_name="gpt-4o",  # Match the model from RAGManager
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
```

## model-definition-fastapi.yaml

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

## model-definition-portal.yaml

```yaml
models:
  - name: "TensroRT-LLM RAG Portal"
    model_path: "/Users/sergeyleksikov/Documents/GitHub/RAGModelService"
    service:
      pre_start_actions:
        - action: run_command
          args:
            command: ["/bin/bash", "/models/RAGModelService/setup_portal.sh"]
      start_command:
        - /bin/bash
        - /models/RAGModelService/start_portal.sh
      port: 8000
```

## model-definition.yaml

```yaml
models:
  - name: "TensroRT-LLM RAG Service"
    model_path: "/models"
    service:
      pre_start_actions:
        - action: run_command
          args:
            command: ["/bin/bash", "/models/RAGModelService/setup.sh"]
      start_command:
        - python3
        - /models/RAGModelService/app/gradio_app.py
      port: 8000
```

## run_fastapi_server.sh

```bash
#!/bin/bash

cd /models/RAGModelService/

source .env
export OPENAI_API_KEY=""
python3 fastapi_server.py
```

## scripts/codebase_to_markdown.py

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

## scripts/python_to_markdown.py

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

## setup.py

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
        "console_scripts": [
            "rag-vectorize=vectordb_manager.cli_vectorizer:app",
            "rag-chat=app.rag_chatbot:main",
            "rag-web=app.gradio_app:main",
        ],
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

## setup.sh

```bash
#!/bin/bash
# Setup script for RAG Model Service
# This script installs dependencies and sets up environment variables

# Install the package in development mode
pip install -e .

# Source environment variables
if [ -f .env ]; then
    source .env
    echo "Environment variables loaded from .env"
    # Add environment variables to bashrc for persistence
    echo "$(cat .env)" >> ~/.bashrc
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

# Check if GitHub URL is provided
if [ -z "$1" ]; then
    echo "Error: GitHub URL is required"
    echo "Usage: $0 <github-url>"
    exit 1
fi

# Generate model definition YAML file
echo "Generating model definition for GitHub URL: $1"
python auto_rag_service/generate_model_definition.py --github-url "$1"

# Print success message
echo "Setup completed successfully!"
```

## setup_portal.sh

```bash
#!/bin/bash

cd /models/RAGModelService

pip install -e .

pip install backend.ai-client==25.4.0

source /models/RAGModelService/.env

echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
cat /models/RAGModelService/.env >> ~/.bashrc
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc
```

## start_portal.sh

```bash
#!/bin/bash

cd /models/RAGModelService

source /models/RAGModelService/.env

python3 auto_rag_service/rag_service_portal.py
```

## vectordb_manager/__init__.py

```python

```

## vectordb_manager/cli_vectorizer.py

```python
#!/usr/bin/env python3

"""
CLI for vectorizing and searching documents

Usage:
    # Process documents
    python cli_vectorizer.py process /home/work/RAGModelService/TensorRT-LLM/docs/source

    # Search in vector store
    python cli_vectorizer.py search "Login"
"""


import asyncio
from pathlib import Path
from typing import List

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from vectordb_manager import VectorDBManager

app = typer.Typer()
console = Console()

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)


@app.command()
def process(
    docs_path: Path = typer.Argument(
        ...,
        help="Path to docs directory containing markdown files",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    indices_path: Path = typer.Option(
        "./embedding_indices",
        "--indices-path",
        "-i",
        help="Path to store FAISS index",
    ),
):
    """Process markdown documents and create vector index"""

    async def run():
        try:
            manager = VectorDBManager(docs_path, indices_path)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Collect documents
                progress.add_task(description="Collecting documents...", total=None)
                documents = await manager.collect_documents()

                # Create index
                progress.add_task(description="Creating index...", total=None)
                await manager.create_indices(documents)

            console.print("\n[green]✓[/green] Document processing completed!")

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)

    asyncio.run(run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    indices_path: Path = typer.Option(
        "./embedding_indices", "--indices-path", "-i", help="Path to FAISS index"
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Maximum number of results"
    ),
):
    """Search in vector index"""

    async def run():
        try:
            manager = VectorDBManager(Path("."), indices_path)
            await manager.load_index()

            results = await manager.search_documents(query, limit)

            if not results:
                console.print("\n[yellow]No results found[/yellow]")
                return

            console.print("\n[bold]Search Results:[/bold]")
            console.print(f"Found {len(results)} results:")

            for i, result in enumerate(results, 1):
                console.print(f"\n{i}. Score: {result['similarity_score']:.4f}")
                console.print(f"File: {result['metadata']['relative_path']}")
                console.print(f"Content preview: {result['content'][:200]}...")

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)

    asyncio.run(run())


if __name__ == "__main__":
    app()
```

## vectordb_manager/vectordb_manager.py

```python
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
import os
from dotenv import load_dotenv


logger = structlog.get_logger()


class DocumentMetadata(BaseModel):
    """Metadata for processed documents"""

    relative_path: str
    filename: str
    last_updated: datetime
    file_size: int


class VectorDBManager:
    def __init__(self, docs_root: Path, indices_path: Path):
        self.docs_root = Path(docs_root)
        self.indices_path = Path(indices_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.logger = logger.bind(component="VectorDBManager")
        self.index: Optional[FAISS] = None
        self.index_name = "vectorstore"

    async def read_markdown_file(self, file_path: Path) -> Optional[str]:
        """Read markdown file content"""
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            self.logger.error("File read error", path=str(file_path), error=str(e))
            return None

    def create_metadata(self, file_path: Path) -> DocumentMetadata:
        """Create metadata for a document"""
        stats = file_path.stat()
        return DocumentMetadata(
            relative_path=str(file_path.relative_to(self.docs_root)),
            filename=file_path.name,
            last_updated=datetime.fromtimestamp(stats.st_mtime),
            file_size=stats.st_size,
        )

    async def collect_documents(self) -> List[Document]:
        """Collect all documents from the documentation directory"""
        docs = []

        if not self.docs_root.exists():
            self.logger.warning("Documentation directory not found", path=str(self.docs_root))
            return docs

        for file_path in self.docs_root.rglob("*.md"):
            content = await self.read_markdown_file(file_path)
            if content:
                metadata = self.create_metadata(file_path)
                docs.append(
                    Document(page_content=content, metadata=metadata.model_dump())
                )

        self.logger.info("Collected documents", count=len(docs))
        return docs

    async def create_indices(self, documents: List[Document]) -> None:
        """Create a single FAISS index from collected documents"""
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
        index_path = self.indices_path / self.index_name
        if not index_path.exists():
            self.logger.warning("Index directory not found")
            return

        try:
            self.index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("Loaded index")
        except Exception as e:
            self.logger.error(
                "Failed to load index", error=str(e)
            )

    async def search_documents(
        self, query: str, k: int = 5
    ) -> List[Dict]:
        """Search documents in the index"""
        if not self.index:
            await self.load_index()
            
        if not self.index:
            raise ValueError("No index loaded")

        try:
            docs_with_scores = self.index.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                }
                for doc, score in docs_with_scores
            ]

        except Exception as e:
            self.logger.error("Search failed", error=str(e))
            raise

    async def test_search(self, query: str, k: int = 5) -> List[Document]:
        """Return documents for a given search query."""
        if self.index is None:
            await self.load_index()
            if self.index is None:
                raise ValueError("No loaded index available.")
        
        return self.index.similarity_search(query, k=k)


async def interactive_mode(vector_manager: VectorDBManager, docs_root: Path, indices_path: Path) -> None:
    """Run the interactive CLI menu for the VectorDBManager."""
    while True:
        print("\n----- VectorDBManager Test Menu -----")
        print("1. Collect and print documents")
        print("2. Create vector index")
        print("3. Load existing vector index")
        print("4. Search documents")
        print("5. Change documentation path")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            try:
                print("\nCollecting documents...")
                documents = await vector_manager.collect_documents()
                print(f"Collected {len(documents)} documents.")
                if documents:
                    print("\nSample documents:")
                    for i, doc in enumerate(documents[:3]):  # Show first 3 docs as samples
                        print(f"\nDocument {i+1}:")
                        print(f"Content (first 100 chars): {doc.page_content[:100]}...")
                        print(f"Metadata: {doc.metadata}")
            except Exception as e:
                print(f"Error collecting documents: {str(e)}")
        
        elif choice == "2":
            try:
                print("\nCreating vector index...")
                documents = await vector_manager.collect_documents()
                
                if not documents:
                    print("No documents found to index.")
                    continue
                
                print(f"Creating index from {len(documents)} documents...")
                await vector_manager.create_indices(documents)
                print("Vector index created successfully.")
            except Exception as e:
                print(f"Error creating index: {str(e)}")
        
        elif choice == "3":
            try:
                print("\nLoading vector index...")
                await vector_manager.load_index()
                if vector_manager.index:
                    print("Vector index loaded successfully.")
                else:
                    print("No vector index found to load.")
            except Exception as e:
                print(f"Error loading index: {str(e)}")
        
        elif choice == "4":
            if not vector_manager.index:
                print("No index loaded. Please load or create an index first.")
                continue
            
            query = input("\nEnter your search query: ")
            if not query.strip():
                continue
            
            try:
                print(f"\nSearching for: '{query}'")
                results = await vector_manager.test_search(query)
                
                print(f"\nFound {len(results)} results.")
                for i, doc in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"Content (first 100 chars): {doc.page_content[:100]}...")
                    print(f"Metadata: {doc.metadata}")
                    if hasattr(doc, 'distance') and doc.distance is not None:
                        print(f"Relevance score: {1 - doc.distance:.4f}")
            except Exception as e:
                print(f"Error during search: {str(e)}")
                
        elif choice == "5":
            print(f"\nCurrent documentation path: {docs_root}")
            new_path = input("Enter new documentation path (or press Enter to keep current): ").strip()
            
            if new_path:
                try:
                    new_path = Path(new_path)
                    if not new_path.exists():
                        print(f"Warning: Path {new_path} does not exist. Creating a new VectorDBManager anyway.")
                    
                    # Create a new VectorDBManager with the updated path
                    docs_root = new_path
                    vector_manager = VectorDBManager(docs_root, indices_path)
                    print(f"Documentation path updated to: {docs_root}")
                except Exception as e:
                    print(f"Error updating path: {str(e)}")
        
        elif choice == "6":
            print("Exiting. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


async def collect_and_print_documents(vector_manager: VectorDBManager, verbose: bool = False) -> None:
    """Collect and print document information."""
    try:
        print("\nCollecting documents...")
        documents = await vector_manager.collect_documents()
        print(f"Collected {len(documents)} documents.")
        
        if documents and verbose:
            print("\nSample documents:")
            for i, doc in enumerate(documents[:5]):  # Show first 5 docs as samples
                print(f"\nDocument {i+1}:")
                print(f"Content (first 100 chars): {doc.page_content[:100]}...")
                print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error collecting documents: {str(e)}")
        raise


async def create_index(vector_manager: VectorDBManager) -> None:
    """Create vector index from documents."""
    try:
        print("\nCreating vector index...")
        documents = await vector_manager.collect_documents()
        
        if not documents:
            print("No documents found to index.")
            return
        
        print(f"Creating index from {len(documents)} documents...")
        await vector_manager.create_indices(documents)
        print("Vector index created successfully.")
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        raise


async def load_index(vector_manager: VectorDBManager) -> None:
    """Load existing vector index."""
    try:
        print("\nLoading vector index...")
        await vector_manager.load_index()
        if vector_manager.index:
            print("Vector index loaded successfully.")
        else:
            print("No vector index found to load.")
            raise ValueError("No vector index found to load.")
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        raise


async def search_documents(vector_manager: VectorDBManager, query: str, k: int = 5) -> None:
    """Search documents in the index."""
    try:
        if not vector_manager.index:
            await load_index(vector_manager)
        
        print(f"\nSearching for: '{query}'")
        results = await vector_manager.test_search(query, k=k)
        
        print(f"\nFound {len(results)} results.")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content (first 100 chars): {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
            if hasattr(doc, 'distance') and doc.distance is not None:
                print(f"Relevance score: {1 - doc.distance:.4f}")
    except Exception as e:
        print(f"Error during search: {str(e)}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VectorDBManager - Create and search vector indices from documentation."
    )
    
    # Path arguments
    parser.add_argument(
        "--docs-path", 
        type=str, 
        help="Path to documentation directory"
    )
    parser.add_argument(
        "--indices-path", 
        type=str, 
        help="Path to store vector indices"
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--collect", 
        action="store_true", 
        help="Collect and print documents"
    )
    action_group.add_argument(
        "--create-index", 
        action="store_true", 
        help="Create vector index from documents"
    )
    action_group.add_argument(
        "--load-index", 
        action="store_true", 
        help="Load existing vector index"
    )
    action_group.add_argument(
        "--search", 
        type=str, 
        metavar="QUERY", 
        help="Search documents with the provided query"
    )
    action_group.add_argument(
        "--interactive", 
        action="store_true", 
        help="Start in interactive mode"
    )
    
    # Additional options
    parser.add_argument(
        "-k", 
        "--top-k", 
        type=int, 
        default=5, 
        help="Number of results to return for search (default: 5)"
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show more detailed output"
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main function to run the VectorDBManager CLI."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or in your environment.")
        return
    
    # Set environment variable for OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    docs_root = Path(args.docs_path) if args.docs_path else base_dir / "docs"
    indices_path = Path(args.indices_path) if args.indices_path else base_dir / "embedding_indices"
    
    print(f"Initializing with docs path: {docs_root}")
    print(f"Vector indices path: {indices_path}")
    
    # Initialize VectorDBManager
    vector_manager = VectorDBManager(docs_root, indices_path)
    
    # Execute the requested action
    try:
        if args.collect:
            await collect_and_print_documents(vector_manager, args.verbose)
        elif args.create_index:
            await create_index(vector_manager)
        elif args.load_index:
            await load_index(vector_manager)
        elif args.search:
            await search_documents(vector_manager, args.search, args.top_k)
        else:
            # Default to interactive mode if no specific action is requested
            await interactive_mode(vector_manager, docs_root, indices_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)
```

