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
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from config.config import load_config
from app.rag_chatbot import LLMConfig, RAGManager
from vectordb_manager.vectordb_manager import VectorDBManager

# Initialize logger and console
logger = logging.getLogger(__name__)
console = Console()

# Load configuration
config = load_config()

DOC_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

class ChatMessage(BaseModel):
    """Single chat message model"""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatState(BaseModel):
    """State management for chat interface"""

    messages: List[ChatMessage] = Field(default_factory=list)
    current_docs: Dict[str, str] = Field(default_factory=dict)
    selected_doc: Optional[str] = Field(None)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
                        doc_rel_path = file_path.relative_to(project_root / "docs_md")
                        img_file = (
                            project_root
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
                        doc_rel_path = file_path.relative_to(project_root / "docs_md")
                        img_file = (
                            project_root
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
) -> gr.Blocks:
    """Create Gradio interface for the RAG chat application"""

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
                full_path = config.paths.docs_root / relative_path

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
                    else await read_markdown_file(
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
                    content = await read_markdown_file(file_path)
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


async def main():
    """Main entry point"""
    try:
        # Initialize vector store
        vector_store = VectorDBManager(
            docs_root=config.paths.docs_root, indices_path=config.paths.indices_path
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
        interface = create_gradio_interface(rag_manager)
        interface.launch(
            server_name=config.server.host,
            server_port=config.server.port,
            share=config.server.share_enabled,
            debug=True,
            theme=gr.themes.Base(),
        )

    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        console.print(f"[red]Error initializing application: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
