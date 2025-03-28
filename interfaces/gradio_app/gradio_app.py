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
                    file_name = file_path.name
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
                    file_name = relative_path.name
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
