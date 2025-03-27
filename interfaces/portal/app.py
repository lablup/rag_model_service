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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import gradio as gr
from dotenv import load_dotenv
import structlog
import asyncio

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import utility modules
from utils.github_utils import validate_github_url, parse_github_url, GitHubInfo
from utils.service_utils import (
    setup_environment, 
    get_unique_service_id,
    ServiceStatus
)
from core.document_processor import DocumentProcessor
from data.vector_store import VectorStore
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from config.config import LLMConfig

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


def generate_model_definition(github_url: str, service_dir: Path) -> Optional[Path]:
    """
    Generate a model definition YAML file for the RAG service.
    
    Args:
        github_url: GitHub URL for documentation
        service_dir: Path to the service directory
        
    Returns:
        Path to the generated model definition file, or None if generation failed
    """
    try:
        github_info = parse_github_url(github_url)
        service_id = service_dir.name
        
        # Create model definition file
        model_def_path = service_dir / f"rag-{service_id}.yml"
        
        repo_name = github_info.repo
        repo_title = repo_name.replace("-", " ").replace("_", " ").title()
        
        # Create model definition YAML
        model_def = f"""
# Model Definition for RAG Service: {repo_title}
model:
  name: rag-{service_id}
  version: 0.1.0
  description: RAG Service for {repo_title} Documentation
  type: inference
  vendor: ai
  labels:
    source: {github_info.owner}/{github_info.repo}
    branch: {github_info.branch}
    service_id: {service_id}
    created_time: {time.strftime("%Y-%m-%d %H:%M:%S")}
    
# Runtime configuration
runtime:
  type: python
  path: /opt/rag/
  command:
    - ./start.sh
  gpu:
    min: 0
    max: 0
  memory:
    min: 1g
    max: 4g
    
# Service configuration
service:
  ports:
    - name: gradio
      protocol: http
      port: 7860
  models:
    - path: /{service_id}
"""
        
        # Write model definition to file
        with open(model_def_path, "w") as f:
            f.write(model_def)
            
        print(f"Generated model definition: {model_def_path}")
        logger.info("Generated model definition", path=str(model_def_path))
        
        return model_def_path
        
    except Exception as e:
        logger.error("Error generating model definition", error=str(e))
        print(f"Error generating model definition: {str(e)}")
        return None
                

async def process_github_url(github_url: str, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Process a GitHub URL to create a RAG service using Backend.AI.
    
    Args:
        github_url: GitHub URL for documentation
        progress_callback: Callback function to report progress
        
    Returns:
        Service information dictionary
    """
    try:
        # Setup environment
        setup_environment()
        
        # Generate unique service ID
        service_id = get_unique_service_id()
        
        # Create service directory
        service_dir = create_service_directory(service_id)
        docs_dir = service_dir / "docs"
        indices_dir = service_dir / "indices"
        indices_dir.mkdir(exist_ok=True)
        
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
        }
        
        # Store service info in global state
        SERVICES[service_id] = service_info
        
        # Report progress
        if progress_callback:
            progress_callback(0.1, f"Created service directory: {service_dir}")
            
        # Process GitHub URL
        document_processor = DocumentProcessor()
        
        # Report progress
        if progress_callback:
            progress_callback(0.2, "Cloning GitHub repository...")
            
        # Clone GitHub repository
        docs_path, error = await document_processor.clone_github_repository(
            github_url=github_url,
            target_dir=docs_dir
        )
        
        if error:
            service_info["status"] = ServiceStatus.ERROR
            service_info["error"] = f"Failed to clone repository: {error}"
            logger.error("Failed to clone repository", error=str(error))
            return service_info
            
        # Report progress
        if progress_callback:
            progress_callback(0.4, "Repository cloned successfully")
            
        # Initialize vector store
        vector_store = VectorStore(docs_path, indices_dir)
        
        # Report progress
        if progress_callback:
            progress_callback(0.5, "Processing documentation...")
            
        # Process documents and create index
        try:
            docs = await document_processor.collect_documents(docs_path)
            if not docs:
                service_info["status"] = ServiceStatus.ERROR
                service_info["error"] = "No documents found in repository"
                logger.error("No documents found in repository")
                return service_info
                
            # Report progress
            if progress_callback:
                progress_callback(0.7, f"Found {len(docs)} documents, creating vector index...")
                
            # Create index
            await vector_store.create_indices(docs)
            
            # Report progress
            if progress_callback:
                progress_callback(0.8, "Vector index created successfully")
                
        except Exception as e:
            service_info["status"] = ServiceStatus.ERROR
            service_info["error"] = f"Failed to process documents: {str(e)}"
            logger.error("Failed to process documents", error=str(e))
            return service_info
            
        # Generate model definition
        model_def_path = generate_model_definition(github_url, service_dir)
        if model_def_path:
            service_info["model_def_path"] = str(model_def_path)
            
        # Report progress
        if progress_callback:
            progress_callback(0.9, "Preparing service...")
            
        # Create start.sh script
        create_start_script(service_id, service_dir)
        
        # Update service status
        service_info["status"] = ServiceStatus.READY
        
        # Report progress
        if progress_callback:
            progress_callback(1.0, "Service ready to start")
            
        return service_info
        
    except Exception as e:
        logger.error("Error processing GitHub URL", error=str(e))
        return {
            "id": service_id if 'service_id' in locals() else str(uuid.uuid4()),
            "status": ServiceStatus.ERROR,
            "error": str(e)
        }


def create_start_script(service_id: str, service_dir: Path) -> None:
    """Create start.sh script for the service"""
    start_script = service_dir / "start.sh"
    
    script_content = f"""#!/bin/bash
# Start script for RAG Service {service_id}

# Set environment variables
export RAG_SERVICE_PATH="{service_id}"

# Start the Gradio server
python -m interfaces.cli_app.launch_gradio \\
    --indices-path ./${{RAG_SERVICE_PATH}}/indices \\
    --docs-path ./${{RAG_SERVICE_PATH}}/docs \\
    --port 8000 \\
    --host 0.0.0.0
"""
    
    with open(start_script, "w") as f:
        f.write(script_content)
        
    # Make the script executable
    start_script.chmod(0o755)


async def start_service(service_id: str) -> Dict[str, Any]:
    """
    Start a RAG service as a Backend.AI model service.
    
    Args:
        service_id: Service ID
        
    Returns:
        Updated service information
    """
    try:
        if service_id not in SERVICES:
            return {
                "status": ServiceStatus.ERROR,
                "error": f"Service not found: {service_id}"
            }
            
        service_info = SERVICES[service_id]
        
        if service_info["status"] != ServiceStatus.READY:
            return {
                "status": service_info["status"],
                "error": "Service is not ready to start"
            }
            
        # Find available port
        port = find_available_port()
        service_info["port"] = port
        
        # Start service in background thread
        def start_service_thread():
            try:
                service_dir = Path(service_info["service_dir"])
                docs_dir = Path(service_info["docs_dir"])
                indices_dir = Path(service_info["indices_dir"])
                
                cmd = [
                    "python", "-m", "interfaces.cli_app.launch_gradio",
                    "--indices-path", str(indices_dir),
                    "--docs-path", str(docs_dir),
                    "--port", str(port),
                    "--host", "0.0.0.0"
                ]
                
                process = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                service_info["pid"] = process.pid
                service_info["url"] = f"http://localhost:{port}"
                service_info["status"] = ServiceStatus.PROCESSING
                
                print(f"Service started: {service_info['url']}")
                logger.info(
                    "Service started", 
                    pid=process.pid, 
                    url=service_info['url']
                )
                
                # Wait for process to finish
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(
                        "Service exited with error",
                        returncode=process.returncode,
                        stderr=stderr
                    )
                    service_info["status"] = ServiceStatus.ERROR
                    service_info["error"] = stderr
                else:
                    logger.info("Service exited normally")
                    service_info["status"] = ServiceStatus.READY
                    
            except Exception as e:
                logger.error("Error starting service", error=str(e))
                service_info["status"] = ServiceStatus.ERROR
                service_info["error"] = str(e)
        
        # Start thread
        thread = threading.Thread(target=start_service_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for service to start
        for _ in range(10):
            if service_info["status"] == ServiceStatus.PROCESSING:
                break
            await asyncio.sleep(0.5)
            
        return service_info
        
    except Exception as e:
        logger.error("Error starting service", error=str(e))
        return {
            "status": ServiceStatus.ERROR,
            "error": str(e)
        }


async def create_rag_service(github_url: str, progress=gr.Progress()) -> Tuple[str, str, str, str]:
    """
    Create a RAG service from a GitHub URL (Gradio interface function).
    
    Args:
        github_url: GitHub URL for documentation
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (status, message, url, model_definition_path) for the Gradio interface
    """
    try:
        # Validate GitHub URL
        if not validate_github_url(github_url):
            return "Error", "Invalid GitHub URL", "", ""
            
        # Process GitHub URL with progress tracking
        def update_progress(progress_value, description):
            progress(progress_value, description)
            
        # Create progress steps for better visualization
        progress_steps = [
            (0.1, "Creating service directory..."),
            (0.2, "Cloning GitHub repository..."),
            (0.4, "Repository cloned successfully"),
            (0.5, "Processing documentation..."),
            (0.7, "Creating vector index..."),
            (0.8, "Vector index created successfully"),
            (0.9, "Preparing service..."),
            (1.0, "Service ready to start")
        ]
        
        # Start progress tracking
        progress(0.0, "Starting RAG service creation...")
        
        # Process GitHub URL
        service_info = await process_github_url(github_url, update_progress)
        
        if service_info["status"] == ServiceStatus.ERROR:
            return "Error", f"Failed to create service: {service_info.get('error', 'Unknown error')}", "", ""
            
        # Start service
        progress(0.95, "Starting service...")
        service_info = await start_service(service_info["id"])
        
        if service_info["status"] == ServiceStatus.ERROR:
            return "Error", f"Failed to start service: {service_info.get('error', 'Unknown error')}", "", ""
            
        # Return success
        progress(1.0, "Service started successfully!")
        model_def_path = service_info.get("model_def_path", "")
        service_id = service_info["id"]
        service_url = service_info["url"]
        
        return (
            "Success ", 
            f"RAG Service created and started. Service ID: {service_id}", 
            service_url, 
            model_def_path
        )
        
    except Exception as e:
        logger.error("Error creating RAG service", error=str(e))
        return "Error", f"Error: {str(e)}", "", ""


def create_interface() -> gr.Blocks:
    """Create the Gradio interface"""
    # Configure blocks with queue enabled for progress tracking
    blocks = gr.Blocks(
        title="RAG Service Creator", 
        analytics_enabled=False,
    )
    
    # Enable queueing explicitly for the blocks instance
    blocks.queue()
    
    with blocks as interface:
        gr.Markdown(
            """
            # RAG Service Creator
            
            Create a RAG (Retrieval-Augmented Generation) service from a GitHub repository containing documentation.
            
            ## Instructions
            
            1. Enter a GitHub URL containing documentation
            2. Click 'Create RAG Service'
            3. Wait for the service to be created
            4. Open the service URL to use the RAG Chatbot
            
            ## Examples
            
            - https://github.com/langchain-ai/langchain
            - https://github.com/NVIDIA/TensorRT-LLM
            - https://github.com/labmlai/annotated_deep_learning_paper_implementations
            """
        )
        
        with gr.Row():
            github_url = gr.Textbox(
                label="GitHub URL",
                placeholder="Enter GitHub repository URL (e.g., https://github.com/owner/repo)",
            )
            create_button = gr.Button("Create RAG Service", variant="primary")
            
        with gr.Row():
            status = gr.Textbox(
                label="Status", 
                value="Ready to create service", 
                interactive=False
            )
            
        with gr.Row():
            message = gr.Textbox(
                label="Message",
                value="Enter a GitHub URL and click 'Create RAG Service'",
                interactive=False,
                lines=2
            )
            
        with gr.Row():
            service_url = gr.Textbox(
                label="Service URL",
                value="",
                interactive=False,
            )
            open_button = gr.Button("Open Service")
            
        with gr.Row():
            model_def_path = gr.Textbox(
                label="Model Definition Path",
                value="",
                interactive=False,
            )
            
        # Button click event
        create_button.click(
            create_rag_service,
            inputs=[github_url],
            outputs=[status, message, service_url, model_def_path],
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
            inputs=[service_url],
            outputs=[message],
        )
        
        # Example buttons
        with gr.Row():
            gr.Markdown("### Example Repositories")
            
        with gr.Row():
            example_1 = gr.Button("LangChain")
            example_2 = gr.Button("TensorRT-LLM")
            example_3 = gr.Button("Annotated Deep Learning")
            
        example_1.click(
            lambda: "https://github.com/langchain-ai/langchain",
            outputs=[github_url],
        )
        example_2.click(
            lambda: "https://github.com/NVIDIA/TensorRT-LLM",
            outputs=[github_url],
        )
        example_3.click(
            lambda: "https://github.com/labmlai/annotated_deep_learning_paper_implementations",
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