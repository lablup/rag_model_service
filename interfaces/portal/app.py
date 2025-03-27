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
import traceback

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
        
        # Create model definition file path
        model_def_path = service_dir / f"model-definition-{service_id}.yml"
        
        # Use the imported function to generate the model definition
        model_definition = gen_model_def(
            github_url=github_url,
            model_name=f"RAG Service for {github_info.repo.replace('-', ' ').replace('_', ' ').title()}",
            port=8000,
            service_type='gradio'
        )
        
        # Write the model definition to file
        write_model_definition(model_definition, model_def_path)
        
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
        logger.info("Initialized service info", service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
        
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
        logger.info("Clone repository result", docs_path=str(docs_path) if docs_path else None, error=error)
        
        if error:
            service_info["status"] = ServiceStatus.ERROR
            service_info["error"] = f"Failed to clone repository: {error}"
            logger.error("Failed to clone repository", error=str(error), service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
            return service_info
            
        # Report progress
        if progress_callback:
            progress_callback(0.4, "Repository cloned successfully")
            
        # Initialize vector store
        vector_store = VectorStore(docs_path, indices_dir)
        logger.info("Initialized vector store", docs_path=str(docs_path), indices_dir=str(indices_dir))
        
        # Report progress
        if progress_callback:
            progress_callback(0.5, "Processing documentation...")
            
        # Process documents and create index
        try:
            docs = await document_processor.collect_documents(docs_path)
            logger.info("Collected documents", docs_count=len(docs) if docs else 0)
            
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
        model_def_path = generate_model_definition(github_url, service_dir)
        logger.info("Generated model definition", model_def_path=str(model_def_path) if model_def_path else None)
        
        if model_def_path:
            service_info["model_def_path"] = str(model_def_path)
        else:
            logger.warning("Failed to generate model definition")
            
        # Report progress
        if progress_callback:
            progress_callback(0.9, "Preparing service...")
            
        # Create start.sh script
        create_start_script(service_id, service_dir)
        logger.info("Created start script", service_id=service_id, service_dir=str(service_dir))
        
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
        logger.info("Starting service", service_id=service_id)
        
        if service_id not in SERVICES:
            error_msg = f"Service not found: {service_id}"
            logger.error(error_msg)
            return {
                "status": ServiceStatus.ERROR,
                "error": error_msg
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
                "error": error_msg
            }
            
        # Find available port
        port = find_available_port()
        service_info["port"] = port
        logger.info("Found available port", port=port)
        
        # Start service in background thread
        def start_service_thread():
            try:
                service_dir = Path(service_info["service_dir"])
                docs_dir = Path(service_info["docs_dir"])
                indices_dir = Path(service_info["indices_dir"])
                
                logger.info("Starting service thread", 
                           service_dir=str(service_dir),
                           docs_dir=str(docs_dir),
                           indices_dir=str(indices_dir))
                
                cmd = [
                    "python", "-m", "interfaces.cli_app.launch_gradio",
                    "--indices-path", str(indices_dir),
                    "--docs-path", str(docs_dir),
                    "--port", str(port),
                    "--host", "0.0.0.0"
                ]
                
                logger.info("Executing command", cmd=cmd, cwd=str(project_root))
                
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
                
                logger.info("Service process started", 
                           pid=process.pid, 
                           url=service_info['url'],
                           service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
                
                # Wait for process to finish
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(
                        "Service exited with error",
                        returncode=process.returncode,
                        stderr=stderr,
                        stdout=stdout
                    )
                    service_info["status"] = ServiceStatus.ERROR
                    service_info["error"] = stderr
                    logger.error("Updated service info after error", 
                               service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
                else:
                    logger.info("Service exited normally", stdout=stdout)
                    service_info["status"] = ServiceStatus.READY
                    logger.info("Updated service info after normal exit", 
                               service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
                    
            except Exception as e:
                logger.error("Error in service thread", 
                            error=str(e), 
                            traceback=traceback.format_exc())
                service_info["status"] = ServiceStatus.ERROR
                service_info["error"] = str(e)
                logger.error("Updated service info after exception", 
                           service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()})
        
        # Start thread
        thread = threading.Thread(target=start_service_thread)
        thread.daemon = True
        thread.start()
        
        # Wait for service to start
        for i in range(10):
            logger.info(f"Waiting for service to start (attempt {i+1}/10)", 
                       service_id=service_id, 
                       status=service_info["status"])
            if service_info["status"] == ServiceStatus.PROCESSING:
                break
            await asyncio.sleep(0.5)
            
        logger.info("Returning service info from start_service", 
                   service_id=service_id, 
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
            "id": service_id
        }
        logger.error("Returning error info from start_service", error_info=error_info)
        return error_info


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
            error_msg = "Invalid GitHub URL"
            logger.error(error_msg, 
                        github_url=github_url,
                        validation_result=False)
            
            # Log the raw values being returned to the status boxes for validation error
            logger.error("Raw validation error values for status boxes", 
                       status_type=type("Error").__name__,
                       status_value="Error",
                       message_type=type(error_msg).__name__,
                       message_value=error_msg,
                       service_url_type=type("").__name__,
                       service_url_value="",
                       model_def_path_type=type("").__name__,
                       model_def_path_value="")
            
            # Create validation error return tuple and log it
            error_tuple = ("Error", error_msg, "", "")
            logger.error("Validation error return tuple for Gradio", 
                       return_tuple=error_tuple,
                       tuple_type=type(error_tuple).__name__,
                       tuple_length=len(error_tuple))
            
            return error_tuple
        
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
        logger.info("Service info after processing GitHub URL", service_info=service_info)
        
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
            
        # Start service
        progress(0.95, "Starting service...")
        service_info = await start_service(service_info["id"])
        logger.info("Service info after starting service", service_info=service_info)
        
        if service_info.get("status") == ServiceStatus.ERROR:
            error_msg = f"Failed to start service: {service_info.get('error', 'Unknown error')}"
            logger.error(error_msg, 
                        service_info={k: str(v) if isinstance(v, Path) else v for k, v in service_info.items()},
                        error_details=service_info.get('error', 'Unknown error'))
            
            # Log the raw values being returned to the status boxes for service start error
            logger.error("Raw service start error values for status boxes", 
                       status_type=type("Error").__name__,
                       status_value="Error",
                       message_type=type(error_msg).__name__,
                       message_value=error_msg,
                       service_url_type=type("").__name__,
                       service_url_value="",
                       model_def_path_type=type("").__name__,
                       model_def_path_value="")
            
            # Create service start error return tuple and log it
            error_tuple = ("Error", error_msg, "", "")
            logger.error("Service start error return tuple for Gradio", 
                       return_tuple=error_tuple,
                       tuple_type=type(error_tuple).__name__,
                       tuple_length=len(error_tuple))
            
            return error_tuple
            
        # Return success only if service is actually ready
        progress(1.0, "Service started successfully!")
        model_def_path = service_info.get("model_def_path", "")
        service_id = service_info.get("id", "")
        service_url = service_info.get("url", "")
        
        # Check if service is in the correct state
        service_status = service_info.get("status", "")
        logger.info("Checking service status before returning", 
                   service_status=service_status,
                   expected_statuses=[ServiceStatus.READY, ServiceStatus.PROCESSING])
        
        # Only return Success if the service is in READY or PROCESSING state
        if service_status not in [ServiceStatus.READY, ServiceStatus.PROCESSING]:
            error_msg = f"Service in unexpected state: {service_status}"
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
                   message_type=type(f"RAG Service created and started. Service ID: {service_id}").__name__,
                   message_value=f"RAG Service created and started. Service ID: {service_id}",
                   service_url_type=type(service_url).__name__,
                   service_url_value=service_url,
                   model_def_path_type=type(model_def_path).__name__,
                   model_def_path_value=model_def_path)
        
        # Log the values being returned to the status boxes
        logger.info("Returning values to status boxes", 
                   status="Success",
                   message=f"RAG Service created and started. Service ID: {service_id}",
                   service_url=service_url,
                   model_def_path=model_def_path)
        
        # Create return tuple and log it
        return_tuple = (
            "Success", 
            f"RAG Service created and started. Service ID: {service_id}", 
            service_url, 
            model_def_path
        )
        logger.info("Final return tuple for Gradio", 
                   return_tuple=return_tuple,
                   tuple_type=type(return_tuple).__name__,
                   tuple_length=len(return_tuple))
        
        return return_tuple
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error("Error creating RAG service", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc())
        
        # Log the raw values being returned to the status boxes in error case
        logger.error("Raw error values for status boxes", 
                   status_type=type("Error").__name__,
                   status_value="Error",
                   message_type=type(error_message).__name__,
                   message_value=error_message,
                   service_url_type=type("").__name__,
                   service_url_value="",
                   model_def_path_type=type("").__name__,
                   model_def_path_value="")
        
        # Create error return tuple and log it
        error_tuple = ("Error", error_message, "", "")
        logger.error("Final error return tuple for Gradio", 
                   return_tuple=error_tuple,
                   tuple_type=type(error_tuple).__name__,
                   tuple_length=len(error_tuple))
        
        return error_tuple


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