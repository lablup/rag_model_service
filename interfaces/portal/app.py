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
        # Load configuration
        config = load_config()
        path_config = config.paths
        
        # Create a service configuration
        service_config = ServerConfig(
            host=config.server.host,
            port=config.server.port
        )
        
        # Define service type (not in ServerConfig)
        service_type = "gradio"  # Default to gradio
        
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
        
        # Get max_results from environment variable or use a default
        max_results = os.environ.get("MAX_RESULTS", 5)
        
        logger.info("Using configuration for model definition", 
                   backend_model_path=backend_model_path,
                   rag_service_path=rag_service_path,
                   service_id=service_id,
                   port=port,
                   service_type=service_type)
        
        # Use the imported function to generate the model definition
        model_definition = gen_model_def(
            github_url=github_url,
            model_name=f"RAG Service for {github_info.repo.replace('-', ' ').replace('_', ' ').title()}",
            port=port,
            service_type=service_type,
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
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Process a GitHub URL to create a RAG service using Backend.AI.
    
    Args:
        github_url: GitHub URL for documentation
        chunk_size: Size of chunks for document splitting
        chunk_overlap: Overlap between chunks
        enable_chunking: Whether to enable document chunking
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
        model_def_path = generate_model_definition(github_url, service_dir)
        logger.info("Generated model definition", model_def_path=str(model_def_path) if model_def_path else None)
        
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


async def start_service(service_id: str) -> Dict[str, Any]:
    """
    Start a RAG service as a Backend.AI model service.
    
    Args:
        service_id: Service ID
        
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
                            regenerated_path = generate_model_definition(service_info["github_url"], service_dir)
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
    max_results = os.environ.get("MAX_RESULTS", "5")
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

        # Process GitHub URL with progress tracking
        def update_progress(progress_value, description):
            progress(progress_value, description)
            
        logger.info(
        "RAG service settings", 
            preset=chunking_preset, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            enabled=enable_chunking,
            max_results=max_results  # Log max_results
        )
        
        # Process GitHub URL with chunking parameters
        service_info = await process_github_url(
        github_url, 
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_chunking=enable_chunking,
            max_results=max_results,  # Add max_results
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
        service_info = await start_service(service_info["id"])
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
            inputs=[github_url, chunking_preset, chunk_size_slider, chunk_overlap_slider, 
                    enable_chunking, max_results],  # Add max_results
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