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