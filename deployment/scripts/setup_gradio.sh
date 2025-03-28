#!/bin/bash
# Setup script for the RAG Model Service Gradio interface
# This script installs dependencies and sets up environment variables

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Install the package in development mode
pip install -e .

# Install Gradio dependency
pip install gradio==5.23.1

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Add environment variables to bashrc for persistence
echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
echo "export BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH}" >> ~/.bashrc
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    cat ${BACKEND_MODEL_PATH}/RAGModelService/.env >> ~/.bashrc
fi
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc

echo "RAG Model Service setup completed successfully"
