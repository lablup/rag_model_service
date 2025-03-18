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


pip install -e .


echo "Setup complete!"

