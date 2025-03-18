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

pip install backend.ai-client