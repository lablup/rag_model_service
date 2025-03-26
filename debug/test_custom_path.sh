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
