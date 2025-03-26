#!/bin/bash
# Test script for the RAG Model Service with the default docs path

# Unset RAG_SERVICE_PATH to ensure we use the default path
unset RAG_SERVICE_PATH
export RAG_SERVICE_PATH="rag_services/303568ea"
# Source environment variables
if [ -f .env ]; then
    source .env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

echo "Using default docs path from config"

# Launch the RAG service using launch_gradio.py without specifying a docs path
# This will use the default path from config.py
python auto_rag_service/launch_gradio.py \
    --host "127.0.0.1" \
    --port 8001
