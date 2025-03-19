#!/bin/bash
# Start script for the RAG Model Service
# This script starts the RAG service with the specified docs path

# Source environment variables
if [ -f /models/RAGModelService/.env ]; then
    source /models/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

# Get the docs path from the first argument


# Launch the RAG service using create_rag_service.py
cd /models/RAGModelService
python /models/RAGModelService/auto_rag_service/launch_gradio.py \
    --docs-path $BACKEND_MODEL_PATH/${RAG_SERVICE_PATH}/"docs" \
    --indices-path $BACKEND_MODEL_PATH/${RAG_SERVICE_PATH}/"indices" \
    --host "0.0.0.0" \
    --port 8000

# Keep the container running
tail -f /dev/null
