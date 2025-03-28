#!/bin/bash
# Start script for the RAG Model Service Portal interface
# This script starts the Portal service with the specified configuration

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Load environment variables if .env file exists
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Set default values if not provided in environment
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7860}

echo "Starting RAG Model Service Portal with:"
echo "  - Host: ${HOST}"
echo "  - Port: ${PORT}"
echo "  - Base path: ${BACKEND_MODEL_PATH}"

# Start the Portal interface
python3 ${BACKEND_MODEL_PATH}/RAGModelService/interfaces/portal/app.py \
    --host "${HOST}" \
    --port "${PORT}"

# Keep the container running
tail -f /dev/null
