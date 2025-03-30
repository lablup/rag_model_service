#!/bin/bash
# Start script for the RAG Model Service Gradio interface
# This script starts the RAG service with the specified configuration

# Get the base path from environment variable or use default
BACKEND_MODEL_PATH=${BACKEND_MODEL_PATH:-/models}
RAG_SERVICE_PATH=${RAG_SERVICE_PATH:-rag_services}

# Source environment variables
if [ -f ${BACKEND_MODEL_PATH}/RAGModelService/.env ]; then
    source ${BACKEND_MODEL_PATH}/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Using default configuration."
fi

# Set default values if not provided in environment
SERVICE_ID=${SERVICE_ID:-default}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Change to the RAG Model Service directory
cd ${BACKEND_MODEL_PATH}/RAGModelService

# Construct paths for docs and indices
DOCS_PATH=${DOCS_PATH:-${BACKEND_MODEL_PATH}/${RAG_SERVICE_PATH}/${SERVICE_ID}/docs}
INDICES_PATH=${INDICES_PATH:-${BACKEND_MODEL_PATH}/${RAG_SERVICE_PATH}/${SERVICE_ID}/indices}

echo "Starting RAG Model Service with:"
echo "  - Docs path: ${DOCS_PATH}"
echo "  - Indices path: ${INDICES_PATH}"
echo "  - Host: ${HOST}"
echo "  - Port: ${PORT}"
echo "  - Service ID: ${SERVICE_ID}"

# Launch the RAG service using launch_gradio.py
python ${BACKEND_MODEL_PATH}/RAGModelService/interfaces/cli_app/launch_gradio.py \
    --docs-path "${DOCS_PATH}" \
    --indices-path "${INDICES_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --service-id "${SERVICE_ID}"

# Keep the container running
tail -f /dev/null
