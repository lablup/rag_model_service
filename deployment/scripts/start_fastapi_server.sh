#!/bin/bash
# Start script for the RAG Model Service FastAPI server
# This script starts the FastAPI server with the specified configuration

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
PORT=${PORT:-8000}
SERVICE_ID=${SERVICE_ID:-default}

# Construct paths for docs and indices
DOCS_PATH=${DOCS_PATH:-${BACKEND_MODEL_PATH}/rag_services/${SERVICE_ID}/docs}
INDICES_PATH=${INDICES_PATH:-${BACKEND_MODEL_PATH}/rag_services/${SERVICE_ID}/indices}

echo "Starting RAG Model Service FastAPI server with:"
echo "  - Host: ${HOST}"
echo "  - Port: ${PORT}"
echo "  - Service ID: ${SERVICE_ID}"
echo "  - Docs path: ${DOCS_PATH}"
echo "  - Indices path: ${INDICES_PATH}"

# Start the FastAPI server
python3 ${BACKEND_MODEL_PATH}/RAGModelService/interfaces/fastapi_app/fastapi_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --docs-path "${DOCS_PATH}" \
    --indices-path "${INDICES_PATH}" \
    --service-id "${SERVICE_ID}"

# Keep the container running
tail -f /dev/null
