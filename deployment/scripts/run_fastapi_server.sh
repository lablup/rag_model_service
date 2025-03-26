#!/bin/bash

cd /models/RAGModelService/

source .env
export OPENAI_API_KEY=""
python3 fastapi_server.py
