#!/bin/bash

cd /models/RAGModelService/

source .env

python3 /models/RAGModelService/interfaces/fastapi_app/fastapi_server.py
