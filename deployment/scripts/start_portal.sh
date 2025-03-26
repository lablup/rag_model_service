#!/bin/bash

cd /models/RAGModelService

source /models/RAGModelService/.env

python3 auto_rag_service/rag_service_portal.py
