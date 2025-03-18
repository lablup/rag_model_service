#!/bin/bash

cd /models/RAGModelService

pip install -e .

source /models/RAGModelService/.env

pip install backend.ai-client

