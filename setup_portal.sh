#!/bin/bash

cd /models/RAGModelService

pip install -e .

source .env

pip install backend.ai-client

