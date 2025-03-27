#!/bin/bash

cd /models/RAGModelService

pip install -e .
pip install gradio==5.23.1

source /models/RAGModelService/.env

echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
cat /models/RAGModelService/.env >> ~/.bashrc
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc
