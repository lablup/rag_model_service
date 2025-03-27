#!/bin/bash

cd /models/RAGModelService

pip install -e .
pip install gradio==5.23.1

nohup pip install backend.ai-client==25.4.0 > backend_install.log 2>&1 &

echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
cat /models/RAGModelService/.env >> ~/.bashrc
echo "# End of RAGModelService environment variables" >> ~/.bashrc
