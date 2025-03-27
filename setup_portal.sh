#!/bin/bash

cd /models/RAGModelService

pip install -e .

# Run backend.ai-client installation in background
pip install backend.ai-client==25.4.0 &


source /models/RAGModelService/.env

echo "" >> ~/.bashrc
echo "# Environment variables from RAGModelService" >> ~/.bashrc
cat /models/RAGModelService/.env >> ~/.bashrc
echo "# End of RAGModelService environment variables" >> ~/.bashrc
source ~/.bashrc
