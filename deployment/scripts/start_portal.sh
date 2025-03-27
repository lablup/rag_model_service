#!/bin/bash

cd /models/RAGModelService

source /models/RAGModelService/.env

python3 interfaces/portal/app.py
