# RAG Model Service

A Retrieval-Augmented Generation (RAG) service for document search and generation, providing a simple and efficient way to build a question-answering system powered by your own documentation.

## Features

- Vector-based document search using OpenAI embeddings and FAISS
- LLM-powered document question-answering with context retrieval
- Simple CLI interface for document indexing and testing
- Interactive web UI built with Gradio
- No project-specific code - works with any documentation structure

## Prerequisites

- Python 3.10+
- OpenAI API key

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/lablup/RAGModelService.git
cd RAGModelService

# Install the package in development mode
pip install -e .
```

### Option 2: Install dependencies directly

```bash
pip install -r requirements.txt
python-dotenv
gradio
```

## Configuration

1. Create a `.env` file in the root directory based on `.env_example`:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini  # or another OpenAI model
TEMPERATURE=0.2
MAX_RESULTS=5
```

## Usage

### 1. Document Indexing

Index your documents to create a vector store:

```bash
# Using the CLI tool with the installed package
python vectordb_manager/vectordb_manager.py --docs-path ./TensorRT-LLM --indices-path ./embedding_indices --create-index

# to Test the search
python vectordb_manager/vectordb_manager.py --search "What is AWQ?" --top-k 5
```

### 2. Terminal Chat Interface

Test the RAG system with a simple command-line interface:

```bash
# run the module directly
python -m app.rag_chatbot
```

### 3. Web Interface

Launch the Gradio web interface:

```bash
# run the module directly
python app/gradio_app.py
```

### 4. Vector DB Testing

Test the vector database functionality:


## Components

- **vectordb_manager**: Handles document collection, vectorization, and storage
- **app/rag_chatbot.py**: Implements the RAG system core functionality
- **app/gradio_app.py**: Provides a web interface using Gradio
- **app/document_filter.py**: Simple document filtering utility

## License

MIT
