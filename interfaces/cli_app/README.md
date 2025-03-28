# RAG Model Service CLI Tools

This document provides a guide to the command-line tools available in the RAG Model Service. The tools are listed in order of complexity, from simplest to most complex.

## Testing Order

### 1. `github_cli.py`
**Description**: Simplest functionality that clones a GitHub repository.  
**Dependencies**: Minimal with simple path handling.

**Example**:
```bash
python interfaces/cli_app/github_cli.py clone https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs
```

### 2. `vectorstore_cli.py`
**Description**: Basic vector store operations (create index, search).  
**Dependencies**: Requires existing indices but doesn't launch any services.

**Examples**:
```bash
# List available indices
python interfaces/cli_app/vectorstore_cli.py list-indices --indices-path ./indices

# Search in vector store
python interfaces/cli_app/vectorstore_cli.py search "How to configure the model?" --indices-path ./indices
```

### 3. `rag_cli.py`
**Description**: Interactive CLI for the RAG system.  
**Dependencies**: Requires vector indices but has a simple terminal interface.

**Example**:
```bash
python interfaces/cli_app/rag_cli.py --indices-path ./indices
```

### 4. `launch_gradio.py`
**Description**: Launches a Gradio web interface for an existing RAG service.  
**Dependencies**: Requires vector indices and documentation, launches a web server.

**Example**:
```bash
python interfaces/cli_app/launch_gradio.py --indices-path ./indices --docs-path ./docs
```

### 5. `create_rag_service.py`
**Description**: End-to-end script that clones a repo, creates indices, and launches a web interface.  
**Dependencies**: Combines functionality of github_cli.py, vectorstore_cli.py, and launch_gradio.py.

**Example**:
```bash
python interfaces/cli_app/create_rag_service.py --github-url https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs
```
```bash
python interfaces/cli_app/create_rag_service.py --service-id NVIDIA_TensorRT-LLM
```

### 6. `rag_launcher.py`
**Description**: Most complex tool that sets up example repositories and launches the portal interface.  
**Dependencies**: Manages multiple services and has the most dependencies.

**Example**:
```bash
python interfaces/cli_app/rag_launcher.py --portal-only
```

## Testing Strategy

This order allows you to test the individual components before testing the integrated scripts that combine multiple functionalities. Each step builds on the previous one, helping you isolate any issues that might arise.

## Additional Commands

```bash
# Process documents
python interfaces/cli_app/vectorstore_cli.py process --docs-path ./docs --indices-path ./indices
# Evaluate search performance
python interfaces/cli_app/vectorstore_cli.py evaluate --indices-path ./indices --queries-file ./queries.txt
```