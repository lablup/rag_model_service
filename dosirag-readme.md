# DosiRAG

A powerful Retrieval-Augmented Generation (RAG) service for creating intelligent documentation assistants from GitHub repositories.
A Retrieval-Augmented Generation (RAG) service for document search and generation, providing a simple and efficient way to build a question-answering system powered by your own documentation.

Powered by Backend.AI, Backend.AI Model Service and Backend.AI CLI Client.

## Overview

DosiRAG enables you to transform documentation repositories into interactive AI assistants that provide accurate, context-aware responses. Simply point DosiRAG at a GitHub repository containing documentation, and it will create a specialized assistant with deep knowledge of your content.

## Key Components

### Portal Application (`interfaces/portal/app.py`)

The Portal is the main entry point for creating new RAG services. This web-based interface allows you to:

- Enter any GitHub URL containing documentation
- Configure document processing settings
- Deploy a new RAG service with a single click

**Process Flow:**
1. You enter a GitHub repository URL and configure settings
2. DosiRAG clones the repository and processes the documentation
3. Vector embeddings are created for semantic search capabilities
4. A Backend.AI service is automatically deployed
5. You receive a URL to access your new documentation assistant

**Configuration Options:**
- **Chunking Strategy:** Choose between Fine-grained (precise answers), Balanced (default), or Contextual (more background)
- **Number of Results:** Control how many document chunks are retrieved for each query
- **Model Settings:** Configure the base model and API endpoint
- **Service Type:** Select between Gradio UI (interactive interface) or FastAPI Server (API endpoints)

### RAG Service Interface (`interfaces/cli_app/launch_gradio.py`)

Once deployed, your RAG service provides an intuitive interface where users can:

- Ask natural language questions about the documentation
- View AI-generated responses that cite relevant sources
- Explore the retrieved documentation directly
- Use suggested questions to get started quickly

The interface combines the power of large language models with the accuracy of retrieval-based approaches, ensuring responses are both helpful and factually grounded in your documentation.

## Command Line Tools

DosiRAG includes several powerful CLI tools for advanced usage scenarios:

### `create_rag_service.py`

Create RAG services directly from the command line:

```bash
python interfaces/cli_app/create_rag_service.py --github-url https://github.com/owner/repo --service-id my_service
```

This script offers the same functionality as the Portal but in a CLI format, perfect for automation and scripting.

### `github_cli.py`

Manage GitHub repository operations:

```bash
# Parse a GitHub URL to extract components
python interfaces/cli_app/github_cli.py parse https://github.com/owner/repo

# Clone a specific directory from a repository
python interfaces/cli_app/github_cli.py clone https://github.com/owner/repo/tree/main/docs
```

### `launch_gradio.py`

Launch the Gradio interface for an existing RAG service:

```bash
python interfaces/cli_app/launch_gradio.py --indices-path ./indices --docs-path ./docs --service-id my_service
```

This script initializes the vector store, retrieval engine, and web interface for interacting with documentation.

### `rag_cli.py`

An interactive command-line interface for querying your documentation:

```bash
python interfaces/cli_app/rag_cli.py --service-id my_service
```

### `vectorstore_cli.py`

Manage vector indices for your documentation:

```bash
# Process documents and create indices
python interfaces/cli_app/vectorstore_cli.py process --docs-path ./docs

# Search in vector indices
python interfaces/cli_app/vectorstore_cli.py search "How do I install the library?" --indices-path ./indices
```

## Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- [Backend.AI](https://www.backend.ai/) credentials (for service deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lablup/dosirag.git
   cd dosirag
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Create a `.env` file with your API keys and configuration.
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o  # or another OpenAI model
TEMPERATURE=0.2
MAX_RESULTS=5
```

### Creating Your First RAG Service

1. Start the Portal application:
   ```bash
   python interfaces/portal/app.py
   ```

2. Open the provided URL in your browser.

3. Enter a GitHub URL containing documentation.

4. Configure settings (or use defaults) and click "Create RAG Service".

5. Once the service is deployed, click "Open Service" to access your documentation assistant.

## Architecture

DosiRAG follows a modular architecture with these core components:

1. **Document Processor:** Handles reading, parsing, and chunking documentation files
2. **Vector Store:** Creates and manages vector embeddings for similarity search
3. **Retrieval Engine:** Fetches relevant documents based on user queries
4. **LLM Interface:** Communicates with language models to generate responses
5. **RAG Engine:** Coordinates between retrieval and language model components
6. **User Interfaces:** Multiple interfaces for interacting with the system

## Advanced Configuration

DosiRAG can be configured through environment variables, command-line arguments, or a configuration file:

- **LLM Settings:** Model name, temperature, API key, etc.
- **Retrieval Settings:** Max results, filter threshold, etc.
- **Server Settings:** Host, port, sharing options, etc.
- **Path Settings:** Custom paths for documents, indices, etc.
- **Chunking Settings:** Chunk size, overlap, and strategy

## Backend.AI Integration

DosiRAG integrates with Backend.AI for scalable service deployment:

1. Automatically generates model definition YAML files
2. Creates Backend.AI services with appropriate resource allocation
3. Manages service lifecycle through Backend.AI's API

## Customization

DosiRAG can be customized in several ways:

- **Document Processing:** Adjust chunking strategy for your specific documentation
- **LLM Integration:** Connect to different LLM providers
- **UI Customization:** Modify the Gradio interface for your needs
- **Retrieval Options:** Fine-tune document retrieval parameters

## Troubleshooting

- **Vector Index Issues:** Check that indices have been properly created
- **Service Deployment Failures:** Verify Backend.AI credentials and permissions
- **Document Retrieval Problems:** Try different chunking strategies
- **LLM Response Issues:** Check your OpenAI API key and model access

## Contributing

Contributions are welcome! Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License.
