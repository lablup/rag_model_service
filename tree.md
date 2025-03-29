# RAG Model Service Project Structure

```
RAGModelService
├── ai_guides
│   ├── backendai_commands.sh
│   ├── codebase_context.md
│   ├── plan.md
│   ├── rag_context.md
│   ├── refactored_suggested_snippets.md
│   ├── refactoring_summary.md
│   ├── summary.md
│   └── todo.md
├── config
│   └── config.py
├── core
│   ├── document_processor.py
│   ├── llm.py
│   ├── rag_engine.py
│   └── retrieval.py
├── data
│   ├── document_filter.py
│   ├── filtering.py
│   └── vector_store.py
├── deployment
│   ├── scripts
│   │   ├── setup_gradio.sh
│   │   ├── setup_portal.sh
│   │   ├── start_fastapi_server.sh
│   │   ├── start_gradio.sh
│   │   └── start_portal.sh
│   └── setup
│       ├── model-definition-fastapi.yaml
│       ├── model-definition-gradio.yml
│       └── model-definition-portal.yaml
├── interfaces
│   ├── cli_app
│   │   ├── README.md
│   │   ├── create_rag_service.py
│   │   ├── generate_model_definition_cli.py
│   │   ├── github_cli.py
│   │   ├── launch_gradio.py
│   │   ├── rag_cli.py
│   │   ├── rag_launcher.py
│   │   └── vectorstore_cli.py
│   ├── fastapi_app
│   │   └── fastapi_server.py
│   ├── gradio_app
│   │   ├── gradio_app.py
│   │   └── gradio_app_original.py
│   └── portal
│       ├── app.py
│       ├── generate_model_definition.py
│       └── github.py
├── scripts
│   ├── debug
│   │   ├── curl_request_examples.txt
│   │   ├── test_backend_ai.py
│   │   ├── test_launch_gradio_custom_path.sh
│   │   ├── test_launch_gradio_default_path.sh
│   │   ├── test_model_definition.py
│   │   └── test_model_service_create.sh
│   ├── codebase_to_markdown.py
│   └── python_to_markdown.py
├── testing
│   └── functional
├── utils
│   ├── github_utils.py
│   └── service_utils.py
├── README.md
├── codebase_analysis.md
├── plan.md
├── requirements.txt
├── setup.py
└── todo.md
```

## Key Components

1. **Configuration (`config/`)**: Contains configuration settings for the system.

2. **Core Components (`core/`)**: 
   - `rag_engine.py`: Main RAG functionality coordinator
   - `llm.py`: Interface to language models
   - `retrieval.py`: Document retrieval module
   - `document_processor.py`: Document processing utilities

3. **Data Management (`data/`)**: 
   - `vector_store.py`: Manages document embeddings and retrieval
   - `document_filter.py`: Filters documents based on relevance
   - `filtering.py`: Additional filtering utilities

4. **Interfaces (`interfaces/`)**:
   - `cli_app/`: Command-line interfaces
   - `gradio_app/`: Web interface using Gradio
   - `portal/`: Service creation interface
   - `fastapi_app/`: API interface

5. **Deployment (`deployment/`)**: 
   - `scripts/`: Deployment scripts for various environments
   - `setup/`: Configuration files for deployment

6. **Utilities (`utils/`)**: Helper modules and utility functions

7. **Scripts (`scripts/`)**: Utility scripts and debugging tools

8. **Documentation & Planning**:
   - `README.md`: Project documentation
   - `codebase_analysis.md`: Analysis of the codebase
   - `plan.md`: Refactoring plan
   - `todo.md`: Detailed task list
