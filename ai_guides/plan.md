# Plan for Modifying RAG Model Service Document Path Handling

## Current Issue
The RAG Model Service currently uses a hardcoded document path from the configuration file (`config.py`), which gets the `DOCS_PATH` from the `.env` file with a default value of `/models/RAGModelService/rag_services`. However, we need to modify the system so that `app/gradio_app.py` receives the document path as an argument from `start.sh` through `launch_gradio.py`.

## Current Flow
1. `start.sh` calls `launch_gradio.py` with arguments including `--docs-path ./${RAG_SERVICE_PATH}/"docs"`
2. `launch_gradio.py` parses these arguments and creates a configuration dictionary
3. `launch_gradio.py` calls `create_gradio_interface(rag_manager)` from `app/gradio_app.py`
4. `app/gradio_app.py` loads its configuration from `config.py`, which uses the hardcoded path from `.env`

## Proposed Changes

### 1. Modify `app/gradio_app.py`
- Update the `create_gradio_interface` function to accept an optional `docs_path` parameter
- Modify the `read_markdown_file` function to use this custom path when available
- Update the initialization in the `main` function to use the custom path
- **Add fallback mechanism**: If no `docs_path` is provided, use the default from `config.py`

### 2. Modify `launch_gradio.py`
- Ensure the `docs_path` from command line arguments is passed to `create_gradio_interface`
- Pass the path to the `RAGManager` initialization
- Ensure the configuration dictionary includes the custom docs_path

### 3. Verify `start.sh`
- Ensure it correctly passes the document path as `--docs-path ./${RAG_SERVICE_PATH}/"docs"`

## Implementation Steps
1. Modify `app/gradio_app.py` to accept and use a custom document path with fallback to default
   - Update `create_gradio_interface` to accept `docs_path=None` parameter
   - In the function, use `docs_path if docs_path else config.paths.docs_root`
2. Update `launch_gradio.py` to pass the document path to `create_gradio_interface`
   - Modify the call to `create_gradio_interface` to include `docs_path=config['paths']['docs_path']`
3. Test the changes to ensure the correct document path is being used
   - Test with explicit path provided
   - Test with no path provided (should use default)

## Testing Plan
1. Run the service with `start.sh` providing a custom docs path
2. Run the service without specifying a docs path to verify fallback to default
3. Verify that the correct document path is being used by checking the logs
4. Test the RAG functionality to ensure documents are being retrieved correctly in both scenarios
