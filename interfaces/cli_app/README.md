# GitHub CLI

python github_cli.py clone https://github.com/owner/repo --output-dir ./output

# Process documents
python vectorstore_cli.py process --docs-path ./docs --indices-path ./indices

# Search in vector store
python vectorstore_cli.py search "How to configure the model?" --indices-path ./indices

# List available indices
python vectorstore_cli.py list-indices --indices-path ./indices

# Evaluate search performance
python vectorstore_cli.py evaluate --indices-path ./indices --queries-file ./queries.txt

# RAG CLI
python interfaces/cli_app/rag_cli.py --docs-path ./your_docs_path
