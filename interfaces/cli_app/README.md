# Process documents
python vectorstore_cli.py process --docs-path ./docs --indices-path ./indices

# Search in vector store
python vectorstore_cli.py search "How to configure the model?" --indices-path ./indices

# List available indices
python vectorstore_cli.py list-indices --indices-path ./indices

# Evaluate search performance
python vectorstore_cli.py evaluate --indices-path ./indices --queries-file ./queries.txt

