#!/bin/bash

# Set paths
DOCS_PATH="/Users/lablup/Documents/GitHub/backend-ai-assistant/docs_md"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean up existing files
echo "Cleaning up existing files..."
# rm -rf "$DOCS_PATH"/*

# Convert RST to MD
echo "Converting RST files to Markdown..."
python3 "$SCRIPT_DIR/convert_docs.py"

# Vectorize the converted files
echo "Vectorizing Markdown files..."
python3 "$SCRIPT_DIR/../vectordb_manager/cli_vectorizer.py" process "$DOCS_PATH"

echo "Done! Files have been converted and vectorized."
