#!/bin/bash
# Test script for github_cli.py
# Tests the ability to clone only a specific directory from a GitHub repository

echo "Testing GitHub CLI tool with sparse checkout functionality"
echo "=========================================================="

# Test URL with a specific path (docs directory)
TEST_URL="https://github.com/vllm-project/vllm/tree/main/docs"
TEST_DIR="./test_github_cli_output"

# Clean up any existing test directory
if [ -d "$TEST_DIR" ]; then
    echo "Cleaning up existing test directory: $TEST_DIR"
    rm -rf "$TEST_DIR"
fi

# Create test directory
mkdir -p "$TEST_DIR"

# Test the parse command
echo -e "\n1. Testing 'parse' command..."
python interfaces/cli_app/github_cli.py parse "$TEST_URL"

# Test the clone command
echo -e "\n2. Testing 'clone' command with specific path..."
python interfaces/cli_app/github_cli.py clone "$TEST_URL" --output-dir "$TEST_DIR/clone_test"

# List the contents of the cloned directory
echo -e "\nContents of cloned directory:"
ls -la "$TEST_DIR/clone_test"

# Test the prepare command
echo -e "\n3. Testing 'prepare' command with specific path..."
python interfaces/cli_app/github_cli.py prepare "$TEST_URL" --output-dir "$TEST_DIR/prepare_test"

# List the contents of the prepared directory
echo -e "\nContents of prepared directory:"
ls -la "$TEST_DIR/prepare_test"

echo -e "\nTest completed!"
