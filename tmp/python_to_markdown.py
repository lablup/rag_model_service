#!/usr/bin/env python3
"""
Python to Markdown Converter

This script traverses directories to find Python files and merges them into a single Markdown file.
Each Python file is represented as a code block in the Markdown file with its path as a heading.

python scripts/python_to_markdown.py --root /Users/sergeyleksikov/Documents/GitHub/RAGModelService --output rag_context.md --relative --exclude "*venv*" "*__pycache__*" "*docs*"

"""

import os
import argparse
from pathlib import Path
import fnmatch


def collect_python_files(root_dir, exclude_patterns=None):
    """
    Recursively collect all Python files from the given directory.
    
    Args:
        root_dir (str): Root directory to start the search from
        exclude_patterns (list): List of glob patterns to exclude
        
    Returns:
        list: List of paths to Python files
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs_to_remove = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if any(fnmatch.fnmatch(dir_path, pattern) for pattern in exclude_patterns):
                dirs_to_remove.append(d)
        
        for d in dirs_to_remove:
            dirs.remove(d)
            
        # Collect Python files
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                    python_files.append(file_path)
    
    return python_files


def merge_to_markdown(python_files, output_file, relative_to=None):
    """
    Merge Python files into a single Markdown file.
    
    Args:
        python_files (list): List of paths to Python files
        output_file (str): Path to the output Markdown file
        relative_to (str): Directory to make paths relative to
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Python Code Collection\n\n")
        md_file.write("This document contains merged Python code files for LLM context.\n\n")
        
        for file_path in sorted(python_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as py_file:
                    content = py_file.read()
                
                # Make path relative if specified
                display_path = file_path
                if relative_to:
                    try:
                        display_path = os.path.relpath(file_path, relative_to)
                    except ValueError:
                        # Keep absolute path if relpath fails
                        pass
                
                md_file.write(f"## {display_path}\n\n")
                md_file.write("```python\n")
                md_file.write(content)
                if not content.endswith('\n'):
                    md_file.write('\n')
                md_file.write("```\n\n")
                
                print(f"Added {display_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert Python files to a single Markdown file')
    parser.add_argument('--root', '-r', type=str, required=True, 
                        help='Root directory to search for Python files')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Output Markdown file path')
    parser.add_argument('--exclude', '-e', type=str, nargs='+', default=[],
                        help='Glob patterns to exclude (e.g. "*venv*" "*/tests/*")')
    parser.add_argument('--relative', action='store_true',
                        help='Make file paths relative to the root directory')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root)
    output_file = args.output
    exclude_patterns = args.exclude
    
    print(f"Searching for Python files in: {root_dir}")
    print(f"Excluding patterns: {exclude_patterns}")
    
    python_files = collect_python_files(root_dir, exclude_patterns)
    print(f"Found {len(python_files)} Python files")
    
    relative_to = root_dir if args.relative else None
    merge_to_markdown(python_files, output_file, relative_to)
    
    print(f"Successfully merged Python files into: {output_file}")


if __name__ == "__main__":
    main()
