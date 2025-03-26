#!/usr/bin/env python3
"""
Codebase to Markdown Converter

This script traverses directories to find code files (Python, YAML, Shell scripts, etc.) 
and merges them into a single Markdown file.
Each file is represented as a code block in the Markdown file with its path as a heading.

Example usage:
python scripts/codebase_to_markdown.py --root /Users/sergeyleksikov/Documents/GitHub/RAGModelService --output codebase_context.md --relative --exclude "*venv*" "*__pycache__*" "*bai_manager*"

"""

import os
import argparse
from pathlib import Path
import fnmatch


def get_file_extension(file_path):
    """Get the file extension for syntax highlighting in markdown."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.py':
        return 'python'
    elif ext in ['.yaml', '.yml']:
        return 'yaml'
    elif ext == '.sh':
        return 'bash'
    elif ext == '.md':
        return 'markdown'
    elif ext in ['.env', '.env.example']:
        return 'ini'
    elif ext == '.json':
        return 'json'
    elif ext == '.js':
        return 'javascript'
    elif ext == '.html':
        return 'html'
    elif ext == '.css':
        return 'css'
    else:
        return ''


def collect_code_files(root_dir, file_extensions, exclude_patterns=None):
    """
    Recursively collect code files with specified extensions from the given directory.
    
    Args:
        root_dir (str): Root directory to start the search from
        file_extensions (list): List of file extensions to include (e.g., ['.py', '.yaml'])
        exclude_patterns (list): List of glob patterns to exclude
        
    Returns:
        list: List of paths to code files
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    code_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs_to_remove = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if any(fnmatch.fnmatch(dir_path, pattern) for pattern in exclude_patterns):
                dirs_to_remove.append(d)
        
        for d in dirs_to_remove:
            dirs.remove(d)
            
        # Collect files with specified extensions
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if file has one of the specified extensions
            if file_ext in file_extensions or file.endswith('.env.example'):
                if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                    code_files.append(file_path)
    
    return code_files


def merge_to_markdown(code_files, output_file, relative_to=None):
    """
    Merge code files into a single Markdown file.
    
    Args:
        code_files (list): List of paths to code files
        output_file (str): Path to the output Markdown file
        relative_to (str): Directory to make paths relative to
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Codebase Collection\n\n")
        md_file.write("This document contains merged code files for LLM context.\n\n")
        
        for file_path in sorted(code_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as code_file:
                    content = code_file.read()
                
                # Make path relative if specified
                display_path = file_path
                if relative_to:
                    try:
                        display_path = os.path.relpath(file_path, relative_to)
                    except ValueError:
                        # Keep absolute path if relpath fails
                        pass
                
                # Get the appropriate language for syntax highlighting
                language = get_file_extension(file_path)
                
                md_file.write(f"## {display_path}\n\n")
                md_file.write(f"```{language}\n")
                md_file.write(content)
                if not content.endswith('\n'):
                    md_file.write('\n')
                md_file.write("```\n\n")
                
                print(f"Added {display_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert code files to a single Markdown file')
    parser.add_argument('--root', '-r', type=str, required=True, 
                        help='Root directory to search for code files')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='Output Markdown file path')
    parser.add_argument('--exclude', '-e', type=str, nargs='+', default=[],
                        help='Glob patterns to exclude (e.g. "*venv*" "*/tests/*")')
    parser.add_argument('--relative', action='store_true',
                        help='Make file paths relative to the root directory')
    parser.add_argument('--extensions', type=str, nargs='+', 
                        default=['.py', '.yaml', '.yml', '.sh', '.env', '.env.example'],
                        help='File extensions to include (e.g. .py .yaml .sh)')
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root)
    output_file = args.output
    exclude_patterns = args.exclude
    file_extensions = args.extensions
    
    print(f"Searching for files with extensions {file_extensions} in: {root_dir}")
    print(f"Excluding patterns: {exclude_patterns}")
    
    code_files = collect_code_files(root_dir, file_extensions, exclude_patterns)
    print(f"Found {len(code_files)} code files")
    
    relative_to = root_dir if args.relative else None
    merge_to_markdown(code_files, output_file, relative_to)
    
    print(f"Successfully merged code files into: {output_file}")


if __name__ == "__main__":
    main()
