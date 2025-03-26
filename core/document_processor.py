#!/usr/bin/env python3
"""
Document Processor for RAG Service

This module provides functionality for:
1. Reading files from various sources
2. Creating langchain Document objects
3. Chunking documents for efficient processing
4. Processing files for RAG applications
5. Handling GitHub repository cloning and processing
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import aiofiles
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from pydantic import BaseModel, Field
from git import Repo

class DocumentMetadata(BaseModel):
    """Metadata for processed documents"""

    source_path: str
    relative_path: Optional[str] = None
    filename: str
    last_updated: datetime
    file_size: int
    file_type: str
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentProcessor:
    """
    Handles document processing operations for RAG applications.
    
    This class provides functionality for:
    1. Reading files from disk
    2. Creating langchain Document objects
    3. Chunking documents for efficient processing
    4. Processing files for RAG applications
    5. Cloning and processing GitHub repositories
    """

    def __init__(
        self,
        docs_root: Optional[Union[str, Path]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        supported_extensions: List[str] = None,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        """
        Initialize the DocumentProcessor.
        
        Args:
            docs_root: Root directory for documents
            chunk_size: Size of chunks for document splitting
            chunk_overlap: Overlap between chunks
            supported_extensions: List of supported file extensions (without dots)
            logger: Logger instance
        """
        self.docs_root = Path(docs_root) if docs_root else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions or ["md", "txt", "rst", "html"]
        self.logger = logger or structlog.get_logger().bind(component="DocumentProcessor")
        
        # Initialize text splitters
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size // 4,  # Approximate token count
            chunk_overlap=self.chunk_overlap // 4
        )

    async def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Read file content asynchronously.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string or None if file cannot be read
        """
        file_path = Path(file_path)
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()
                self.logger.debug("File read successfully", path=str(file_path))
                return content
        except Exception as e:
            self.logger.error("File read error", path=str(file_path), error=str(e))
            return None

    def create_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Create metadata for a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentMetadata object
        """
        stats = file_path.stat()
        
        metadata = DocumentMetadata(
            source_path=str(file_path.absolute()),
            filename=file_path.name,
            last_updated=datetime.fromtimestamp(stats.st_mtime),
            file_size=stats.st_size,
            file_type=file_path.suffix.lstrip(".").lower(),
        )
        
        # Add relative path if docs_root is set
        if self.docs_root and file_path.is_relative_to(self.docs_root):
            metadata.relative_path = str(file_path.relative_to(self.docs_root))
            
        return metadata

    def get_text_splitter(self, file_type: str) -> Any:
        """
        Get appropriate text splitter based on file type.
        
        Args:
            file_type: Type of file (extension without dot)
            
        Returns:
            Text splitter instance
        """
        if file_type.lower() == "md":
            return self.markdown_splitter
        else:
            return self.text_splitter

    def chunk_document(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        file_type: str = "txt"
    ) -> List[Document]:
        """
        Split document into chunks.
        
        Args:
            content: Document content
            metadata: Document metadata
            file_type: Type of file
            
        Returns:
            List of Document objects
        """
        if not content:
            return []
            
        splitter = self.get_text_splitter(file_type)
        chunks = splitter.split_text(content)
        
        documents = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Create a copy of metadata to avoid modifying the original
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = total_chunks
            
            documents.append(
                Document(page_content=chunk, metadata=chunk_metadata)
            )
            
        self.logger.debug(
            "Document chunked", 
            chunk_count=len(documents), 
            filename=metadata.get("filename", "unknown")
        )
        
        return documents

    async def process_file(
        self, 
        file_path: Union[str, Path],
        chunk: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a single file into Document objects.
        
        Args:
            file_path: Path to the file
            chunk: Whether to chunk the document
            custom_metadata: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error("File not found", path=str(file_path))
            return []
            
        content = await self.read_file(file_path)
        if content is None:
            return []
            
        metadata = self.create_metadata(file_path)
        
        # Add custom metadata if provided
        if custom_metadata:
            metadata.custom_metadata.update(custom_metadata)
            
        # Convert metadata to dict for Document creation
        metadata_dict = metadata.model_dump()
        
        if chunk:
            return self.chunk_document(
                content, 
                metadata_dict, 
                file_type=metadata.file_type
            )
        else:
            return [Document(page_content=content, metadata=metadata_dict)]

    async def collect_documents(
        self,
        directory: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        chunk: bool = True,
        file_filter: Optional[Callable[[Path], bool]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Collect all documents from a directory.
        
        Args:
            directory: Directory to search for documents (defaults to docs_root)
            recursive: Whether to search recursively
            chunk: Whether to chunk the documents
            file_filter: Optional function to filter files
            custom_metadata: Additional metadata to include for all documents
            
        Returns:
            List of Document objects
        """
        search_dir = Path(directory) if directory else self.docs_root
        
        if not search_dir:
            self.logger.error("No directory specified and no docs_root set")
            return []
            
        if not search_dir.exists():
            self.logger.warning("Directory not found", path=str(search_dir))
            return []
            
        # Default file filter based on supported extensions
        if file_filter is None:
            file_filter = lambda p: p.is_file() and p.suffix.lstrip(".").lower() in self.supported_extensions
            
        # Find all matching files
        if recursive:
            files = [p for p in search_dir.rglob("*") if file_filter(p)]
        else:
            files = [p for p in search_dir.iterdir() if file_filter(p)]
            
        self.logger.info("Found files", count=len(files), directory=str(search_dir))
        print(f"Found {len(files)} files in {search_dir}")
        
        # Process all files
        tasks = [
            self.process_file(file_path, chunk=chunk, custom_metadata=custom_metadata)
            for file_path in files
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten the list of lists
        documents = [doc for sublist in results for doc in sublist]
        
        self.logger.info(
            "Collected documents", 
            file_count=len(files), 
            document_count=len(documents)
        )
        print(f"Collected {len(documents)} documents from {len(files)} files")
        
        return documents

    async def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
        file_type: str = "txt"
    ) -> List[Document]:
        """
        Process text content into Document objects.
        
        Args:
            text: Text content
            metadata: Metadata for the document
            chunk: Whether to chunk the document
            file_type: Type of file for choosing appropriate splitter
            
        Returns:
            List of Document objects
        """
        if not text:
            return []
            
        # Create default metadata if not provided
        if metadata is None:
            metadata = {
                "source_path": "text_input",
                "filename": "text_input.txt",
                "last_updated": datetime.now(),
                "file_size": len(text),
                "file_type": file_type,
            }
            
        if chunk:
            return self.chunk_document(text, metadata, file_type)
        else:
            return [Document(page_content=text, metadata=metadata)]

    def filter_documents(
        self,
        documents: List[Document],
        filter_func: Callable[[Document], bool]
    ) -> List[Document]:
        """
        Filter documents based on a filter function.
        
        Args:
            documents: List of documents to filter
            filter_func: Function that returns True for documents to keep
            
        Returns:
            Filtered list of documents
        """
        filtered = [doc for doc in documents if filter_func(doc)]
        
        self.logger.debug(
            "Filtered documents", 
            original_count=len(documents), 
            filtered_count=len(filtered)
        )
        
        return filtered

    def merge_documents(
        self,
        documents: List[Document],
        separator: str = "\n\n",
        max_tokens: Optional[int] = None
    ) -> Document:
        """
        Merge multiple documents into a single document.
        
        Args:
            documents: List of documents to merge
            separator: Separator to use between documents
            max_tokens: Maximum number of tokens in the merged document
            
        Returns:
            Merged document
        """
        if not documents:
            return Document(page_content="", metadata={})
            
        # Start with the first document's metadata
        merged_metadata = documents[0].metadata.copy()
        merged_metadata["merged_count"] = len(documents)
        merged_metadata["sources"] = [
            doc.metadata.get("source_path", "unknown") 
            for doc in documents
        ]
        
        # Merge content
        content = separator.join(doc.page_content for doc in documents)
        
        # Truncate if max_tokens is specified
        if max_tokens:
            # Approximate token count using token splitter
            if len(content) > max_tokens * 4:  # Rough estimate of 4 chars per token
                chunks = self.token_splitter.split_text(content)
                content = separator.join(chunks[:max_tokens // self.token_splitter.chunk_size])
                merged_metadata["truncated"] = True
                
        return Document(page_content=content, metadata=merged_metadata)

            
    async def process_documentation(self, docs_path: Path, vector_store_manager: Any) -> List[Document]:
        """
        Process documentation to create vector indices.
        
        Args:
            docs_path: Path to documentation directory
            vector_store_manager: Vector store manager instance
            
        Returns:
            List of processed documents
        """
        try:
            print(f"Processing documentation in {docs_path}...")
            
            # Set docs_root to the provided path
            self.docs_root = docs_path
            
            # Collect documents
            print("Collecting documents...")
            documents = await self.collect_documents()
            
            if not documents:
                self.logger.warning("No documents found", docs_path=str(docs_path))
                print(f"Warning: No documents found in {docs_path}")
                return []
            
            print(f"Found {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            self.logger.error("Error processing documentation", error=str(e))
            print(f"Error processing documentation: {str(e)}")
            raise