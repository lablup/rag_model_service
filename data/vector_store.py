#!/usr/bin/env python3
"""
Vector Store Operations for RAG Service

This module provides functionality for:
1. Creating vector indices from documents
2. Loading and searching vector indices
3. Managing vector store operations for RAG applications
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from core.document_processor import DocumentProcessor

# Initialize logger
logger = structlog.get_logger()


class VectorStore:
    """Manager for vector store operations"""

    def __init__(self, docs_root: Path, indices_path: Path):
        """
        Initialize VectorStore
        
        Args:
            docs_root: Path to documentation directory
            indices_path: Path to store vector indices
        """
        self.docs_root = Path(docs_root)
        self.indices_path = Path(indices_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.logger = logger.bind(component="VectorStore")
        self.index: Optional[FAISS] = None
        self.index_name = "vectorstore"
        self.document_processor = DocumentProcessor(docs_root=self.docs_root)

    async def collect_documents(self) -> List[Document]:
        """
        Collect all documents from the documentation directory
        
        Returns:
            List of Document objects
        """
        if not self.docs_root.exists():
            self.logger.warning("Documentation directory not found", path=str(self.docs_root))
            return []

        try:
            documents = await self.document_processor.process_directory(
                directory=self.docs_root,
                file_pattern="*.md"
            )
            self.logger.info("Collected documents", count=len(documents))
            return documents
        except Exception as e:
            self.logger.error("Error collecting documents", error=str(e))
            return []

    async def create_indices(self, documents: List[Document]) -> None:
        """
        Create a single FAISS index from collected documents
        
        Args:
            documents: List of Document objects
        """
        self.indices_path.mkdir(exist_ok=True, parents=True)

        if not documents:
            self.logger.warning("No documents to index")
            return

        try:
            # Create and save index
            index = FAISS.from_documents(documents, self.embeddings)
            index.save_local(str(self.indices_path / self.index_name))
            self.logger.info(
                "Created and saved index", doc_count=len(documents)
            )
        except Exception as e:
            self.logger.error(
                "Failed to create index", error=str(e)
            )

    async def load_index(self) -> None:
        """Load the FAISS index into memory"""
        index_path = self.indices_path / self.index_name
        self.logger.info(f"Attempting to load index from: {index_path}")
        
        # Try the primary path first
        if not index_path.exists():
            self.logger.warning(f"Index directory not found at: {index_path}")
            
            # Check if parent directories exist
            self.logger.info(f"Parent directory exists: {self.indices_path.exists()}")
            
            # List contents of parent directory if it exists
            if self.indices_path.exists():
                self.logger.info(f"Contents of {self.indices_path}: {list(self.indices_path.iterdir())}")
            
            # Try alternative paths
            alternative_paths = []
            
            # Try with RAGModelService in the path
            if "RAGModelService" not in str(index_path):
                path_parts = str(index_path).split("/")
                for i in range(len(path_parts)):
                    if path_parts[i] == "rag_services" and i > 0:
                        # Insert RAGModelService before rag_services
                        alt_path = "/".join(path_parts[:i]) + "/RAGModelService/" + "/".join(path_parts[i:])
                        alternative_paths.append(Path(alt_path))
            
            # Try without RAGModelService in the path
            if "RAGModelService" in str(index_path):
                alt_path = str(index_path).replace("/RAGModelService", "")
                alternative_paths.append(Path(alt_path))
            
            # Check alternative paths
            for alt_path in alternative_paths:
                self.logger.info(f"Trying alternative path: {alt_path}")
                if alt_path.exists():
                    self.logger.info(f"Found index at alternative path: {alt_path}")
                    index_path = alt_path
                    break
            
            # If still not found, return
            if not index_path.exists():
                self.logger.warning(f"Index not found at any path. Tried: {index_path} and {alternative_paths}")
                return

        try:
            self.logger.info(f"Loading index from: {index_path}")
            self.index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info(f"Successfully loaded index from: {index_path}")
        except Exception as e:
            self.logger.error(
                "Failed to load index", error=str(e), path=str(index_path)
            )

    async def search_documents(
        self, query: str, k: int = 5
    ) -> List[Dict]:
        """
        Search documents in the index
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document dictionaries with content, metadata, and similarity score
        """
        self.logger.info(f"Searching for query: '{query}'")
        
        if not self.index:
            self.logger.info("Index not loaded, attempting to load it now")
            await self.load_index()
            
        if not self.index:
            self.logger.error("Failed to load index for search", indices_path=str(self.indices_path))
            raise ValueError("No index loaded")

        try:
            self.logger.info(f"Performing similarity search with k={k}")
            docs_with_scores = self.index.similarity_search_with_score(query, k=k)
            self.logger.info(f"Search successful, found {len(docs_with_scores)} results")
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                }
                for doc, score in docs_with_scores
            ]

        except Exception as e:
            self.logger.error("Search failed", error=str(e))
            raise

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Return documents for a given search query
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if self.index is None:
            await self.load_index()
            if self.index is None:
                raise ValueError("No loaded index available.")
        
        return self.index.similarity_search(query, k=k)

    async def process_documentation(self, docs_path: Optional[Path] = None, indices_path: Optional[Path] = None) -> List[Document]:
        """
        Process documentation to create vector indices.
        
        Args:
            docs_path: Optional override for documentation directory
            indices_path: Optional override for indices directory
            
        Returns:
            List of collected documents
        """
        try:
            # Use provided paths or instance paths
            docs_path = docs_path or self.docs_root
            indices_path = indices_path or self.indices_path
            
            print(f"Processing documentation in {docs_path}...")
            
            # Update document processor with new docs_path if provided
            if docs_path != self.docs_root:
                self.document_processor = DocumentProcessor(docs_root=docs_path)
            
            # Collect documents
            print("Collecting documents...")
            documents = await self.collect_documents()
            
            if not documents:
                logger.warning("No documents found", docs_path=str(docs_path))
                print(f"Warning: No documents found in {docs_path}")
                return []
            
            print(f"Found {len(documents)} documents")
            
            # Create vector indices
            print("Creating vector indices...")
            await self.create_indices(documents)
            
            print(f"Vector indices created successfully in {indices_path}")
            
            return documents
            
        except Exception as e:
            logger.error("Error processing documentation", error=str(e))
            print(f"Error processing documentation: {str(e)}")
            raise
