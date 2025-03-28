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
from typing import Dict, List, Optional, Tuple, Union

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from core.document_processor import DocumentProcessor
from config.config import LLMConfig, RetrievalSettings, PathConfig

# Initialize logger
logger = structlog.get_logger()


class VectorStore:
    """Manager for vector store operations"""

    def __init__(
        self, 
        docs_root: Union[str, Path], 
        indices_path: Union[str, Path],
        llm_config: Optional[LLMConfig] = None,
        retrieval_settings: Optional[RetrievalSettings] = None,
        path_config: Optional[PathConfig] = None,
        service_id: Optional[str] = None
    ):
        """
        Initialize VectorStore
        
        Args:
            docs_root: Path to documentation directory
            indices_path: Path to store vector indices
            llm_config: Optional LLM configuration for embeddings
            retrieval_settings: Optional retrieval settings
            path_config: Optional PathConfig instance for path resolution
            service_id: Optional service ID for service-specific paths
        """
        self.docs_root = Path(docs_root)
        self.indices_path = Path(indices_path)
        self.llm_config = llm_config
        self.retrieval_settings = retrieval_settings
        self.path_config = path_config
        self.service_id = service_id
        
        # Initialize embeddings with configuration if provided
        embedding_model = "text-embedding-3-small"  # Default model
        embedding_kwargs = {}
        
        if llm_config:
            # Use API key from config if available
            if llm_config.openai_api_key:
                embedding_kwargs["openai_api_key"] = llm_config.openai_api_key
            
            # Use base URL from config if available
            if llm_config.base_url:
                embedding_kwargs["base_url"] = llm_config.base_url
        
        self.embeddings = OpenAIEmbeddings(model=embedding_model, **embedding_kwargs)
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
        # Primary index path
        index_path = self.indices_path / self.index_name
        self.logger.info(f"Attempting to load index from: {index_path}")
        
        # If the primary path doesn't exist, try to resolve using PathConfig
        if not index_path.exists() and self.path_config:
            self.logger.info("Primary index path not found, trying to resolve using PathConfig")
            
            try:
                # Try to get service-specific indices path
                if self.service_id:
                    resolved_path = self.path_config.get_service_indices_path(self.service_id) / self.index_name
                    self.logger.info(f"Trying service-specific path: {resolved_path}")
                    if resolved_path.exists():
                        index_path = resolved_path
                        self.logger.debug(f"Found index at service-specific path: {index_path}")
                
                # If still not found, try the default indices path
                if not index_path.exists():
                    resolved_path = self.path_config.indices_path / self.index_name
                    self.logger.info(f"Trying default indices path: {resolved_path}")
                    if resolved_path.exists():
                        index_path = resolved_path
                        self.logger.debug(f"Found index at default indices path: {index_path}")
                
                # If still not found, try to resolve the path
                if not index_path.exists():
                    resolved_path = self.path_config.resolve_path(self.indices_path / self.index_name)
                    self.logger.info(f"Trying resolved path: {resolved_path}")
                    if resolved_path.exists():
                        index_path = resolved_path
                        self.logger.debug(f"Found index at resolved path: {index_path}")
            
            except Exception as e:
                self.logger.error(f"Error resolving paths: {str(e)}", exc_info=True)
        
        # If index still not found, log warning and return
        if not index_path.exists():
            self.logger.warning(f"Index not found at any path: {index_path}")
            # List all files in the indices directory to help debugging
            try:
                self.logger.debug(f"Contents of indices directory {self.indices_path}:")
                if self.indices_path.exists():
                    for file in self.indices_path.iterdir():
                        self.logger.debug(f"  - {file.name} ({file.stat().st_size} bytes)")
                else:
                    self.logger.debug(f"  Directory does not exist")
            except Exception as e:
                self.logger.error(f"Error listing indices directory: {str(e)}")
            return
        
        try:
            self.logger.info(f"Loading index from: {index_path}")
            # Check if both required files exist
            faiss_file = index_path / "index.faiss"
            pkl_file = index_path / "index.pkl"
            
            if not faiss_file.exists() or not pkl_file.exists():
                self.logger.warning(
                    "Missing index files",
                    faiss_exists=faiss_file.exists(),
                    pkl_exists=pkl_file.exists(),
                    path=str(index_path)
                )
            
            self.index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info(f"Successfully loaded index from: {index_path}")
            
            # Add debug info about the loaded index
            if hasattr(self.index, 'docstore') and hasattr(self.index.docstore, '_dict'):
                doc_count = len(self.index.docstore._dict)
                self.logger.debug(f"Loaded index contains {doc_count} documents")
                
                # Sample a few document IDs for debugging
                if doc_count > 0:
                    sample_ids = list(self.index.docstore._dict.keys())[:3]
                    self.logger.debug(f"Sample document IDs: {sample_ids}")
            
        except Exception as e:
            self.logger.error(
                "Failed to load index", error=str(e), path=str(index_path), exc_info=True
            )

    async def search_documents(
        self, query: str, k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search documents in the index
        
        Args:
            query: Search query
            k: Number of results to return (uses retrieval_settings.max_results if not provided)
            
        Returns:
            List of document dictionaries with content, metadata, and similarity score
        """
        self.logger.info(f"Searching for query: '{query}'")
        
        # Use retrieval settings for max_results if available
        if k is None and self.retrieval_settings and self.retrieval_settings.max_results:
            k = self.retrieval_settings.max_results
        elif k is None:
            k = 5  # Default value
        
        # Safe access to embeddings model name
        embeddings_model = getattr(self.embeddings, 'model_name', 'unknown')
        self.logger.debug(f"Search parameters: k={k}, embeddings_model={embeddings_model}")
        
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
            
            # Debug the first few results
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                self.logger.debug(
                    f"Search result {i+1}",
                    score=score,
                    content_preview=doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    metadata=doc.metadata
                )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                }
                for doc, score in docs_with_scores
            ]

        except Exception as e:
            self.logger.error("Search failed", error=str(e), exc_info=True)
            raise

    async def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Return documents for a given search query
        
        Args:
            query: Search query
            k: Number of results to return (uses retrieval_settings.max_results if not provided)
            
        Returns:
            List of Document objects
        """
        # Use retrieval settings for max_results if available
        if k is None and self.retrieval_settings and self.retrieval_settings.max_results:
            k = self.retrieval_settings.max_results
        elif k is None:
            k = 5  # Default value
            
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
