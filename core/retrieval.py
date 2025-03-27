"""Retrieval engine for RAG functionality."""
from typing import Dict, List, Tuple, Any, Optional
import structlog

from data.vector_store import VectorStore
from data.filtering import DocumentFilter
from config.config import RetrievalSettings

class RetrievalEngine:
    """Handles document retrieval and context preparation for RAG."""
    
    def __init__(
        self,
        settings: RetrievalSettings,
        vector_store: VectorStore,
        document_filter: Optional[DocumentFilter] = None
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            settings: Configuration settings for retrieval
            vector_store: Vector database for similarity search
            document_filter: Optional filter for search results
        """
        self.logger = structlog.get_logger().bind(component="RetrievalEngine")
        self.settings = settings
        self.vector_store = vector_store
        self.document_filter = document_filter or DocumentFilter()
        
    async def get_relevant_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve and format relevant context for a query.
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (formatted context string, filtered results)
        """
        try:
            # Search for relevant documents
            results = await self.vector_store.search_documents(
                query=query, 
                k=self.settings.max_results
            )
            
            # Filter results if a document filter is provided
            if self.document_filter:
                filtered_results = await self.document_filter.filter_search_results(query, results)
            else:
                filtered_results = results
                
            # Format context from filtered results
            context_parts = []
            
            for doc in filtered_results:
                # Get metadata fields safely with defaults
                metadata = doc.get("metadata", {})
                relative_path = metadata.get("relative_path", "unknown_path")
                similarity_score = doc.get("similarity_score", 0.0)
                content = doc.get("content", "")
                
                # Truncate content to limit tokens
                if len(content) > self.settings.max_tokens_per_doc * 4:
                    content = content[:self.settings.max_tokens_per_doc * 4] + "..."
                
                # Format context entry
                context_parts.append(
                    f"[Source: {relative_path} (Similarity: {similarity_score:.2f})]\n{content}\n"
                )
            
            if not context_parts:
                self.logger.warning("No documents passed filtering", query=query)
                return "No relevant context found after filtering.", []
            
            return "\n".join(context_parts), filtered_results
            
        except Exception as e:
            self.logger.error("Error retrieving context", error=str(e), query=query)
            return f"Error retrieving context: {str(e)}", []
