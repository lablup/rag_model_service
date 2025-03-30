"""Document filtering for RAG functionality."""
from typing import Dict, List, Any, Optional
import structlog

class DocumentFilter:
    """Filter for search results to improve relevance."""
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the document filter.
        
        Args:
            threshold: Similarity threshold for filtering (lower is more permissive)
        """
        self.logger = structlog.get_logger().bind(component="DocumentFilter")
        self.threshold = threshold
        
    async def filter_search_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter search results based on relevance.
        
        Args:
            query: User's question
            results: List of search results
            
        Returns:
            Filtered list of results
        """
        if not results:
            return []
            
        # Get the best score (lowest value is best for Euclidean distance)
        best_score = min(result.get("similarity_score", float("inf")) for result in results)
        
        # Set a dynamic threshold based on the best score
        dynamic_threshold = best_score * (1 + self.threshold)
        
        # Filter results
        filtered_results = [
            result for result in results 
            if result.get("similarity_score", float("inf")) <= dynamic_threshold
        ]
        
        # Log filtering results
        self.logger.debug(
            "Filtered search results",
            original_count=len(results),
            filtered_count=len(filtered_results),
            best_score=best_score,
            threshold=dynamic_threshold,
        )
        
        return filtered_results
