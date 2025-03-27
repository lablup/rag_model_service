"""RAG engine that coordinates retrieval and LLM components."""
from typing import AsyncGenerator
import structlog

from core.retrieval import RetrievalEngine
from core.llm import LLMInterface

class RAGEngine:
    """Coordinates retrieval and language model components for RAG functionality."""
    
    def __init__(self, retrieval_engine: RetrievalEngine, llm_interface: LLMInterface):
        """
        Initialize the RAG engine.
        
        Args:
            retrieval_engine: Engine for retrieving relevant context
            llm_interface: Interface for language model interactions
        """
        self.logger = structlog.get_logger().bind(component="RAGEngine")
        self.retrieval_engine = retrieval_engine
        self.llm_interface = llm_interface
    
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's question
            
        Yields:
            Response chunks
        """
        try:
            # Get relevant context
            context, _ = await self.retrieval_engine.get_relevant_context(query)
            
            # Generate response with context
            async for chunk in self.llm_interface.generate_response(query, context):
                yield chunk
                
        except Exception as e:
            self.logger.error("Error processing query", error=str(e), query=query)
            yield f"Error processing your query: {str(e)}"
