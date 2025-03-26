"""Command-line interface for RAG functionality."""
import asyncio
import argparse
import os
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

from rag_model_service.config.loader import load_config
from rag_model_service.config.models import LLMSettings, RetrievalSettings
from rag_model_service.core.retrieval import RetrievalEngine
from rag_model_service.core.llm import LLMInterface
from rag_model_service.core.rag_engine import RAGEngine
from rag_model_service.data.vector_store import VectorStore

logger = structlog.get_logger()

async def interactive_mode(rag_engine: RAGEngine, verbose: bool = False) -> None:
    """Run the interactive chatbot interface."""
    print("\n----- RAG Chatbot Test Interface -----")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Command line chat loop
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        # Process user query and get AI response
        print("\nThinking...")
        
        # Collect response chunks
        response_text = ""
        try:
            async for chunk in rag_engine.process_query(user_input):
                # Print chunk without newline to simulate streaming
                print(chunk, end="", flush=True)
                response_text += chunk
            print()  # Add a newline at the end
            
            if verbose:
                print("\n--- Debug Info ---")
                print(f"Model: {rag_engine.llm_interface.settings.model_name}")
                print(f"Temperature: {rag_engine.llm_interface.settings.temperature}")
                print("------------------")
        except Exception as e:
            print(f"\nError: {str(e)}")

# Main function and CLI parsing logic...

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)