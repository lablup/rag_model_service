"""Command-line interface for RAG functionality."""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.retrieval import RetrievalEngine
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from config.config import LLMConfig, RetrievalSettings

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

async def main() -> int:
    """
    Main entry point for the RAG CLI application.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Command-line Interface")
    parser.add_argument("--docs-path", type=str, help="Path to documentation directory")
    parser.add_argument("--indices-path", type=str, default="./embedding_indices", 
                       help="Path to vector indices directory")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                       help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.2, 
                       help="Temperature for text generation")
    parser.add_argument("--max-results", type=int, default=5, 
                       help="Maximum number of search results to use")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or as an environment variable.")
        return 1
    
    try:
        # Set up paths
        docs_path = Path(args.docs_path) if args.docs_path else Path("./docs")
        indices_path = Path(args.indices_path)
        
        # Ensure paths exist
        if not docs_path.exists():
            print(f"Warning: Documentation directory {docs_path} does not exist.")
        
        # Create indices directory if it doesn't exist
        indices_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize vector store
        vector_store = VectorStore(docs_path, indices_path)
        
        # Load vector index
        print("Loading vector index...")
        await vector_store.load_index()
        
        # Set up LLM settings
        llm_settings = LLMConfig(
            openai_api_key=api_key,
            model_name=args.model,
            temperature=args.temperature,
            streaming=True,
        )
        
        # Set up retrieval settings
        retrieval_settings = RetrievalSettings(
            max_results=args.max_results,
            docs_path=str(docs_path),
            indices_path=str(indices_path),
        )
        
        # Initialize components
        llm_interface = LLMInterface(llm_settings)
        retrieval_engine = RetrievalEngine(retrieval_settings, vector_store)
        rag_engine = RAGEngine(retrieval_engine, llm_interface)
        
        # Run interactive mode
        await interactive_mode(rag_engine, args.verbose)
        
        return 0
        
    except Exception as e:
        logger.error("Error in RAG CLI", error=str(e))
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)