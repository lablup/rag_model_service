"""Command-line interface for RAG functionality."""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

from core.retrieval import RetrievalEngine
from core.llm import LLMInterface
from core.rag_engine import RAGEngine
from data.vector_store import VectorStore
from config.config import load_config, LLMConfig, RetrievalSettings, PathConfig

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
    # Load configuration first
    config = load_config()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Command-line Interface")
    parser.add_argument("--docs-path", type=str, help="Path to documentation directory")
    parser.add_argument("--indices-path", type=str, help="Path to vector indices directory")
    parser.add_argument("--model", type=str, help=f"OpenAI model to use (default: {config.llm.model_name})")
    parser.add_argument("--temperature", type=float, help=f"Temperature for text generation (default: {config.llm.temperature})")
    parser.add_argument("--max-results", type=int, help=f"Maximum number of search results to use (default: {config.rag.max_results})")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--service-id", type=str, help="Service ID for service-specific paths")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Set up paths using PathConfig
        path_config = config.paths
        
        # Update service_id if provided
        if args.service_id:
            path_config.service_id = args.service_id
        
        # Resolve paths using config if not explicitly provided
        if args.docs_path:
            docs_path = Path(args.docs_path)
            print(f"Using provided docs path: {docs_path}")
        else:
            docs_path = path_config.get_service_docs_path(path_config.service_id)
            print(f"Using docs path from config: {docs_path}")
        
        if args.indices_path:
            indices_path = Path(args.indices_path)
            print(f"Using provided indices path: {indices_path}")
        else:
            indices_path = path_config.get_service_indices_path(path_config.service_id)
            print(f"Using indices path from config: {indices_path}")
        
        # Ensure paths exist
        if not docs_path.exists():
            print(f"Warning: Documentation directory {docs_path} does not exist.")
        
        # Create indices directory if it doesn't exist
        indices_path.mkdir(exist_ok=True, parents=True)
        
        # Set up LLM settings
        llm_settings = LLMConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY", config.llm.openai_api_key),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", config.llm.openai_base_url),
            model_name=args.model or config.llm.model_name,
            temperature=args.temperature if args.temperature is not None else config.llm.temperature,
            streaming=True,
        )
        
        # Set up retrieval settings
        retrieval_settings = RetrievalSettings(
            max_results=args.max_results or config.rag.max_results,
            max_tokens_per_doc=config.rag.max_tokens_per_doc,
            filter_threshold=config.rag.filter_threshold,
            docs_path=str(docs_path),
            indices_path=str(indices_path),
        )
        
        # Initialize vector store with configuration
        vector_store = VectorStore(
            docs_root=docs_path,
            indices_path=indices_path,
            llm_config=llm_settings,
            retrieval_settings=retrieval_settings,
            path_config=path_config,
            service_id=args.service_id or path_config.service_id
        )
        
        # Load vector index
        print("Loading vector index...")
        await vector_store.load_index()
        
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