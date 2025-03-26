import argparse
import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

import structlog
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from vectordb_manager.vectordb_manager import VectorDBManager

logger = structlog.get_logger()


class ChatResponse(BaseModel):
    """Model for chat response"""

    role: str
    content: str


class LLMConfig(BaseModel):
    """Configuration for LLM"""

    openai_api_key: str
    model_name: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.2
    streaming: bool = True
    memory_k: int = 25
    max_results: int = 5  # Reduced from 20 to limit context size
    max_tokens_per_doc: int = 8000  # New: limit tokens per document
    filter_model: str = "gpt-4o"  # Model for document filtering
    base_url: str = ""  # Base URL for custom model endpoints


class RAGManager:
    """Manager for RAG-enhanced chatbot using LangChain"""

    def __init__(self, config: LLMConfig, vector_store: VectorDBManager):
        """Initialize RAG Manager"""
        self.logger = logger.bind(component="RAGManager")
        self.config = config
        self.vector_store = vector_store

        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            streaming=config.streaming,
            base_url=config.base_url if config.base_url else None,
            max_tokens=4096,
            timeout=120,  # Increased timeout to 5 minutes
            max_retries=3,  # Increased retries
        )

        # Initialize memory as a list of messages
        self.messages = []
        self.memory_k = config.memory_k

        system_prompt = """
        You are a helpful AI Assistant with document search and retrieval capabilities. Answer questions based on the provided context.
        Provide the detailed explanation.
        The provided context is a list of documents from a vector store knowledge base.
        The similarity score for each document is also provided as Euclidean distance where the lower the number the more similar.
        If the context doesn't contain relevant information, use your general knowledge but mention this fact. Keep answers focused and relevant to the query.
        If there is no context provided and you don't know, then answer "I don't know".
        """
        self.system_prompt = SystemMessage(content=system_prompt)

        # Create RAG-specific prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                self.system_prompt,
                ("placeholder", "{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Question: {input}"),
            ]
        )

        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def _get_relevant_context(self, query: str) -> tuple[str, List[Dict]]:
        """
        Retrieve and filter relevant context from vector store

        Args:
            query: User's question

        Returns:
            Tuple of (formatted context string, filtered results)
        """
        try:
            results = await self.vector_store.search_documents(
                query=query, k=self.config.max_results
            )

            filtered_results = results

            context_parts = []
            
            for doc in filtered_results:
                # Get metadata fields safely with defaults
                metadata = doc.get("metadata", {})
                relative_path = metadata.get("relative_path", "unknown_path")
                similarity_score = doc.get("similarity_score", 0.0)
                content = doc.get("content", "")

                # Truncate content to limit tokens
                if (
                    len(content) > self.config.max_tokens_per_doc * 4
                ):  # Rough estimate of 4 chars per token
                    content = (
                        content[: self.config.max_tokens_per_doc * 4] + "..."
                    )

                context_parts.append(
                    f"[Source: {relative_path} "
                    f"(Similarity: {similarity_score:.2f})]\n"
                    f"{content}\n"
                )

            if not context_parts:
                self.logger.warning("No documents passed filtering", query=query)
                return "No relevant context found after filtering.", []

            return "\n".join(context_parts), filtered_results

        except Exception as e:
            self.logger.error(
                "Error retrieving/filtering context", error=str(e), query=query
            )
            return "Error retrieving context.", []

    async def generate_response(
        self, user_input: str
    ) -> AsyncGenerator[str, None]:
        """Generate RAG-enhanced streaming response"""
        try:
            # Retrieve and filter relevant context
            context, filtered_results = await self._get_relevant_context(user_input)
            context_msg = HumanMessage(content=f"<context>\n{context}\n</context>\n")

            # Get chat history
            history = self.get_chat_history()
            
            async for chunk in self.chain.astream(
                {
                    "input": user_input,
                    "context": [context_msg],
                    "chat_history": history,
                }
            ):
                yield chunk

            # Update memory
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=chunk))

        except Exception as e:
            self.logger.error(
                "Error generating response",
                error=str(e),
                user_input=user_input,
            )
            yield f"Error: {str(e)}"

    async def generate_response_with_context(
        self, user_input: str, context: str
    ) -> AsyncGenerator[str, None]:
        """Generate response using provided context"""
        try:
            context_msg = HumanMessage(content=f"<context>\n{context}\n</context>\n")
            history = self.get_chat_history()

            async for chunk in self.chain.astream(
                {
                    "input": user_input,
                    "context": [context_msg],
                    "chat_history": history,
                }
            ):
                yield chunk

            # Update memory
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=chunk))

        except Exception as e:
            self.logger.error(
                "Error generating response with context",
                error=str(e),
                user_input=user_input,
            )
            yield f"Error: {str(e)}"

    def get_chat_history(self) -> List[Union[HumanMessage, AIMessage]]:
        """Get the chat history for the context window"""
        # Limit to the last k exchanges
        if len(self.messages) > self.memory_k * 2:
            return self.messages[-self.memory_k * 2 :]
        return self.messages


async def interactive_mode(rag_manager: RAGManager, verbose: bool = False) -> None:
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
            async for chunk in rag_manager.generate_response(user_input):
                # Print chunk without newline to simulate streaming
                print(chunk, end="", flush=True)
                response_text += chunk
            print()  # Add a newline at the end
            
            if verbose:
                print("\n--- Debug Info ---")
                print(f"Model: {rag_manager.config.model_name}")
                print(f"Temperature: {rag_manager.config.temperature}")
                print(f"Max Results: {rag_manager.config.max_results}")
                print(f"Messages in History: {len(rag_manager.messages)}")
                print("------------------")
        except Exception as e:
            print(f"\nError: {str(e)}")


async def process_single_query(rag_manager: RAGManager, query: str, show_context: bool = False) -> None:
    """Process a single query and exit."""
    print(f"Query: {query}")
    
    if show_context:
        context, _ = await rag_manager._get_relevant_context(query)
        print("\n----- Retrieved Context -----")
        print(context)
        print("----------------------------\n")
    
    print("\nResponse:")
    
    # Collect response chunks
    response_text = ""
    try:
        async for chunk in rag_manager.generate_response(query):
            # Print chunk without newline to simulate streaming
            print(chunk, end="", flush=True)
            response_text += chunk
        print()  # Add a newline at the end
    except Exception as e:
        print(f"\nError: {str(e)}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Query documents using a RAG-enhanced LLM."
    )
    
    # Path arguments
    parser.add_argument(
        "--docs-path", 
        type=str,
        help="Path to documentation directory"
    )
    parser.add_argument(
        "--indices-path", 
        type=str,
        help="Path to vector indices"
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--query", 
        type=str,
        metavar="QUERY",
        help="Single query mode: Process one query and exit"
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Start in interactive chat mode (default if no mode is specified)"
    )
    
    # LLM configuration
    parser.add_argument(
        "--model", 
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the LLM (default: 0.2)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for the LLM response (default: 2048)"
    )
    
    # Additional options
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context before response (only in query mode)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL for custom model endpoints"
    )
    
    return parser.parse_args()


async def main() -> int:
    """Main function to run the RAG chatbot."""
    # Load environment variables from .env file
    load_dotenv()

    # Check for required environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or in your environment.")
        return 1
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    docs_root = Path(args.docs_path) if args.docs_path else base_dir / "docs"
    indices_path = Path(args.indices_path) if args.indices_path else base_dir / "embedding_indices"

    if args.verbose:
        print(f"Initializing with docs path: {docs_root}")
        print(f"Vector indices path: {indices_path}")

    # Initialize VectorDBManager
    vector_manager = VectorDBManager(docs_root, indices_path)
    
    # Load the vector index
    if args.verbose:
        print("Loading vector index...")
    
    try:
        await vector_manager.load_index()
        if not vector_manager.index:
            print("Error: Failed to load index. Please check that the index exists.")
            return 1
    except Exception as e:
        print(f"Error loading index: {e}")
        return 1
    
    # Set up LLM config from arguments and environment variables
    llm_config = LLMConfig(
        openai_api_key=openai_api_key,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_results=args.max_results,
        streaming=True,
        base_url=args.base_url
    )
    
    # Initialize RAG manager
    rag_manager = RAGManager(llm_config, vector_manager)
    
    # Execute the requested mode
    try:
        if args.query:
            await process_single_query(rag_manager, args.query, args.show_context)
        else:
            # Default to interactive mode
            await interactive_mode(rag_manager, args.verbose)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)