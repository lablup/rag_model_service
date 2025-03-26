import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
import os
from dotenv import load_dotenv


logger = structlog.get_logger()


class DocumentMetadata(BaseModel):
    """Metadata for processed documents"""

    relative_path: str
    filename: str
    last_updated: datetime
    file_size: int


class VectorDBManager:
    def __init__(self, docs_root: Path, indices_path: Path):
        self.docs_root = Path(docs_root)
        self.indices_path = Path(indices_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.logger = logger.bind(component="VectorDBManager")
        self.index: Optional[FAISS] = None
        self.index_name = "vectorstore"

    async def read_markdown_file(self, file_path: Path) -> Optional[str]:
        """Read markdown file content"""
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            self.logger.error("File read error", path=str(file_path), error=str(e))
            return None

    def create_metadata(self, file_path: Path) -> DocumentMetadata:
        """Create metadata for a document"""
        stats = file_path.stat()
        return DocumentMetadata(
            relative_path=str(file_path.relative_to(self.docs_root)),
            filename=file_path.name,
            last_updated=datetime.fromtimestamp(stats.st_mtime),
            file_size=stats.st_size,
        )

    async def collect_documents(self) -> List[Document]:
        """Collect all documents from the documentation directory"""
        docs = []

        if not self.docs_root.exists():
            self.logger.warning("Documentation directory not found", path=str(self.docs_root))
            return docs

        for file_path in self.docs_root.rglob("*.md"):
            content = await self.read_markdown_file(file_path)
            if content:
                metadata = self.create_metadata(file_path)
                docs.append(
                    Document(page_content=content, metadata=metadata.model_dump())
                )

        self.logger.info("Collected documents", count=len(docs))
        return docs

    async def create_indices(self, documents: List[Document]) -> None:
        """Create a single FAISS index from collected documents"""
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
        if not index_path.exists():
            self.logger.warning("Index directory not found")
            return

        try:
            self.index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("Loaded index")
        except Exception as e:
            self.logger.error(
                "Failed to load index", error=str(e)
            )

    async def search_documents(
        self, query: str, k: int = 5
    ) -> List[Dict]:
        """Search documents in the index"""
        if not self.index:
            await self.load_index()
            
        if not self.index:
            raise ValueError("No index loaded")

        try:
            docs_with_scores = self.index.similarity_search_with_score(query, k=k)
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

    async def test_search(self, query: str, k: int = 5) -> List[Document]:
        """Return documents for a given search query."""
        if self.index is None:
            await self.load_index()
            if self.index is None:
                raise ValueError("No loaded index available.")
        
        return self.index.similarity_search(query, k=k)


async def interactive_mode(vector_manager: VectorDBManager, docs_root: Path, indices_path: Path) -> None:
    """Run the interactive CLI menu for the VectorDBManager."""
    while True:
        print("\n----- VectorDBManager Test Menu -----")
        print("1. Collect and print documents")
        print("2. Create vector index")
        print("3. Load existing vector index")
        print("4. Search documents")
        print("5. Change documentation path")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            try:
                print("\nCollecting documents...")
                documents = await vector_manager.collect_documents()
                print(f"Collected {len(documents)} documents.")
                if documents:
                    print("\nSample documents:")
                    for i, doc in enumerate(documents[:3]):  # Show first 3 docs as samples
                        print(f"\nDocument {i+1}:")
                        print(f"Content (first 100 chars): {doc.page_content[:100]}...")
                        print(f"Metadata: {doc.metadata}")
            except Exception as e:
                print(f"Error collecting documents: {str(e)}")
        
        elif choice == "2":
            try:
                print("\nCreating vector index...")
                documents = await vector_manager.collect_documents()
                
                if not documents:
                    print("No documents found to index.")
                    continue
                
                print(f"Creating index from {len(documents)} documents...")
                await vector_manager.create_indices(documents)
                print("Vector index created successfully.")
            except Exception as e:
                print(f"Error creating index: {str(e)}")
        
        elif choice == "3":
            try:
                print("\nLoading vector index...")
                await vector_manager.load_index()
                if vector_manager.index:
                    print("Vector index loaded successfully.")
                else:
                    print("No vector index found to load.")
            except Exception as e:
                print(f"Error loading index: {str(e)}")
        
        elif choice == "4":
            if not vector_manager.index:
                print("No index loaded. Please load or create an index first.")
                continue
            
            query = input("\nEnter your search query: ")
            if not query.strip():
                continue
            
            try:
                print(f"\nSearching for: '{query}'")
                results = await vector_manager.test_search(query)
                
                print(f"\nFound {len(results)} results.")
                for i, doc in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"Content (first 100 chars): {doc.page_content[:100]}...")
                    print(f"Metadata: {doc.metadata}")
                    if hasattr(doc, 'distance') and doc.distance is not None:
                        print(f"Relevance score: {1 - doc.distance:.4f}")
            except Exception as e:
                print(f"Error during search: {str(e)}")
                
        elif choice == "5":
            print(f"\nCurrent documentation path: {docs_root}")
            new_path = input("Enter new documentation path (or press Enter to keep current): ").strip()
            
            if new_path:
                try:
                    new_path = Path(new_path)
                    if not new_path.exists():
                        print(f"Warning: Path {new_path} does not exist. Creating a new VectorDBManager anyway.")
                    
                    # Create a new VectorDBManager with the updated path
                    docs_root = new_path
                    vector_manager = VectorDBManager(docs_root, indices_path)
                    print(f"Documentation path updated to: {docs_root}")
                except Exception as e:
                    print(f"Error updating path: {str(e)}")
        
        elif choice == "6":
            print("Exiting. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


async def collect_and_print_documents(vector_manager: VectorDBManager, verbose: bool = False) -> None:
    """Collect and print document information."""
    try:
        print("\nCollecting documents...")
        documents = await vector_manager.collect_documents()
        print(f"Collected {len(documents)} documents.")
        
        if documents and verbose:
            print("\nSample documents:")
            for i, doc in enumerate(documents[:5]):  # Show first 5 docs as samples
                print(f"\nDocument {i+1}:")
                print(f"Content (first 100 chars): {doc.page_content[:100]}...")
                print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error collecting documents: {str(e)}")
        raise


async def create_index(vector_manager: VectorDBManager) -> None:
    """Create vector index from documents."""
    try:
        print("\nCreating vector index...")
        documents = await vector_manager.collect_documents()
        
        if not documents:
            print("No documents found to index.")
            return
        
        print(f"Creating index from {len(documents)} documents...")
        await vector_manager.create_indices(documents)
        print("Vector index created successfully.")
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        raise


async def load_index(vector_manager: VectorDBManager) -> None:
    """Load existing vector index."""
    try:
        print("\nLoading vector index...")
        await vector_manager.load_index()
        if vector_manager.index:
            print("Vector index loaded successfully.")
        else:
            print("No vector index found to load.")
            raise ValueError("No vector index found to load.")
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        raise


async def search_documents(vector_manager: VectorDBManager, query: str, k: int = 5) -> None:
    """Search documents in the index."""
    try:
        if not vector_manager.index:
            await load_index(vector_manager)
        
        print(f"\nSearching for: '{query}'")
        results = await vector_manager.test_search(query, k=k)
        
        print(f"\nFound {len(results)} results.")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content (first 100 chars): {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
            if hasattr(doc, 'distance') and doc.distance is not None:
                print(f"Relevance score: {1 - doc.distance:.4f}")
    except Exception as e:
        print(f"Error during search: {str(e)}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VectorDBManager - Create and search vector indices from documentation."
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
        help="Path to store vector indices"
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--collect", 
        action="store_true", 
        help="Collect and print documents"
    )
    action_group.add_argument(
        "--create-index", 
        action="store_true", 
        help="Create vector index from documents"
    )
    action_group.add_argument(
        "--load-index", 
        action="store_true", 
        help="Load existing vector index"
    )
    action_group.add_argument(
        "--search", 
        type=str, 
        metavar="QUERY", 
        help="Search documents with the provided query"
    )
    action_group.add_argument(
        "--interactive", 
        action="store_true", 
        help="Start in interactive mode"
    )
    
    # Additional options
    parser.add_argument(
        "-k", 
        "--top-k", 
        type=int, 
        default=5, 
        help="Number of results to return for search (default: 5)"
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show more detailed output"
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main function to run the VectorDBManager CLI."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or in your environment.")
        return
    
    # Set environment variable for OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    docs_root = Path(args.docs_path) if args.docs_path else base_dir / "docs"
    indices_path = Path(args.indices_path) if args.indices_path else base_dir / "embedding_indices"
    
    print(f"Initializing with docs path: {docs_root}")
    print(f"Vector indices path: {indices_path}")
    
    # Initialize VectorDBManager
    vector_manager = VectorDBManager(docs_root, indices_path)
    
    # Execute the requested action
    try:
        if args.collect:
            await collect_and_print_documents(vector_manager, args.verbose)
        elif args.create_index:
            await create_index(vector_manager)
        elif args.load_index:
            await load_index(vector_manager)
        elif args.search:
            await search_documents(vector_manager, args.search, args.top_k)
        else:
            # Default to interactive mode if no specific action is requested
            await interactive_mode(vector_manager, docs_root, indices_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)