
#!/usr/bin/env python3
"""
Vector Store CLI Tool

This CLI tool provides a command-line interface for:
1. Processing documents and creating vector indices
2. Searching in vector indices
3. Testing and evaluating vector store functionality

Usage:
    # Process documents
    python vectorstore_cli.py process --docs-path ./docs --indices-path ./indices
    
    # Search in vector store
    python vectorstore_cli.py search "How to configure the model?" --indices-path ./indices
    
    # List available indices
    python vectorstore_cli.py list-indices --indices-path ./indices
    
    # Evaluate search performance
    python vectorstore_cli.py evaluate --indices-path ./indices --queries-file ./queries.txt
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data.vector_store import VectorStore
from core.document_processor import DocumentProcessor

# Initialize logger and console
logger = structlog.get_logger()
console = Console()

# Create Typer app
app = typer.Typer(
    help="Vector Store CLI Tool for processing and searching documents",
    add_completion=False,
)


@app.command("process")
def process_documents(
    docs_path: Path = typer.Option(
        ...,
        "--docs-path",
        "-d",
        help="Path to documentation directory",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    indices_path: Path = typer.Option(
        "./embedding_indices",
        "--indices-path",
        "-i",
        help="Path to store vector indices",
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        "-c",
        help="Size of chunks for document splitting",
    ),
    chunk_overlap: int = typer.Option(
        200,
        "--chunk-overlap",
        "-o",
        help="Overlap between chunks",
    ),
    file_pattern: str = typer.Option(
        "*.md",
        "--file-pattern",
        "-p",
        help="File pattern to match (e.g., '*.md')",
    ),
):
    """Process documents and create vector indices"""
    console.print(Panel.fit("Processing Documents", style="bold blue"))
    
    async def run():
        try:
            # Initialize vector store
            vector_store = VectorStore(docs_path, indices_path)
            
            # Initialize document processor with custom chunk settings
            doc_processor = DocumentProcessor(
                docs_root=docs_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Process documentation
                task = progress.add_task(description="Processing documentation...", total=None)
                
                # Define a file filter based on the pattern
                extension = file_pattern.replace("*.", "")
                file_filter = lambda p: p.is_file() and p.suffix.lstrip(".").lower() == extension
                
                # Collect documents using DocumentProcessor
                documents = await doc_processor.collect_documents(
                    directory=docs_path,
                    recursive=True,
                    chunk=True,
                    file_filter=file_filter
                )
                
                if not documents:
                    console.print("[yellow]No documents found to process[/yellow]")
                    return
                
                progress.update(task, description=f"Creating vector indices for {len(documents)} documents...")
                
                # Create vector indices
                await vector_store.create_indices(documents)
                
                progress.update(task, description=f"Processed {len(documents)} documents")
            
            # Display summary
            console.print("\n[green]✓[/green] Document processing completed!")
            console.print(f"Documents processed: [bold]{len(documents)}[/bold]")
            console.print(f"Vector indices saved to: [bold]{indices_path}[/bold]")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


@app.command("search")
def search_documents(
    query: str = typer.Argument(..., help="Search query"),
    indices_path: Path = typer.Option(
        "./embedding_indices",
        "--indices-path",
        "-i",
        help="Path to vector indices",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum number of results to return",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (optional)",
    ),
):
    """Search for documents in vector indices"""
    console.print(Panel.fit(f"Searching: {query}", style="bold blue"))
    
    async def run():
        try:
            # Initialize vector store
            vector_store = VectorStore(
                docs_path or Path("."), 
                indices_path
            )
            
            # Load index
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(description="Loading vector index...", total=None)
                await vector_store.load_index()
                
                if not vector_store.index:
                    console.print("[red]Error:[/red] Failed to load vector index")
                    return
                
                progress.update(task, description="Searching documents...")
                results = await vector_store.search_documents(query, limit)
                
            # Display results
            if not results:
                console.print("\n[yellow]No results found[/yellow]")
                return
            
            console.print(f"\nFound [bold]{len(results)}[/bold] results:")
            
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                similarity = result["similarity_score"]
                
                console.print(f"\n[bold]{i}.[/bold] [Score: {similarity:.4f}]")
                console.print(f"[bold]Source:[/bold] {metadata.get('relative_path', 'unknown')}")
                
                # Show content preview (first 300 chars)
                content = result["content"]
                preview = content[:300] + ("..." if len(content) > 300 else "")
                console.print(f"[bold]Preview:[/bold] {preview}")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


@app.command("list-indices")
def list_indices(
    indices_path: Path = typer.Option(
        "./embedding_indices",
        "--indices-path",
        "-i",
        help="Path to vector indices",
    ),
):
    """List available vector indices"""
    console.print(Panel.fit("Available Vector Indices", style="bold blue"))
    
    try:
        # Check if indices directory exists
        if not indices_path.exists():
            console.print(f"[yellow]Indices directory not found:[/yellow] {indices_path}")
            return
        
        # Get all subdirectories (potential indices)
        indices = [d for d in indices_path.iterdir() if d.is_dir()]
        
        if not indices:
            console.print("[yellow]No vector indices found[/yellow]")
            return
        
        # Create table
        table = Table(title="Vector Indices")
        table.add_column("Index Name", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Size", style="magenta")
        
        for index in indices:
            # Calculate size
            size_bytes = sum(f.stat().st_size for f in index.glob('**/*') if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            
            table.add_row(
                index.name,
                str(index),
                f"{size_mb:.2f} MB"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate_search(
    indices_path: Path = typer.Option(
        "./embedding_indices",
        "--indices-path",
        "-i",
        help="Path to vector indices",
    ),
    queries_file: Path = typer.Option(
        ...,
        "--queries-file",
        "-q",
        help="Path to file containing evaluation queries (one per line)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum number of results per query",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (optional)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path to save evaluation results (JSON format)",
    ),
):
    """Evaluate search performance using a set of queries"""
    console.print(Panel.fit("Evaluating Search Performance", style="bold blue"))
    
    async def run():
        try:
            # Read queries
            with open(queries_file, "r") as f:
                queries = [line.strip() for line in f if line.strip()]
            
            if not queries:
                console.print("[yellow]No queries found in the file[/yellow]")
                return
            
            console.print(f"Loaded [bold]{len(queries)}[/bold] evaluation queries")
            
            # Initialize vector store
            vector_store = VectorStore(
                docs_path or Path("."), 
                indices_path
            )
            
            # Load index
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(description="Loading vector index...", total=None)
                await vector_store.load_index()
                
                if not vector_store.index:
                    console.print("[red]Error:[/red] Failed to load vector index")
                    return
                
                progress.update(task, description="Running evaluation queries...", total=len(queries))
                
                results = []
                total_time = 0
                
                # Process each query
                for i, query in enumerate(queries):
                    start_time = time.time()
                    search_results = await vector_store.search_documents(query, limit)
                    query_time = time.time() - start_time
                    total_time += query_time
                    
                    # Store results
                    results.append({
                        "query": query,
                        "time_seconds": query_time,
                        "result_count": len(search_results),
                        "results": search_results,
                    })
                    
                    progress.update(task, advance=1)
            
            # Calculate statistics
            avg_time = total_time / len(queries) if queries else 0
            avg_results = sum(r["result_count"] for r in results) / len(results) if results else 0
            
            # Display summary
            console.print("\n[bold]Evaluation Summary:[/bold]")
            console.print(f"Total queries: [bold]{len(queries)}[/bold]")
            console.print(f"Average query time: [bold]{avg_time:.4f}[/bold] seconds")
            console.print(f"Average results per query: [bold]{avg_results:.2f}[/bold]")
            
            # Save results if requested
            if output_file:
                output_data = {
                    "summary": {
                        "total_queries": len(queries),
                        "total_time": total_time,
                        "average_time": avg_time,
                        "average_results": avg_results,
                    },
                    "queries": results,
                }
                
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                console.print(f"\nResults saved to: [bold]{output_file}[/bold]")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


if __name__ == "__main__":
    app()