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

from data.vector_store import VectorStore
from core.document_processor import DocumentProcessor
from config.config import load_config, LLMConfig, RetrievalSettings, PathConfig, ChunkingSettings

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
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (if not provided, uses config default)",
    ),
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to store vector indices (if not provided, uses config default)",
    ),
    chunk: bool = typer.Option(
        True,
        "--chunk",
        "-c",
        help="Whether to chunk the documents",
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
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """Process documents and create vector indices"""
    console.print(Panel.fit("Processing Documents", style="bold blue"))
    
    async def run():
        try:
            # Load configuration
            config = load_config()
            path_config = config.paths
            
            # Update path_config with service_id if provided
            if service_id:
                path_config.service_id = service_id
            
            # Resolve paths using config if not explicitly provided
            if docs_path is None:
                resolved_docs_path = path_config.get_service_docs_path(service_id)
                console.print(f"Using docs path from config: [bold]{resolved_docs_path}[/bold]")
            else:
                resolved_docs_path = docs_path
                console.print(f"Using provided docs path: [bold]{resolved_docs_path}[/bold]")
            
            if indices_path is None:
                resolved_indices_path = path_config.get_service_indices_path(service_id)
                console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
            else:
                resolved_indices_path = indices_path
                console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
            
            # Initialize vector store with configuration
            vector_store = VectorStore(
                docs_root=resolved_docs_path, 
                indices_path=resolved_indices_path,
                llm_config=config.llm,
                path_config=path_config,
                service_id=service_id
            )
            
            # Initialize document processor with custom chunk settings
            doc_processor = DocumentProcessor(
                docs_root=resolved_docs_path,
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
                    directory=resolved_docs_path,
                    recursive=True,
                    chunk=chunk,
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
            console.print(f"Vector indices saved to: [bold]{resolved_indices_path}[/bold]")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


@app.command("search")
def search_documents(
    query: str = typer.Argument(..., help="Search query"),
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to vector indices (if not provided, uses config default)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of results to return (if not provided, uses config default)",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (if not provided, uses config default)",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """Search for documents in vector indices"""
    console.print(Panel.fit(f"Searching: {query}", style="bold blue"))
    
    async def run():
        try:
            # Load configuration
            config = load_config()
            path_config = config.paths
            retrieval_settings = config.rag
            
            # Update path_config with service_id if provided
            if service_id:
                path_config.service_id = service_id
            
            # Resolve paths using config if not explicitly provided
            if docs_path is None:
                resolved_docs_path = path_config.get_service_docs_path(service_id)
                console.print(f"Using docs path from config: [bold]{resolved_docs_path}[/bold]")
            else:
                resolved_docs_path = docs_path
                console.print(f"Using provided docs path: [bold]{resolved_docs_path}[/bold]")
            
            if indices_path is None:
                resolved_indices_path = path_config.get_service_indices_path(service_id)
                console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
            else:
                resolved_indices_path = indices_path
                console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
            
            # Initialize vector store with configuration
            vector_store = VectorStore(
                docs_root=resolved_docs_path, 
                indices_path=resolved_indices_path,
                llm_config=config.llm,
                retrieval_settings=retrieval_settings,
                path_config=path_config,
                service_id=service_id
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
                console.print("[yellow]No matching documents found[/yellow]")
                return
                
            console.print(f"\nFound [bold]{len(results)}[/bold] matching documents:")
            
            # Create a table for results
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Score", justify="right", style="cyan", no_wrap=True)
            table.add_column("Source", style="green")
            table.add_column("Content", style="white")
            
            for i, result in enumerate(results):
                # Extract metadata
                metadata = result.get("metadata", {})
                source = metadata.get("relative_path", metadata.get("source_path", "Unknown"))
                
                # Format content (truncate if too long)
                content = result.get("content", "")
                if len(content) > 200:
                    content = content[:197] + "..."
                
                # Add row to table
                score = result.get("similarity_score", 0.0)
                table.add_row(f"{score:.4f}", source, content)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


@app.command("list-indices")
def list_indices(
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to vector indices (if not provided, uses config default)",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """List available vector indices"""
    console.print(Panel.fit("Available Vector Indices", style="bold blue"))
    
    # Load configuration
    config = load_config()
    path_config = config.paths
    
    # Update path_config with service_id if provided
    if service_id:
        path_config.service_id = service_id
    
    # Resolve indices path using config if not explicitly provided
    if indices_path is None:
        resolved_indices_path = path_config.get_service_indices_path(service_id)
        console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
    else:
        resolved_indices_path = indices_path
        console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
    
    # Check if indices directory exists
    if not resolved_indices_path.exists():
        console.print("[yellow]Indices directory does not exist[/yellow]")
        return
    
    # List all subdirectories
    indices = [d for d in resolved_indices_path.iterdir() if d.is_dir()]
    
    if not indices:
        console.print("[yellow]No vector indices found[/yellow]")
        return
    
    # Create a table for indices
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Index", style="green")
    table.add_column("Size", justify="right", style="cyan")
    table.add_column("Last Modified", style="magenta")
    
    for index_dir in indices:
        # Get directory size
        size = sum(f.stat().st_size for f in index_dir.glob('**/*') if f.is_file())
        size_str = f"{size / 1024 / 1024:.2f} MB"
        
        # Get last modified time
        mtime = index_dir.stat().st_mtime
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        
        # Add row to table
        table.add_row(index_dir.name, size_str, mtime_str)
    
    console.print(table)


@app.command("evaluate")
def evaluate_search(
    indices_path: Optional[Path] = typer.Option(
        None,
        "--indices-path",
        "-i",
        help="Path to vector indices (if not provided, uses config default)",
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
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of results per query (if not provided, uses config default)",
    ),
    docs_path: Optional[Path] = typer.Option(
        None,
        "--docs-path",
        "-d",
        help="Path to documentation directory (if not provided, uses config default)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path to save evaluation results (JSON format)",
    ),
    service_id: Optional[str] = typer.Option(
        None,
        "--service-id",
        "-s",
        help="Service ID for service-specific paths",
    ),
):
    """Evaluate search performance using a set of queries"""
    console.print(Panel.fit("Evaluating Search Performance", style="bold blue"))
    
    async def run():
        try:
            # Load configuration
            config = load_config()
            path_config = config.paths
            retrieval_settings = config.rag
            
            # Update path_config with service_id if provided
            if service_id:
                path_config.service_id = service_id
            
            # Resolve paths using config if not explicitly provided
            if docs_path is None:
                resolved_docs_path = path_config.get_service_docs_path(service_id)
                console.print(f"Using docs path from config: [bold]{resolved_docs_path}[/bold]")
            else:
                resolved_docs_path = docs_path
                console.print(f"Using provided docs path: [bold]{resolved_docs_path}[/bold]")
            
            if indices_path is None:
                resolved_indices_path = path_config.get_service_indices_path(service_id)
                console.print(f"Using indices path from config: [bold]{resolved_indices_path}[/bold]")
            else:
                resolved_indices_path = indices_path
                console.print(f"Using provided indices path: [bold]{resolved_indices_path}[/bold]")
            
            # Initialize vector store with configuration
            vector_store = VectorStore(
                docs_root=resolved_docs_path, 
                indices_path=resolved_indices_path,
                llm_config=config.llm,
                retrieval_settings=retrieval_settings,
                path_config=path_config,
                service_id=service_id
            )
            
            # Load queries from file
            with open(queries_file, "r") as f:
                queries = [line.strip() for line in f if line.strip()]
            
            console.print(f"Loaded [bold]{len(queries)}[/bold] queries from {queries_file}")
            
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
                
                # Process each query
                results = []
                task = progress.add_task(description="Evaluating queries...", total=len(queries))
                
                for query in queries:
                    start_time = time.time()
                    search_results = await vector_store.search_documents(query, limit)
                    end_time = time.time()
                    
                    query_result = {
                        "query": query,
                        "time_seconds": end_time - start_time,
                        "num_results": len(search_results),
                        "results": search_results,
                    }
                    
                    results.append(query_result)
                    progress.update(task, advance=1)
            
            # Calculate statistics
            times = [r["time_seconds"] for r in results]
            avg_time = sum(times) / len(times) if times else 0
            result_counts = [r["num_results"] for r in results]
            avg_results = sum(result_counts) / len(result_counts) if result_counts else 0
            
            # Display summary
            console.print("\n[bold]Evaluation Summary:[/bold]")
            console.print(f"Queries processed: [bold]{len(queries)}[/bold]")
            console.print(f"Average query time: [bold]{avg_time:.4f}[/bold] seconds")
            console.print(f"Average results per query: [bold]{avg_results:.2f}[/bold]")
            
            # Save results if output file is specified
            if output_file:
                output_data = {
                    "summary": {
                        "num_queries": len(queries),
                        "avg_time": avg_time,
                        "avg_results": avg_results,
                    },
                    "queries": results,
                }
                
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                
                console.print(f"\nResults saved to: [bold]{output_file}[/bold]")
            
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)
    
    asyncio.run(run())


if __name__ == "__main__":
    app()