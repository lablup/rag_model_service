#!/usr/bin/env python3

"""
CLI for vectorizing and searching documents

Usage:
    # Process documents
    python cli_vectorizer.py process /home/work/RAGModelService/TensorRT-LLM/docs/source

    # Search in vector store
    python cli_vectorizer.py search "Login"
"""


import asyncio
from pathlib import Path
from typing import List

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from vectordb_manager import VectorDBManager

app = typer.Typer()
console = Console()

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)


@app.command()
def process(
    docs_path: Path = typer.Argument(
        ...,
        help="Path to docs directory containing markdown files",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    indices_path: Path = typer.Option(
        "./embedding_indices",
        "--indices-path",
        "-i",
        help="Path to store FAISS index",
    ),
):
    """Process markdown documents and create vector index"""

    async def run():
        try:
            manager = VectorDBManager(docs_path, indices_path)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Collect documents
                progress.add_task(description="Collecting documents...", total=None)
                documents = await manager.collect_documents()

                # Create index
                progress.add_task(description="Creating index...", total=None)
                await manager.create_indices(documents)

            console.print("\n[green]✓[/green] Document processing completed!")

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)

    asyncio.run(run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    indices_path: Path = typer.Option(
        "./embedding_indices", "--indices-path", "-i", help="Path to FAISS index"
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Maximum number of results"
    ),
):
    """Search in vector index"""

    async def run():
        try:
            manager = VectorDBManager(Path("."), indices_path)
            await manager.load_index()

            results = await manager.search_documents(query, limit)

            if not results:
                console.print("\n[yellow]No results found[/yellow]")
                return

            console.print("\n[bold]Search Results:[/bold]")
            console.print(f"Found {len(results)} results:")

            for i, result in enumerate(results, 1):
                console.print(f"\n{i}. Score: {result['similarity_score']:.4f}")
                console.print(f"File: {result['metadata']['relative_path']}")
                console.print(f"Content preview: {result['content'][:200]}...")

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise typer.Exit(code=1)

    asyncio.run(run())


if __name__ == "__main__":
    app()
