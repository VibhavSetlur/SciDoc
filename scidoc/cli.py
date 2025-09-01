"""
Simplified SciDoc CLI - Documentation and Summary Tool.

This module provides a streamlined command-line interface focused on
essential functionality: exploring directories, summarizing files,
and interactive chat assistance.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_config, validate_config
from .core import SciDoc
from .summarizer import DocumentSummarizer
from .document_generator import SciDocGenerator

# Initialize Typer app
app = typer.Typer(
    name="scidoc",
    help="Intelligent Documentation & Summary Assistant",
    add_completion=False,
)

# Rich console for pretty output
console = Console()


@app.command()
def explore(
    directory: Path = typer.Argument(
        Path.cwd(),
        help="Directory to explore and analyze"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-analysis of all files"
    ),
):
    """
    Explore and analyze a project directory with user-friendly summaries.
    
    This command scans the directory, analyzes files, and provides
    clear summaries of what's in the directory and what has changed.
    Results are stored in the .metadata/ directory for future reference.
    """
    try:
        console.print(f"[bold blue]Exploring project: {directory}[/bold blue]")
        
        # Load configuration
        if config_file:
            config = get_config(config_file.parent)
        else:
            config = get_config(directory)
        
        # Validate configuration
        if not validate_config(config):
            console.print("[red]❌ Configuration validation failed[/red]")
            raise typer.Exit(1)
        
        # Initialize SciDoc
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing SciDoc...", total=None)
            scidoc = SciDoc(config)
            progress.update(task, description="Analyzing project...")
        
        # Explore project
        metadata = scidoc.explore(directory, force=force, verbose=verbose)
        
        # Display user-friendly results
        _display_explore_results(metadata, directory)
        
        console.print(f"[green]Project analysis complete![/green]")
        console.print(f"Metadata stored in: {config.metadata_dir}")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def summarize(
    target: Path = typer.Argument(
        Path.cwd(),
        help="File or directory to summarize"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
):
    """
    ChatGPT-like summarization of files and directories.
    
    Provides intelligent summaries explaining what files do,
    what's in directories, and answers questions about content
    in a conversational manner.
    """
    try:
        from .summarizer import DocumentSummarizer
        
        # Initialize summarizer
        summarizer = DocumentSummarizer()
        
        if target.is_file():
            # Summarize a single file
            summary = summarizer.summarize_file(target)
            console.print(f"\n[bold cyan]File Summary: {target.name}[/bold cyan]")
            console.print(summary)
        else:
            # Summarize a directory
            summary = summarizer.summarize_directory(target)
            console.print(f"\n[bold cyan]Directory Summary: {target.name}[/bold cyan]")
            console.print(summary)
        
        if verbose:
            console.print(f"\n[green]Summary completed successfully[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def chat(
    directory: Path = typer.Argument(
        Path.cwd(),
        help="Directory to chat about"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    ),
):
    """
    Interactive chat assistant for project queries.
    
    Ask questions about files, directories, and project content
    in natural language. Get instant answers about what's in your project.
    """
    try:
        from .summarizer import DocumentSummarizer
        
        # Initialize summarizer
        summarizer = DocumentSummarizer()
        
        console.print(f"\n[bold blue]SciDoc Chat Assistant[/bold blue]")
        console.print(f"Chatting about: {directory}")
        console.print("Try asking questions like:")
        console.print("   • 'What files are in this directory?'")
        console.print("   • 'How many Python files are there?'")
        console.print("   • 'Find all CSV files'")
        console.print("   • 'What are the most recently modified files?'")
        console.print("   • 'Give me a summary of this project'")
        console.print("   • Type 'quit' or 'exit' to end the session\n")
        
        # Start interactive chat
        _start_interactive_chat(summarizer, directory, verbose)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def generate(
    directory: Path = typer.Argument(..., help="Directory to generate .scidoc file for"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path for .scidoc file"),
    summary_only: bool = typer.Option(False, "--summary", "-s", help="Generate summary .scidoc only")
):
    """Generate comprehensive .scidoc files with scientific analysis."""
    try:
        generator = SciDocGenerator()
        
        console.print(f"[bold blue]Generating .scidoc file for: {directory}[/bold blue]")
        
        if summary_only:
            output_path = generator.generate_summary_scidoc(directory, output)
            console.print(f"[green]Summary .scidoc generated: {output_path}[/green]")
        else:
            output_path = generator.generate_scidoc(directory, output)
            console.print(f"[green]Comprehensive .scidoc generated: {output_path}[/green]")
        
        console.print(f"\n[bold cyan]File Contents:[/bold cyan]")
        console.print(f"• Summary: Project overview and scientific assessment")
        console.print(f"• Scientific Analysis: File type distribution and insights")
        console.print(f"• Biological Insights: Sequence and variant analysis")
        console.print(f"• Statistical Summary: Data analysis results")
        console.print(f"• Data Quality: Assessment and recommendations")
        console.print(f"• Next Steps: Research recommendations")
        
        console.print(f"\n[bold green]Successfully generated .scidoc file![/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error generating .scidoc file: {e}[/red]")
        raise typer.Exit(1)


def _display_explore_results(metadata, directory: Path):
    """Display user-friendly exploration results."""
    console.print(f"\n[bold green]Project Analysis Results[/bold green]")
    
    # Project overview
    console.print(f"\nProject: {directory.name}")
    console.print(f"Total Files: {len(metadata.files)}")
    
    # File type breakdown
    file_types = {}
    for file_meta in metadata.files:
        file_type = file_meta.file_type.value
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    console.print(f"\nFile Types:")
    for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        console.print(f"   • {file_type}: {count} files")
    
    # Recent changes (if any)
    if hasattr(metadata, 'changes') and metadata.changes:
        console.print(f"\nRecent Changes:")
        for change in metadata.changes[:5]:  # Show last 5 changes
            console.print(f"   • {change.filename}: {change.change_type.value}")
    
    # Summary
    console.print(f"\nSummary: This appears to be a project with {len(metadata.files)} files.")
    if file_types:
        main_type = max(file_types.items(), key=lambda x: x[1])[0]
        console.print(f"   The project primarily contains {main_type} files.")
    
    console.print(f"\nNext Steps:")
    console.print(f"   • Use 'scidoc summarize {directory}' for a detailed summary")
    console.print(f"   • Use 'scidoc chat {directory}' to ask questions")
    console.print(f"   • Use 'scidoc explore {directory} --force' to re-analyze")


def _start_interactive_chat(summarizer, directory: Path, verbose: bool):
    """Start interactive chat session."""
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            elif query.lower() == 'help':
                console.print("[bold]💡 Available commands:[/bold]")
                console.print("  help - Show this help")
                console.print("  quit/exit/q - Exit chat")
                console.print("  summary - Get project summary")
                console.print("  files - List all files")
                console.print("  recent - Show recent changes")
                continue
            elif query.lower() == 'summary':
                summary = summarizer.summarize_directory(directory)
                console.print(f"\n[bold green]Assistant:[/bold green] {summary}")
                continue
            elif query.lower() == 'files':
                files = list(directory.rglob("*"))
                files = [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
                console.print(f"\n[bold green]Assistant:[/bold green] Found {len(files)} files:")
                for file_path in files[:10]:  # Show first 10
                    console.print(f"  • {file_path.name}")
                if len(files) > 10:
                    console.print(f"  ... and {len(files) - 10} more files")
                continue
            elif query.lower() == 'recent':
                files = list(directory.rglob("*"))
                files = [f for f in files if f.is_file() and not any(part.startswith('.') for part in f.parts)]
                file_times = [(f, f.stat().st_mtime) for f in files]
                file_times.sort(key=lambda x: x[1], reverse=True)
                
                console.print(f"\n[bold green]Assistant:[/bold green] Recent files:")
                for file_path, mtime in file_times[:5]:
                    from datetime import datetime
                    mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    console.print(f"  • {file_path.name} (modified: {mod_time})")
                continue
            
            # Process query using the summarizer
            response = summarizer.answer_question(query, directory)
            console.print(f"[bold green]Assistant:[/bold green] {response}")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    app()
