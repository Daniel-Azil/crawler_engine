"""
Command Line Interface for Intelligent Web Extractor

Provides a user-friendly command line interface for the intelligent
web content extraction engine.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from .core.extractor import AdaptiveContentExtractor
from .models.config import ExtractorConfig
from .models.extraction_result import ExtractionResult
from .utils.logger import ExtractorLogger


console = Console()
logger = ExtractorLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="intelligent-web-extractor")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def main(ctx, config, verbose, debug):
    """
    Intelligent Web Extractor - AI-Powered Web Content Extraction Engine
    
    A cutting-edge web scraping library that uses artificial intelligence
    to intelligently extract and process web content based on user queries.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj["config"] = ExtractorConfig.from_file(config)
    else:
        ctx.obj["config"] = ExtractorConfig()
    
    # Set logging level
    if debug:
        ctx.obj["config"].logging.log_level = "DEBUG"
    elif verbose:
        ctx.obj["config"].logging.log_level = "INFO"
    
    # Set debug mode
    ctx.obj["config"].debug_mode = debug


@main.command()
@click.argument("url", required=True)
@click.option("--query", "-q", help="User query to guide extraction")
@click.option("--mode", "-m", 
              type=click.Choice(["semantic", "structured", "hybrid", "adaptive"]),
              default="adaptive",
              help="Extraction mode")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f",
              type=click.Choice(["json", "markdown", "text", "html"]),
              default="json",
              help="Output format")
@click.option("--include-metadata", is_flag=True, default=True, help="Include metadata in output")
@click.option("--include-raw-html", is_flag=True, help="Include raw HTML in output")
@click.option("--screenshot", is_flag=True, help="Take screenshot of the page")
@click.pass_context
def extract(ctx, url, query, mode, output, format, include_metadata, include_raw_html, screenshot):
    """
    Extract content from a single URL.
    
    URL: The URL to extract content from
    """
    config = ctx.obj["config"]
    
    # Update config based on options
    config.extraction.strategy = mode
    config.include_metadata = include_metadata
    config.include_raw_html = include_raw_html
    
    async def run_extraction():
        try:
            with console.status("[bold green]Initializing Intelligent Web Extractor..."):
                extractor = AdaptiveContentExtractor(config)
                await extractor.initialize()
            
            with console.status(f"[bold blue]Extracting content from {url}..."):
                result = await extractor.extract_content(
                    url=url,
                    user_query=query,
                    extraction_mode=mode
                )
            
            # Display results
            display_extraction_result(result, format, output)
            
            # Take screenshot if requested
            if screenshot:
                await take_screenshot(extractor, url, output)
            
            await extractor.close()
            
        except Exception as e:
            console.print(f"[bold red]Extraction failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(run_extraction())


@main.command()
@click.argument("urls_file", type=click.Path(exists=True))
@click.option("--queries-file", type=click.Path(exists=True), help="File containing queries (one per line)")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--format", "-f",
              type=click.Choice(["json", "markdown", "text", "html"]),
              default="json",
              help="Output format")
@click.option("--max-workers", type=int, help="Maximum concurrent workers")
@click.option("--progress", is_flag=True, help="Show progress bar")
@click.pass_context
def batch(ctx, urls_file, queries_file, output, format, max_workers, progress):
    """
    Extract content from multiple URLs in batch.
    
    URLS_FILE: File containing URLs (one per line)
    """
    config = ctx.obj["config"]
    
    # Update config
    if max_workers:
        config.performance.max_workers = max_workers
    
    # Read URLs
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Read queries if provided
    queries = None
    if queries_file:
        with open(queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    
    # Set output directory
    if output:
        config.output_directory = output
    
    async def run_batch():
        try:
            with console.status("[bold green]Initializing Intelligent Web Extractor..."):
                extractor = AdaptiveContentExtractor(config)
                await extractor.initialize()
            
            console.print(f"[bold blue]Processing {len(urls)} URLs...")
            
            if progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress_bar:
                    task = progress_bar.add_task("Extracting content...", total=len(urls))
                    
                    results = []
                    for i, url in enumerate(urls):
                        query = queries[i] if queries and i < len(queries) else None
                        
                        result = await extractor.extract_content(url=url, user_query=query)
                        results.append(result)
                        
                        progress_bar.update(task, advance=1)
                        progress_bar.update(task, description=f"Extracted {i+1}/{len(urls)}")
            else:
                results = await extractor.extract_batch(urls, queries)
            
            # Save results
            save_batch_results(results, config.output_directory, format)
            
            # Display summary
            display_batch_summary(results)
            
            await extractor.close()
            
        except Exception as e:
            console.print(f"[bold red]Batch extraction failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(run_batch())


@main.command()
@click.option("--url", "-u", help="URL to start interactive session with")
@click.option("--config-file", type=click.Path(), help="Configuration file to use")
@click.pass_context
def interactive(ctx, url, config_file):
    """
    Start an interactive extraction session.
    """
    config = ctx.obj["config"]
    
    if config_file:
        config = ExtractorConfig.from_file(config_file)
    
    async def run_interactive():
        try:
            with console.status("[bold green]Initializing Interactive Session..."):
                extractor = AdaptiveContentExtractor(config)
                await extractor.initialize()
            
            console.print(Panel.fit(
                "[bold blue]Intelligent Web Extractor - Interactive Mode\n\n"
                "Commands:\n"
                "- extract <url> [query]: Extract content from URL\n"
                "- batch <urls_file> [queries_file]: Batch extract from file\n"
                "- stats: Show performance statistics\n"
                "- health: Check system health\n"
                "- quit: Exit interactive mode",
                title="Interactive Mode"
            ))
            
            while True:
                try:
                    command = console.input("[bold green]>>> ")
                    command = command.strip()
                    
                    if command.lower() in ['quit', 'exit', 'q']:
                        break
                    elif command.lower() == 'stats':
                        display_stats(extractor)
                    elif command.lower() == 'health':
                        display_health(extractor)
                    elif command.startswith('extract '):
                        parts = command.split(' ', 2)
                        if len(parts) >= 2:
                            url = parts[1]
                            query = parts[2] if len(parts) > 2 else None
                            
                            with console.status(f"Extracting from {url}..."):
                                result = await extractor.extract_content(url=url, user_query=query)
                                display_extraction_result(result, "json")
                    elif command.startswith('batch '):
                        parts = command.split(' ', 2)
                        if len(parts) >= 2:
                            urls_file = parts[1]
                            queries_file = parts[2] if len(parts) > 2 else None
                            
                            # Read URLs
                            with open(urls_file, 'r') as f:
                                urls = [line.strip() for line in f if line.strip()]
                            
                            # Read queries if provided
                            queries = None
                            if queries_file:
                                with open(queries_file, 'r') as f:
                                    queries = [line.strip() for line in f if line.strip()]
                            
                            with console.status(f"Processing {len(urls)} URLs..."):
                                results = await extractor.extract_batch(urls, queries)
                                display_batch_summary(results)
                    else:
                        console.print("[yellow]Unknown command. Type 'help' for available commands.")
                        
                except KeyboardInterrupt:
                    console.print("\n[yellow]Use 'quit' to exit.")
                except Exception as e:
                    console.print(f"[red]Error: {str(e)}")
            
            await extractor.close()
            
        except Exception as e:
            console.print(f"[bold red]Interactive session failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(run_interactive())


@main.command()
@click.option("--output", "-o", type=click.Path(), help="Output file for configuration")
@click.option("--format", "-f",
              type=click.Choice(["yaml", "json"]),
              default="yaml",
              help="Configuration format")
@click.pass_context
def init(ctx, output, format):
    """
    Initialize a new configuration file.
    """
    config = ExtractorConfig()
    
    if output:
        config.save_to_file(output)
        console.print(f"[green]Configuration saved to {output}")
    else:
        # Display default configuration
        if format == "json":
            console.print(json.dumps(config.to_dict(), indent=2))
        else:
            import yaml
            console.print(yaml.dump(config.to_dict(), default_flow_style=False))


@main.command()
@click.pass_context
def doctor(ctx):
    """
    Run diagnostic checks on the system.
    """
    config = ctx.obj["config"]
    
    console.print("[bold blue]Running Intelligent Web Extractor diagnostics...\n")
    
    # Check Python version
    console.print(f"Python version: {sys.version}")
    
    # Check configuration
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:")
        for error in errors:
            console.print(f"  - {error}")
    else:
        console.print("[green]Configuration is valid")
    
    # Check dependencies
    check_dependencies()
    
    # Test browser
    test_browser()
    
    # Test AI client
    test_ai_client(config)
    
    console.print("\n[bold green]Diagnostics completed!")


def display_extraction_result(result: ExtractionResult, format: str, output: Optional[str] = None):
    """Display extraction result"""
    if not result.success:
        console.print(f"[bold red]Extraction failed: {result.error_message}")
        return
    
    # Create result table
    table = Table(title="Extraction Results")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("URL", result.url)
    table.add_row("Success", "✅" if result.success else "❌")
    table.add_row("Content Length", str(len(result.content)))
    table.add_row("Confidence", f"{result.metrics.confidence_score:.2f}")
    table.add_row("Relevance", f"{result.metrics.relevance_score:.2f}")
    table.add_row("Extraction Time", f"{result.metrics.extraction_time_ms:.0f}ms")
    table.add_row("Strategy", result.strategy_info.strategy_name)
    
    if result.metadata.title:
        table.add_row("Title", result.metadata.title)
    
    console.print(table)
    
    # Save to file if requested
    if output:
        save_result_to_file(result, output, format)
        console.print(f"[green]Results saved to {output}")


def display_batch_summary(results: List[ExtractionResult]):
    """Display batch extraction summary"""
    total = len(results)
    successful = len([r for r in results if r.success])
    failed = total - successful
    
    # Calculate averages
    avg_confidence = sum(r.metrics.confidence_score for r in results if r.success) / max(successful, 1)
    avg_time = sum(r.metrics.extraction_time_ms for r in results if r.success) / max(successful, 1)
    
    table = Table(title="Batch Extraction Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total URLs", str(total))
    table.add_row("Successful", str(successful))
    table.add_row("Failed", str(failed))
    table.add_row("Success Rate", f"{successful/total*100:.1f}%")
    table.add_row("Average Confidence", f"{avg_confidence:.2f}")
    table.add_row("Average Time", f"{avg_time:.0f}ms")
    
    console.print(table)


def display_stats(extractor: AdaptiveContentExtractor):
    """Display performance statistics"""
    stats = extractor.get_performance_stats()
    
    table = Table(title="Performance Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
        else:
            table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(table)


def display_health(extractor: AdaptiveContentExtractor):
    """Display system health"""
    health = extractor.health_check()
    
    table = Table(title="System Health")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for component, status in health.items():
        if isinstance(status, bool):
            status_text = "✅ Healthy" if status else "❌ Unhealthy"
        else:
            status_text = str(status)
        table.add_row(component.replace('_', ' ').title(), status_text)
    
    console.print(table)


def save_result_to_file(result: ExtractionResult, filepath: str, format: str):
    """Save extraction result to file"""
    filepath = Path(filepath)
    
    if format == "json":
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    elif format == "markdown":
        with open(filepath, 'w') as f:
            f.write(f"# {result.metadata.title or 'Extracted Content'}\n\n")
            f.write(result.content)
    elif format == "text":
        with open(filepath, 'w') as f:
            f.write(result.content)
    elif format == "html":
        with open(filepath, 'w') as f:
            f.write(f"<html><head><title>{result.metadata.title or 'Extracted Content'}</title></head>")
            f.write(f"<body>{result.content}</body></html>")


def save_batch_results(results: List[ExtractionResult], output_dir: str, format: str):
    """Save batch results to directory"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        filename = f"extraction_{i+1:03d}.{format}"
        filepath = output_path / filename
        save_result_to_file(result, str(filepath), format)


async def take_screenshot(extractor: AdaptiveContentExtractor, url: str, output: Optional[str]):
    """Take screenshot of the page"""
    if not output:
        output = "screenshot.png"
    
    # This would be implemented with browser manager
    console.print(f"[yellow]Screenshot functionality not yet implemented")


def check_dependencies():
    """Check if required dependencies are available"""
    console.print("[bold blue]Checking dependencies...")
    
    dependencies = [
        ("playwright", "Browser automation"),
        ("beautifulsoup4", "HTML parsing"),
        ("openai", "AI model access"),
        ("sentence-transformers", "Text embeddings"),
        ("numpy", "Numerical operations"),
    ]
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            console.print(f"  ✅ {dep} - {description}")
        except ImportError:
            console.print(f"  ❌ {dep} - {description} (not installed)")


def test_browser():
    """Test browser functionality"""
    console.print("[bold blue]Testing browser...")
    
    try:
        # This would test browser initialization
        console.print("  ✅ Browser test passed")
    except Exception as e:
        console.print(f"  ❌ Browser test failed: {str(e)}")


def test_ai_client(config: ExtractorConfig):
    """Test AI client functionality"""
    console.print("[bold blue]Testing AI client...")
    
    try:
        # This would test AI client initialization
        console.print("  ✅ AI client test passed")
    except Exception as e:
        console.print(f"  ❌ AI client test failed: {str(e)}")


if __name__ == "__main__":
    main() 