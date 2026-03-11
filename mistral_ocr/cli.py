"""Command-line interface for Mistral OCR."""

import logging
import os
import sys
from pathlib import Path

import click
from rich.logging import RichHandler

from . import __version__
from .config import Config
from .processor import OCRProcessor, console
from .utils import format_file_size, get_supported_files

# Get the original working directory if set
ORIGINAL_CWD = os.environ.get("MISTRAL_OCR_CWD", os.getcwd())


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    required=False,
    help="Path to output directory (default: <input_dir>/mistral_ocr_output/)",
)
@click.option(
    "--api-key",
    type=str,
    envvar="MISTRAL_API_KEY",
    help="Mistral API key (can also be set via MISTRAL_API_KEY env var)",
)
@click.option(
    "--model",
    type=str,
    default="mistral-ocr-latest",
    help="Mistral OCR model to use (default: mistral-ocr-latest)",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file containing configuration",
)
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Include extracted images in output (default: True)",
)
@click.option(
    "--save-originals/--no-save-originals",
    default=True,
    help="Save original input images alongside OCR results (default: True)",
)
@click.option(
    "--metadata/--no-metadata",
    "include_metadata",
    default=True,
    help="Include markdown metadata header block (default: True)",
)
@click.option(
    "--page-headings/--no-page-headings",
    "include_page_headings",
    default=True,
    help="Include markdown headings for each OCR page (default: True)",
)
@click.option(
    "--add-timestamp/--no-timestamp",
    default=False,
    help="Add timestamp to output folder name (default: False)",
)
@click.option(
    "--table-format",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default=None,
    help="Extract tables in a separate format (markdown or html). OCR 3+ only.",
)
@click.option(
    "--extract-headers/--no-extract-headers",
    default=False,
    help="Extract page headers (default: False). OCR 3+ only.",
)
@click.option(
    "--extract-footers/--no-extract-footers",
    default=False,
    help="Extract page footers (default: False). OCR 3+ only.",
)
@click.option(
    "--reprocess",
    is_flag=True,
    default=False,
    help="Reprocess files even if they already exist in metadata (default: False)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List files that would be processed without calling the API",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write logs to file (useful for batch processing)",
)
@click.version_option(version=__version__, prog_name="mistral-ocr")
def main(
    input_path: Path,
    output_path: Path | None,
    api_key: str | None,
    model: str,
    env_file: Path | None,
    include_images: bool,
    save_originals: bool,
    include_metadata: bool,
    include_page_headings: bool,
    add_timestamp: bool,
    table_format: str | None,
    extract_headers: bool,
    extract_footers: bool,
    reprocess: bool,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    log_file: Path | None,
) -> None:
    """
    Mistral OCR - Process documents using Mistral AI's OCR API.

    This tool processes PDF and image files using Mistral's powerful OCR capabilities,
    extracting text, tables, equations, and images with high accuracy.

    Examples:

        # Process a single PDF file
        mistral-ocr document.pdf

        # Process all files in a directory
        mistral-ocr ./documents --output-path ./results

        # Use a specific .env file
        mistral-ocr doc.pdf --env-file .env.production
    """
    try:
        # Resolve input path relative to original working directory
        if not input_path.is_absolute():
            input_path = Path(ORIGINAL_CWD) / input_path

        # Check if input path exists
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        # Resolve output path if provided
        if output_path and not output_path.is_absolute():
            output_path = Path(ORIGINAL_CWD) / output_path

        # Quiet mode: suppress non-error output (uses processor's shared console)
        if quiet:
            console.quiet = True

        # Configure logging — use verbose flag now, may upgrade later from config
        log_level = logging.DEBUG if verbose else logging.WARNING
        handlers: list[logging.Handler] = []
        if not quiet:
            handlers.append(RichHandler(console=console, show_time=False, show_path=False))
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            file_handler.setLevel(logging.DEBUG)
            handlers.append(file_handler)
        logging.basicConfig(level=log_level, handlers=handlers, force=True)

        # Print header
        console.print("\n[bold blue]🔍 Mistral OCR[/bold blue]")
        console.print("[dim]Powered by Mistral AI's OCR API[/dim]\n")

        # Dry-run: list files that would be processed, then exit (no API key needed)
        if dry_run:
            if input_path.is_file():
                size = format_file_size(input_path.stat().st_size)
                console.print(f"  {input_path.name}  ({size})")
                console.print("\n[dim]1 file would be processed (dry run)[/dim]")
            elif input_path.is_dir():
                files = get_supported_files(input_path)
                if not files:
                    console.print("[yellow]No supported files found.[/yellow]")
                else:
                    for f in files:
                        size = format_file_size(f.stat().st_size)
                        console.print(f"  {f.relative_to(input_path)}  ({size})")
                    console.print(f"\n[dim]{len(files)} file(s) would be processed (dry run)[/dim]")
            return

        # Load configuration (requires API key — after dry-run check)

        # If API key is provided via CLI, set it before loading config
        # (must happen before load_dotenv, which won't override existing vars)
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

        # Create config from environment
        config = Config.from_env(env_file)

        # If config has VERBOSE=true but CLI didn't pass --verbose, upgrade log level
        if config.verbose and not verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Only override config with CLI options that were explicitly passed
        ctx = click.get_current_context()
        if (
            "model" in ctx.params
            and ctx.get_parameter_source("model") != click.core.ParameterSource.DEFAULT
        ):
            config.model = model
        if (
            "include_images" in ctx.params
            and ctx.get_parameter_source("include_images") != click.core.ParameterSource.DEFAULT
        ):
            config.include_images = include_images
        if (
            "save_originals" in ctx.params
            and ctx.get_parameter_source("save_originals") != click.core.ParameterSource.DEFAULT
        ):
            config.save_original_images = save_originals
        if (
            "include_metadata" in ctx.params
            and ctx.get_parameter_source("include_metadata") != click.core.ParameterSource.DEFAULT
        ):
            config.include_metadata = include_metadata
        if (
            "include_page_headings" in ctx.params
            and ctx.get_parameter_source("include_page_headings")
            != click.core.ParameterSource.DEFAULT
        ):
            config.include_page_headings = include_page_headings
        if (
            "verbose" in ctx.params
            and ctx.get_parameter_source("verbose") != click.core.ParameterSource.DEFAULT
        ):
            config.verbose = verbose
        if (
            "table_format" in ctx.params
            and ctx.get_parameter_source("table_format") != click.core.ParameterSource.DEFAULT
        ):
            config.table_format = table_format
        if (
            "extract_headers" in ctx.params
            and ctx.get_parameter_source("extract_headers") != click.core.ParameterSource.DEFAULT
        ):
            config.extract_header = extract_headers
        if (
            "extract_footers" in ctx.params
            and ctx.get_parameter_source("extract_footers") != click.core.ParameterSource.DEFAULT
        ):
            config.extract_footer = extract_footers

        config.quiet = quiet

        # Create processor
        processor = OCRProcessor(config)

        # Process input
        processor.process(input_path, output_path, add_timestamp=add_timestamp, reprocess=reprocess)

        # Print summary
        if processor.errors:
            if verbose:
                console.print("\n[yellow]⚠ Errors encountered:[/yellow]")
                for error in processor.errors:
                    console.print(f"  [red]• {error['file']}: {error['error']}[/red]")
            console.print("\n[bold yellow]⚠ Processing complete with errors.[/bold yellow]\n")
            sys.exit(1)

        console.print("\n[bold green]✨ Processing complete![/bold green]\n")

    except ValueError as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user.[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]\n")
        logging.debug("Unexpected error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
