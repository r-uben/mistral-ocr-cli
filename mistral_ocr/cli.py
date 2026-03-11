"""Command-line interface for Mistral OCR."""

import os
import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .config import Config
from .processor import OCRProcessor

console = Console()

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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.version_option(version=__version__, prog_name="mistral-ocr")
def main(
    input_path: Path,
    output_path: Path | None,
    api_key: str | None,
    model: str,
    env_file: Path | None,
    include_images: bool,
    save_originals: bool,
    add_timestamp: bool,
    table_format: str | None,
    extract_headers: bool,
    extract_footers: bool,
    reprocess: bool,
    verbose: bool,
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

        # Print header
        console.print("\n[bold blue]🔍 Mistral OCR[/bold blue]")
        console.print("[dim]Powered by Mistral AI's OCR API[/dim]\n")

        # Load configuration
        if verbose:
            console.print("[dim]Loading configuration...[/dim]")

        # If API key is provided via CLI, set it before loading config
        # (must happen before load_dotenv, which won't override existing vars)
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

        # Create config from environment
        config = Config.from_env(env_file)

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
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
