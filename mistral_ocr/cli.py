"""Command-line interface for Mistral OCR.

Output is written through the shared ``ocr-output-contract`` package, so
mistral's output structure is byte-identical to every sibling engine::

    mistral-ocr <input> [-o DIR] [--model M] [--include-images/--no-images]
                [--table-format markdown|html] [--extract-headers] [--extract-footers]
                [--max-pages N] [-w N] [--reprocess] [--dry-run] [-q] [-v]
    mistral-ocr --version

The default output root is ``<input-parent>/ocr/``; ``-o`` overrides verbatim but
is never required. The exit code is nonzero if any file failed (uniform across
single-file and batch).
"""

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
    "--output-dir",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    required=False,
    help="Output root (default: <input-parent>/ocr/). Writes <stem>/<stem>.md per document.",
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
    default=None,
    help="Mistral OCR model to use (default: mistral-ocr-latest / $MISTRAL_MODEL)",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file containing configuration",
)
@click.option(
    "--include-images/--no-images",
    default=None,
    help="Extract embedded figures as figures/figure_<N>_page<P>.png (default: True)",
)
@click.option(
    "--table-format",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default=None,
    help="Extract tables in a separate format (markdown or html). OCR 3+ only.",
)
@click.option(
    "--extract-headers/--no-extract-headers",
    default=None,
    help="Extract page headers (default: False). OCR 3+ only.",
)
@click.option(
    "--extract-footers/--no-extract-footers",
    default=None,
    help="Extract page footers (default: False). OCR 3+ only.",
)
@click.option(
    "--max-pages",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum number of PDF pages to process (default: all pages)",
)
@click.option(
    "--workers",
    "-w",
    type=click.IntRange(min=1),
    default=None,
    help="Number of concurrent workers for batch processing (default: 1)",
)
@click.option(
    "--reprocess",
    is_flag=True,
    default=False,
    help="Reprocess files even if recorded completed with a matching checksum.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List files that would be processed without calling the API",
)
# Deprecated no-ops kept for invocation compatibility. The canonical output
# contract owns the markdown body (always clean: ## Page N, no header block, no
# frontmatter, no original copy), so these flags no longer change anything.
@click.option("--save-originals/--no-save-originals", default=None, hidden=True)
@click.option("--metadata/--no-metadata", "include_metadata", default=None, hidden=True)
@click.option("--page-headings/--no-page-headings", "page_headings", default=None, hidden=True)
@click.option("--add-timestamp/--no-timestamp", default=None, hidden=True)
@click.option(
    "--quiet", "-q", is_flag=True, help="Suppress output except file paths (for scripting)"
)
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
    output_dir: Path | None,
    api_key: str | None,
    model: str | None,
    env_file: Path | None,
    include_images: bool | None,
    table_format: str | None,
    extract_headers: bool | None,
    extract_footers: bool | None,
    max_pages: int | None,
    workers: int | None,
    reprocess: bool,
    dry_run: bool,
    save_originals: bool | None,  # deprecated no-op
    include_metadata: bool | None,  # deprecated no-op
    page_headings: bool | None,  # deprecated no-op
    add_timestamp: bool | None,  # deprecated no-op
    quiet: bool,
    verbose: bool,
    log_file: Path | None,
) -> None:
    """Mistral OCR - Process documents using Mistral AI's OCR API.

    \b
    Examples:
        mistral-ocr document.pdf
        mistral-ocr ./documents -o ./results
        mistral-ocr doc.pdf --env-file .env.production
    """
    try:
        if not input_path.is_absolute():
            input_path = Path(ORIGINAL_CWD) / input_path
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        if output_dir and not output_dir.is_absolute():
            output_dir = Path(ORIGINAL_CWD) / output_dir

        if quiet:
            console.quiet = True

        log_level = logging.DEBUG if verbose else logging.WARNING
        handlers: list[logging.Handler] = []
        if not quiet:
            handlers.append(RichHandler(console=console, show_time=False, show_path=False))
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            # Track effective verbosity rather than pinning to DEBUG, so a plain
            # --log-file does not capture httpx/SDK DEBUG bodies.
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        logging.basicConfig(level=log_level, handlers=handlers, force=True)

        if not quiet:
            console.print("\n[bold blue]Mistral OCR[/bold blue]")
            console.print("[dim]Powered by Mistral AI's OCR API[/dim]\n")

        # Dry-run: list files that would be processed (no API key needed).
        if dry_run:
            _dry_run(input_path)
            return

        # API key provided via CLI: set it before load_dotenv (won't override).
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

        config = Config.from_env(env_file)

        if config.verbose and not verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Only override config with CLI options that were explicitly passed, so
        # env-var / .env precedence is respected rather than clobbered by defaults.
        ctx = click.get_current_context()

        def _set(name: str) -> bool:
            return ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT

        if model is not None:
            config.model = model
        if include_images is not None:
            config.include_images = include_images
        if table_format is not None:
            config.table_format = table_format
        if extract_headers is not None:
            config.extract_header = extract_headers
        if extract_footers is not None:
            config.extract_footer = extract_footers
        if workers is not None:
            config.max_workers = workers
        if max_pages is not None:
            config.max_pages = max_pages
        if _set("verbose"):
            config.verbose = verbose
        config.quiet = quiet

        processor = OCRProcessor(config)
        outcome = processor.process(input_path, output_dir, reprocess=reprocess)

        if quiet:
            # Scripting contract: emit one output .md path per line on stdout.
            for path in outcome.outputs:
                click.echo(path)
        elif outcome.has_failures:
            console.print(
                f"\n[bold yellow]Processing complete with {outcome.failed} failure(s).[/bold yellow]\n"
            )
        else:
            console.print("\n[bold green]Processing complete![/bold green]\n")

        # Uniform exit policy (canon SYS-02): nonzero if any file failed.
        if outcome.exit_code != 0:
            sys.exit(outcome.exit_code)

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


def _dry_run(input_path: Path) -> None:
    """List files that would be processed without calling the API."""
    if input_path.is_file():
        size = format_file_size(input_path.stat().st_size)
        console.print(f"  {input_path.name}  ({size})")
        console.print("\n[dim]1 file would be processed (dry run)[/dim]")
    else:
        files = get_supported_files(input_path)
        if not files:
            console.print("[yellow]No supported files found.[/yellow]")
            return
        for f in files:
            size = format_file_size(f.stat().st_size)
            console.print(f"  {f.relative_to(input_path)}  ({size})")
        console.print(f"\n[dim]{len(files)} file(s) would be processed (dry run)[/dim]")


if __name__ == "__main__":
    main()
