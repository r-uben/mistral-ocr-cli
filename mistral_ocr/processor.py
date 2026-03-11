"""Core OCR processing module using Mistral AI."""

import shutil
import time
from pathlib import Path

from mistralai import Mistral
from mistralai import models as mistral_models
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .config import Config
from .utils import (
    DOCUMENT_EXTENSIONS,
    create_data_uri,
    determine_output_path,
    format_file_size,
    get_supported_files,
    load_metadata,
    make_unique_basename,
    save_base64_image,
    save_metadata,
)

console = Console()


class OCRProcessor:
    """OCR processor using Mistral AI API."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        try:
            self.client = Mistral(api_key=config.api_key)
        except (ValueError, TypeError, RuntimeError) as e:
            console.print(f"[red]Failed to initialize Mistral client: {e}[/red]")
            raise
        self.errors: list[dict] = []
        self.processed_files: list[dict] = []

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is transient and worth retrying."""
        # SDK-typed rate limit / server errors
        for exc_name in ("RateLimitError", "InternalServerError", "ServiceUnavailableError"):
            if type(error).__name__ == exc_name:
                return True
        # httpx-level HTTP status errors
        if hasattr(error, "response"):
            status = getattr(error.response, "status_code", 0)
            if status in (429, 500, 502, 503, 504):
                return True
        # Network-level transient errors
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        # Catch SDK errors that wrap HTTP status codes
        return hasattr(mistral_models, "SDKError") and isinstance(error, mistral_models.SDKError)

    def _call_with_retry(self, **ocr_kwargs: object) -> object:
        """Call ocr.process with exponential backoff on transient errors."""
        max_attempts = self.config.max_retries + 1
        base_delay = self.config.retry_base_delay

        for attempt in range(max_attempts):
            try:
                return self.client.ocr.process(**ocr_kwargs)
            except Exception as e:
                is_last = attempt == max_attempts - 1
                if is_last or not self._is_retryable(e):
                    raise
                delay = base_delay * (2**attempt)
                if self.config.verbose:
                    console.print(
                        f"[yellow]Retryable error (attempt {attempt + 1}/{max_attempts}): "
                        f"{e}. Retrying in {delay:.1f}s...[/yellow]"
                    )
                time.sleep(delay)
        # Unreachable, but keeps mypy happy
        raise RuntimeError("Retry loop exited unexpectedly")

    def process_file(self, file_path: Path) -> dict | None:
        """Process a single file with OCR."""
        try:
            # Validate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if self.config.verbose:
                console.print(f"[dim]File size: {file_size_mb:.2f} MB[/dim]")
            self.config.validate_file_size(file_path)

            # Create data URI for the file
            if self.config.verbose:
                console.print(f"[dim]Creating data URI for {file_path.suffix} file...[/dim]")
            data_uri = create_data_uri(file_path)

            # Determine document type based on file extension
            if file_path.suffix.lower() in DOCUMENT_EXTENSIONS:
                document = {"type": "document_url", "document_url": data_uri}
            else:
                document = {"type": "image_url", "image_url": data_uri}

            # Process with Mistral OCR
            if not hasattr(self.client, "ocr"):
                raise AttributeError(
                    "OCR endpoint not available in Mistral client. "
                    "Please ensure you have the latest mistralai package "
                    "and OCR access enabled for your API key."
                )

            if self.config.verbose:
                console.print("[dim]Sending to Mistral OCR API...[/dim]")
                console.print(f"[dim]Model: {self.config.model}[/dim]")

            # Build API kwargs (only pass OCR 3 params if set)
            ocr_kwargs = {
                "model": self.config.model,
                "document": document,
                "include_image_base64": self.config.include_images,
            }
            if self.config.table_format:
                ocr_kwargs["table_format"] = self.config.table_format
            if self.config.extract_header:
                ocr_kwargs["extract_header"] = True
            if self.config.extract_footer:
                ocr_kwargs["extract_footer"] = True

            response = self._call_with_retry(**ocr_kwargs)

            return {"file_path": file_path, "response": response, "success": True}

        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            # Always show errors, not just in verbose mode
            console.print(f"[red]{error_msg}[/red]")
            if self.config.verbose:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.errors.append({"file": str(file_path.resolve()), "error": str(e)})
            return None

    def save_results(
        self,
        result: dict,
        output_dir: Path,
        is_single_file: bool = False,
        base_dir: Path | None = None,
    ) -> None:
        """Save OCR results in a per-document folder structure.

        Output layout:
            output_dir/
            ├── doc_name/
            │   ├── doc_name.pdf        # original copy
            │   ├── doc_name.md         # OCR markdown
            │   ├── figures/            # extracted images
            │   └── tables/             # extracted tables
            └── metadata.json
        """
        file_path = result["file_path"]
        response = result["response"]

        # Per-document folder
        base_name = make_unique_basename(file_path, base_dir=base_dir)
        doc_dir = output_dir / base_name
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Copy original file into the document folder
        if self.config.save_original_images:
            original_copy = doc_dir / f"{base_name}{file_path.suffix}"
            shutil.copy2(file_path, original_copy)
            if self.config.verbose:
                console.print(f"[green]✓[/green] Saved original to {original_copy}")

        markdown_content = []

        # File header
        markdown_content.append("# OCR Results\n\n")
        markdown_content.append(f"**Original File:** {file_path.name}\n")
        markdown_content.append(f"**Full Path:** `{file_path}`\n")
        markdown_content.append(f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if self.config.save_original_images:
            markdown_content.append(
                f"**Original:** [{base_name}{file_path.suffix}](./{base_name}{file_path.suffix})\n\n"
            )

        markdown_content.append("---\n\n")

        # Process each page
        if hasattr(response, "pages"):
            for page in response.pages:
                markdown_content.append(f"## Page {page.index + 1}\n\n")

                # Page dimensions (OCR 3)
                if hasattr(page, "dimensions") and page.dimensions:
                    dims = page.dimensions
                    w = getattr(dims, "width", None)
                    h = getattr(dims, "height", None)
                    if w and h:
                        markdown_content.append(f"*Page size: {w} x {h}*\n\n")

                # Header (OCR 3)
                if hasattr(page, "header") and page.header:
                    markdown_content.append(f"> **Header:** {page.header}\n\n")

                # Extracted text
                if hasattr(page, "markdown"):
                    markdown_content.append(page.markdown)
                    markdown_content.append("\n\n")

                # Tables (OCR 3)
                if hasattr(page, "tables") and page.tables:
                    tables_dir = doc_dir / "tables"
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    ext = "html" if self.config.table_format == "html" else "md"
                    for tidx, table in enumerate(page.tables):
                        table_content = (
                            getattr(table, "content", None)
                            or getattr(table, "markdown", None)
                            or str(table)
                        )
                        table_filename = f"page{page.index + 1}_table{tidx + 1}.{ext}"
                        table_path = tables_dir / table_filename
                        with open(table_path, "w", encoding="utf-8") as tf:
                            tf.write(table_content)
                        markdown_content.append(
                            f"[Table {tidx + 1}](./tables/{table_filename})\n\n"
                        )

                # Hyperlinks (OCR 3)
                if hasattr(page, "hyperlinks") and page.hyperlinks:
                    markdown_content.append("**Hyperlinks:**\n")
                    for link in page.hyperlinks:
                        text = getattr(link, "text", "") or ""
                        url = getattr(link, "url", "") or getattr(link, "href", "") or ""
                        if url:
                            markdown_content.append(f"- [{text or url}]({url})\n")
                    markdown_content.append("\n")

                # Figures
                if self.config.include_images and hasattr(page, "images") and page.images:
                    figures_dir = doc_dir / "figures"
                    figures_dir.mkdir(parents=True, exist_ok=True)

                    for idx, image in enumerate(page.images):
                        b64_data = getattr(image, "image_base64", None) or getattr(
                            image, "base64", None
                        )
                        if b64_data:
                            img_id = getattr(image, "id", None) or f"img{idx + 1}"
                            img_ext = Path(img_id).suffix if "." in str(img_id) else ".png"
                            image_filename = f"page{page.index + 1}_img{idx + 1}{img_ext}"
                            image_path = figures_dir / image_filename
                            save_base64_image(b64_data, image_path)
                            markdown_content.append(
                                f"![Image {idx + 1}](./figures/{image_filename})\n\n"
                            )

                # Footer (OCR 3)
                if hasattr(page, "footer") and page.footer:
                    markdown_content.append(f"> **Footer:** {page.footer}\n\n")

        # Write markdown file
        markdown_path = doc_dir / f"{base_name}.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("".join(markdown_content))

        if self.config.verbose:
            console.print(f"[green]✓[/green] Saved results to {markdown_path}")

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        add_timestamp: bool = False,
        reprocess: bool = False,
    ) -> tuple[int, int]:
        """Process all supported files in a directory."""
        output_path = determine_output_path(input_dir, output_dir, add_timestamp=add_timestamp)

        # Exclude the output directory from file discovery
        files = get_supported_files(
            input_dir,
            exclude_paths=[output_path.resolve()],
        )

        if not files:
            console.print("[yellow]No supported files found in the directory.[/yellow]")
            return 0, 0

        # Load existing metadata to check for already processed files
        existing_metadata = load_metadata(output_path)
        existing_files_set = {
            str(Path(item["file"]).resolve()) for item in existing_metadata["files_processed"]
        }

        # Filter files based on reprocess flag
        files_to_process = []
        skipped_files = []
        for file_path in files:
            if str(file_path.resolve()) in existing_files_set and not reprocess:
                skipped_files.append(file_path)
                if self.config.verbose:
                    console.print(f"[dim]Skipping already processed: {file_path.name}[/dim]")
            else:
                files_to_process.append(file_path)

        if skipped_files:
            console.print(
                f"[yellow]Skipping {len(skipped_files)} already processed file(s)[/yellow]"
            )
            if not self.config.verbose:
                console.print("[dim]Use --verbose to see which files were skipped[/dim]")

        if not files_to_process:
            console.print(
                "[green]All files already processed. Use --reprocess to force reprocessing.[/green]"
            )
            return 0, 0

        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output directory: {output_path}[/blue]\n")

        start_time = time.time()
        success_count = 0
        # Capture prior-session time once to avoid overcounting on incremental flushes
        base_processing_time = existing_metadata.get("processing_time_seconds", 0)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing files...", total=len(files_to_process))

            for file_path in files_to_process:
                file_size = format_file_size(file_path.stat().st_size)
                progress.update(task, description=f"Processing {file_path.name} ({file_size})...")

                result = self.process_file(file_path)
                if result:
                    try:
                        self.save_results(
                            result, output_path, is_single_file=False, base_dir=input_dir
                        )
                        success_count += 1
                        base_name = make_unique_basename(file_path, base_dir=input_dir)
                        self.processed_files.append(
                            {
                                "file": str(file_path.resolve()),
                                "size": file_path.stat().st_size,
                                "output": str(output_path / base_name / f"{base_name}.md"),
                            }
                        )
                    except (OSError, ValueError) as e:
                        console.print(f"[red]Error saving results for {file_path.name}: {e}[/red]")
                        self.errors.append(
                            {"file": str(file_path.resolve()), "error": f"Save failed: {e}"}
                        )

                progress.update(task, advance=1)

                # Flush metadata incrementally so progress survives interruptions
                processing_time = time.time() - start_time
                save_metadata(
                    output_path,
                    self.processed_files,
                    processing_time,
                    self.errors,
                    base_processing_time=base_processing_time,
                )

        return success_count, len(files_to_process)

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        add_timestamp: bool = False,
        reprocess: bool = False,
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            # Process single file
            output_dir = determine_output_path(input_path, output_path, add_timestamp=add_timestamp)

            # Check if file already processed
            existing_metadata = load_metadata(output_dir)
            existing_files_set = {
                str(Path(item["file"]).resolve()) for item in existing_metadata["files_processed"]
            }

            if str(input_path.resolve()) in existing_files_set and not reprocess:
                base_name = make_unique_basename(input_path)
                output_file = output_dir / base_name / f"{base_name}.md"
                console.print(f"[yellow]File already processed: {input_path.name}[/yellow]")
                console.print(f"[dim]Output exists at: {output_file}[/dim]")
                console.print("[dim]Use --reprocess to force reprocessing.[/dim]")
                return

            console.print(f"[blue]Processing file: {input_path}[/blue]")
            console.print(f"[blue]Output directory: {output_dir}[/blue]\n")

            start_time = time.time()
            result = self.process_file(input_path)

            if result:
                self.save_results(result, output_dir, is_single_file=True)
                base_name = make_unique_basename(input_path)
                self.processed_files.append(
                    {
                        "file": str(input_path.resolve()),
                        "size": input_path.stat().st_size,
                        "output": str(output_dir / base_name / f"{base_name}.md"),
                    }
                )

                # Save metadata
                processing_time = time.time() - start_time
                save_metadata(output_dir, self.processed_files, processing_time, self.errors)

                console.print("\n[green]✓ Successfully processed 1 file[/green]")
                console.print(f"[dim]Processing time: {processing_time:.2f} seconds[/dim]")
            else:
                console.print("\n[red]✗ Failed to process file[/red]")

        elif input_path.is_dir():
            # Process directory
            success_count, total_count = self.process_directory(
                input_path, output_path, add_timestamp, reprocess
            )

            console.print(
                f"\n[green]✓ Successfully processed {success_count}/{total_count} files[/green]"
            )
            if self.errors:
                console.print(f"[red]✗ {len(self.errors)} file(s) failed[/red]")

        else:
            raise ValueError(f"Input path does not exist: {input_path}")
