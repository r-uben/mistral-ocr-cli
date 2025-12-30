"""Core OCR processing module using Mistral AI."""

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mistralai import Mistral
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .config import Config
from .utils import (
    create_data_uri,
    determine_output_path,
    format_file_size,
    get_supported_files,
    load_metadata,
    sanitize_filename,
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
        except Exception as e:
            console.print(f"[red]Failed to initialize Mistral client: {e}[/red]")
            raise
        self.errors: List[Dict] = []
        self.processed_files: List[Dict] = []
    
    def process_file(self, file_path: Path) -> Optional[Dict]:
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
            if file_path.suffix.lower() == ".pdf":
                document = {
                    "type": "document_url",
                    "document_url": data_uri
                }
            else:
                document = {
                    "type": "image_url",
                    "image_url": data_uri
                }
            
            # Process with Mistral OCR
            if not hasattr(self.client, 'ocr'):
                raise AttributeError(
                    "OCR endpoint not available in Mistral client. "
                    "Please ensure you have the latest mistralai package "
                    "and OCR access enabled for your API key."
                )
            
            if self.config.verbose:
                console.print(f"[dim]Sending to Mistral OCR API...[/dim]")
                console.print(f"[dim]Model: {self.config.model}[/dim]")
            
            response = self.client.ocr.process(
                model=self.config.model,
                document=document,
                include_image_base64=self.config.include_images
            )
            
            return {
                "file_path": file_path,
                "response": response,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            # Always show errors, not just in verbose mode
            console.print(f"[red]{error_msg}[/red]")
            if self.config.verbose:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.errors.append({
                "file": str(file_path),
                "error": str(e)
            })
            return None
    
    def save_results(
        self, 
        result: Dict, 
        output_dir: Path,
        is_single_file: bool = False
    ) -> None:
        """Save OCR results to files."""
        file_path = result["file_path"]
        response = result["response"]
        
        # Always use the original filename (just sanitized, no truncation)
        base_name = sanitize_filename(file_path.stem, max_length=None)
        markdown_path = output_dir / f"{base_name}.md"
        
        # Save original input image if it's an image file (not PDF) and saving is enabled
        if self.config.save_original_images and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff']:
            originals_dir = output_dir / "original_images"
            originals_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the original file to the output directory with unique name
            # Prefix with base_name to avoid conflicts
            original_output_path = originals_dir / f"{base_name}{file_path.suffix}"
            shutil.copy2(file_path, original_output_path)
            
            if self.config.verbose:
                console.print(f"[green]✓[/green] Saved original image to {original_output_path}")
        
        markdown_content = []
        
        # Add file header
        markdown_content.append(f"# OCR Results\n\n")
        markdown_content.append(f"**Original File:** {file_path.name}\n")
        markdown_content.append(f"**Full Path:** `{file_path}`\n")
        markdown_content.append(f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add reference to original image if saved
        if self.config.save_original_images and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff']:
            markdown_content.append(f"**Original Image:** [View](./original_images/{base_name}{file_path.suffix})\n\n")
        
        markdown_content.append("---\n\n")
        
        # Process each page
        if hasattr(response, 'pages'):
            for page in response.pages:
                markdown_content.append(f"## Page {page.index + 1}\n\n")
                
                # Add extracted text
                if hasattr(page, 'markdown'):
                    markdown_content.append(page.markdown)
                    markdown_content.append("\n\n")
                
                # Save images if included - prefix with document name for clarity
                if self.config.include_images and hasattr(page, 'images') and page.images:
                    images_dir = output_dir / "extracted_images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    
                    for idx, image in enumerate(page.images):
                        if hasattr(image, 'base64'):
                            # Include document name in extracted image filename
                            image_filename = f"{base_name}_page{page.index + 1}_img{idx + 1}.png"
                            image_path = images_dir / image_filename
                            save_base64_image(image.base64, image_path)
                            
                            # Add image reference to markdown
                            markdown_content.append(f"![Image {idx + 1}](./extracted_images/{image_filename})\n\n")
        
        # Write markdown file
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("".join(markdown_content))
        
        if self.config.verbose:
            console.print(f"[green]✓[/green] Saved results to {markdown_path}")
    
    def process_directory(
        self, 
        input_dir: Path, 
        output_dir: Optional[Path] = None,
        add_timestamp: bool = False,
        reprocess: bool = False
    ) -> Tuple[int, int]:
        """Process all supported files in a directory."""
        files = get_supported_files(input_dir)
        
        if not files:
            console.print("[yellow]No supported files found in the directory.[/yellow]")
            return 0, 0
        
        output_path = determine_output_path(input_dir, output_dir, add_timestamp=add_timestamp)
        
        # Load existing metadata to check for already processed files
        existing_metadata = load_metadata(output_path)
        existing_files_set = {item["file"] for item in existing_metadata["files_processed"]}
        
        # Filter files based on reprocess flag
        files_to_process = []
        skipped_files = []
        for file_path in files:
            if str(file_path) in existing_files_set and not reprocess:
                skipped_files.append(file_path)
                if self.config.verbose:
                    console.print(f"[dim]Skipping already processed: {file_path.name}[/dim]")
            else:
                files_to_process.append(file_path)
        
        if skipped_files:
            console.print(f"[yellow]Skipping {len(skipped_files)} already processed file(s)[/yellow]")
            if not self.config.verbose:
                console.print("[dim]Use --verbose to see which files were skipped[/dim]")
        
        if not files_to_process:
            console.print("[green]All files already processed. Use --reprocess to force reprocessing.[/green]")
            return 0, 0
        
        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output directory: {output_path}[/blue]\n")
        
        start_time = time.time()
        success_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(files_to_process))
            
            for file_path in files_to_process:
                file_size = format_file_size(file_path.stat().st_size)
                progress.update(
                    task, 
                    description=f"Processing {file_path.name} ({file_size})..."
                )
                
                result = self.process_file(file_path)
                if result:
                    self.save_results(result, output_path, is_single_file=False)
                    success_count += 1
                    base_name = sanitize_filename(file_path.stem, max_length=None)
                    self.processed_files.append({
                        "file": str(file_path),
                        "size": file_path.stat().st_size,
                        "output": str(output_path / f"{base_name}.md")
                    })
                
                progress.update(task, advance=1)
        
        # Save metadata
        processing_time = time.time() - start_time
        save_metadata(output_path, self.processed_files, processing_time, self.errors)
        
        return success_count, len(files_to_process)
    
    def process(
        self, 
        input_path: Path, 
        output_path: Optional[Path] = None,
        add_timestamp: bool = False,
        reprocess: bool = False
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            # Process single file
            output_dir = determine_output_path(input_path, output_path, add_timestamp=add_timestamp)
            
            # Check if file already processed
            existing_metadata = load_metadata(output_dir)
            existing_files_set = {item["file"] for item in existing_metadata["files_processed"]}
            
            if str(input_path) in existing_files_set and not reprocess:
                base_name = sanitize_filename(input_path.stem, max_length=None)
                output_file = output_dir / f"{base_name}.md"
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
                base_name = sanitize_filename(input_path.stem, max_length=None)
                self.processed_files.append({
                    "file": str(input_path),
                    "size": input_path.stat().st_size,
                    "output": str(output_dir / f"{base_name}.md")
                })
                
                # Save metadata
                processing_time = time.time() - start_time
                save_metadata(output_dir, self.processed_files, processing_time, self.errors)
                
                console.print(f"\n[green]✓ Successfully processed 1 file[/green]")
                console.print(f"[dim]Processing time: {processing_time:.2f} seconds[/dim]")
            else:
                console.print(f"\n[red]✗ Failed to process file[/red]")
        
        elif input_path.is_dir():
            # Process directory
            success_count, total_count = self.process_directory(input_path, output_path, add_timestamp, reprocess)
            
            console.print(f"\n[green]✓ Successfully processed {success_count}/{total_count} files[/green]")
            if self.errors:
                console.print(f"[red]✗ {len(self.errors)} file(s) failed[/red]")
        
        else:
            raise ValueError(f"Input path does not exist: {input_path}")