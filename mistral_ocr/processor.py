"""Core OCR processing module using Mistral AI."""

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
        
        # For single files, use simpler naming
        if is_single_file:
            base_name = "output"
            markdown_path = output_dir / "output.md"
        else:
            # For multiple files, use sanitized filename
            base_name = sanitize_filename(file_path.stem, max_length=40)
            markdown_path = output_dir / f"{base_name}.md"
        
        markdown_content = []
        
        # Add file header
        markdown_content.append(f"# OCR Results\n\n")
        markdown_content.append(f"**Original File:** {file_path.name}\n")
        markdown_content.append(f"**Full Path:** `{file_path}`\n")
        markdown_content.append(f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        markdown_content.append("---\n\n")
        
        # Process each page
        if hasattr(response, 'pages'):
            for page in response.pages:
                markdown_content.append(f"## Page {page.index + 1}\n\n")
                
                # Add extracted text
                if hasattr(page, 'markdown'):
                    markdown_content.append(page.markdown)
                    markdown_content.append("\n\n")
                
                # Save images if included
                if self.config.include_images and hasattr(page, 'images') and page.images:
                    images_dir = output_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    
                    for idx, image in enumerate(page.images):
                        if hasattr(image, 'base64'):
                            image_filename = f"page{page.index + 1}_img{idx + 1}.png"
                            image_path = images_dir / image_filename
                            save_base64_image(image.base64, image_path)
                            
                            # Add image reference to markdown
                            markdown_content.append(f"![Image {idx + 1}](./images/{image_filename})\n\n")
        
        # Write markdown file
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("".join(markdown_content))
        
        if self.config.verbose:
            console.print(f"[green]✓[/green] Saved results to {markdown_path}")
    
    def process_directory(
        self, 
        input_dir: Path, 
        output_dir: Optional[Path] = None,
        add_timestamp: bool = False
    ) -> Tuple[int, int]:
        """Process all supported files in a directory."""
        files = get_supported_files(input_dir)
        
        if not files:
            console.print("[yellow]No supported files found in the directory.[/yellow]")
            return 0, 0
        
        output_path = determine_output_path(input_dir, output_dir, add_timestamp=add_timestamp)
        console.print(f"[blue]Processing {len(files)} file(s)...[/blue]")
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
            task = progress.add_task("Processing files...", total=len(files))
            
            for file_path in files:
                file_size = format_file_size(file_path.stat().st_size)
                progress.update(
                    task, 
                    description=f"Processing {file_path.name} ({file_size})..."
                )
                
                result = self.process_file(file_path)
                if result:
                    self.save_results(result, output_path, is_single_file=False)
                    success_count += 1
                    base_name = sanitize_filename(file_path.stem, max_length=40)
                    self.processed_files.append({
                        "file": str(file_path),
                        "size": file_path.stat().st_size,
                        "output": str(output_path / f"{base_name}.md")
                    })
                
                progress.update(task, advance=1)
        
        # Save metadata
        processing_time = time.time() - start_time
        save_metadata(output_path, self.processed_files, processing_time, self.errors)
        
        return success_count, len(files)
    
    def process(
        self, 
        input_path: Path, 
        output_path: Optional[Path] = None,
        add_timestamp: bool = False
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            # Process single file
            output_dir = determine_output_path(input_path, output_path, add_timestamp=add_timestamp)
            console.print(f"[blue]Processing file: {input_path}[/blue]")
            console.print(f"[blue]Output directory: {output_dir}[/blue]\n")
            
            start_time = time.time()
            result = self.process_file(input_path)
            
            if result:
                self.save_results(result, output_dir, is_single_file=True)
                self.processed_files.append({
                    "file": str(input_path),
                    "size": input_path.stat().st_size,
                    "output": str(output_dir / "output.md")
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
            success_count, total_count = self.process_directory(input_path, output_path, add_timestamp)
            
            console.print(f"\n[green]✓ Successfully processed {success_count}/{total_count} files[/green]")
            if self.errors:
                console.print(f"[red]✗ {len(self.errors)} file(s) failed[/red]")
        
        else:
            raise ValueError(f"Input path does not exist: {input_path}")