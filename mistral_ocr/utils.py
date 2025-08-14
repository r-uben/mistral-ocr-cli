"""Utility functions for Mistral OCR."""

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def encode_file_to_base64(file_path: Path) -> str:
    """Encode a file to base64 string."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_mime_type(file_path: Path) -> str:
    """Get MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        if file_path.suffix.lower() == ".pdf":
            return "application/pdf"
        elif file_path.suffix.lower() in [".jpg", ".jpeg"]:
            return "image/jpeg"
        elif file_path.suffix.lower() == ".png":
            return "image/png"
        elif file_path.suffix.lower() == ".webp":
            return "image/webp"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return mime_type


def create_data_uri(file_path: Path) -> str:
    """Create a data URI from a file."""
    mime_type = get_mime_type(file_path)
    base64_data = encode_file_to_base64(file_path)
    return f"data:{mime_type};base64,{base64_data}"


def save_base64_image(base64_string: str, output_path: Path) -> None:
    """Save a base64 encoded image to file."""
    image_data = base64.b64decode(base64_string)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_data)


def get_supported_files(directory: Path) -> List[Path]:
    """Get all supported files from a directory."""
    supported_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
    files = []
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(file_path)
    
    return sorted(files)


def determine_output_path(
    input_path: Path, 
    output_path: Optional[Path] = None,
    default_folder_name: str = "mistral_ocr_output",
    add_timestamp: bool = False
) -> Path:
    """Determine the output path for OCR results."""
    if output_path:
        return output_path
    
    if input_path.is_file():
        parent_dir = input_path.parent
    else:
        parent_dir = input_path
    
    # Add timestamp if requested
    if add_timestamp:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{default_folder_name}_{timestamp}"
    else:
        folder_name = default_folder_name
    
    output_dir = parent_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_metadata(output_dir: Path) -> Dict:
    """Load existing metadata from JSON file."""
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {
        "files_processed": [],
        "total_files": 0,
        "processing_time_seconds": 0,
        "errors": [],
        "error_count": 0
    }


def save_metadata(
    output_dir: Path,
    files_processed: List[Dict],
    processing_time: float,
    errors: List[Dict]
) -> None:
    """Save processing metadata to JSON file (append/update mode)."""
    # Load existing metadata
    existing_metadata = load_metadata(output_dir)
    
    # Create a dict of existing files for quick lookup
    existing_files = {item["file"]: item for item in existing_metadata["files_processed"]}
    
    # Update with new files (overwrite if exists, add if new)
    for new_file in files_processed:
        new_file["last_processed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        existing_files[new_file["file"]] = new_file
    
    # Update metadata
    metadata = {
        "files_processed": list(existing_files.values()),
        "total_files": len(existing_files),
        "processing_time_seconds": existing_metadata["processing_time_seconds"] + processing_time,
        "errors": errors,  # Errors from current session only
        "error_count": len(errors),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def sanitize_filename(filename: str, max_length: Optional[int] = None) -> str:
    """Sanitize filename by removing or replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    
    # Only truncate if max_length is specified
    if max_length is not None:
        # Truncate long filenames but keep extension
        if len(filename) > max_length and '.' in filename:
            name, ext = filename.rsplit('.', 1)
            if len(name) > max_length - len(ext) - 1:
                name = name[:max_length - len(ext) - 4] + "..."
            filename = f"{name}.{ext}"
        elif len(filename) > max_length:
            filename = filename[:max_length - 3] + "..."
    
    return filename