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
        _fallback = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".avif": "image/avif",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        mime_type = _fallback.get(file_path.suffix.lower())
        if not mime_type:
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


def get_supported_files(
    directory: Path,
    exclude_dirs: Optional[List[str]] = None
) -> List[Path]:
    """Get all supported files from a directory, excluding output directories."""
    supported_extensions = {
        ".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff",
        ".avif", ".docx", ".pptx",
    }
    if exclude_dirs is None:
        exclude_dirs = ["mistral_ocr_output"]
    files = []

    exclude_set = set(exclude_dirs)
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            # Skip files inside excluded directories (check parent dirs only, not filename)
            rel_parts = file_path.relative_to(directory).parts[:-1]  # exclude filename
            if any(part in exclude_set for part in rel_parts):
                continue
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
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    if input_path.is_file():
        parent_dir = input_path.parent
    else:
        parent_dir = input_path
    
    # Add timestamp if requested
    if add_timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{default_folder_name}_{timestamp}"
    else:
        folder_name = default_folder_name
    
    output_dir = parent_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _empty_metadata() -> Dict:
    return {
        "files_processed": [],
        "total_files": 0,
        "processing_time_seconds": 0,
        "errors": [],
        "error_count": 0
    }


def load_metadata(output_dir: Path) -> Dict:
    """Load existing metadata from JSON file.

    Returns empty metadata on missing or corrupt files.
    """
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            return _empty_metadata()
    return _empty_metadata()


def save_metadata(
    output_dir: Path,
    files_processed: List[Dict],
    processing_time: float,
    errors: List[Dict],
    base_processing_time: Optional[float] = None
) -> None:
    """Save processing metadata to JSON file (append/update mode).

    Args:
        processing_time: Elapsed time for the *current* session.
        base_processing_time: Accumulated time from *prior* sessions. If None,
            loaded from existing metadata (use this on the first call). Pass
            the returned value on subsequent calls within the same session to
            avoid overcounting.
    """
    # Load existing metadata
    existing_metadata = load_metadata(output_dir)

    if base_processing_time is None:
        base_processing_time = existing_metadata.get("processing_time_seconds", 0)

    # Create a dict of existing files for quick lookup
    existing_files = {item["file"]: item for item in existing_metadata["files_processed"]}

    # Update with new files (overwrite if exists, add if new)
    for new_file in files_processed:
        new_file["last_processed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        existing_files[new_file["file"]] = new_file

    # Merge errors: keep historical errors, append new ones (deduplicate by file)
    existing_errors = {e["file"]: e for e in existing_metadata.get("errors", [])}
    for err in errors:
        err["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        existing_errors[err["file"]] = err
    all_errors = list(existing_errors.values())

    # Update metadata — total time = prior sessions + current session elapsed
    metadata = {
        "files_processed": list(existing_files.values()),
        "total_files": len(existing_files),
        "processing_time_seconds": base_processing_time + processing_time,
        "errors": all_errors,
        "error_count": len(all_errors),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = output_dir / "metadata.json"
    tmp_path = metadata_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    tmp_path.replace(metadata_path)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def make_unique_basename(file_path: Path, base_dir: Optional[Path] = None) -> str:
    """Create a unique base name for output files.

    When base_dir is provided (directory mode), includes the relative path
    to disambiguate files with the same stem in different subdirectories.
    E.g., subdir/report.pdf -> subdir__report
    """
    stem = file_path.stem
    if base_dir is not None:
        try:
            rel = file_path.parent.relative_to(base_dir)
            if rel != Path("."):
                prefix = str(rel).replace("/", "__").replace("\\", "__")
                stem = f"{prefix}__{stem}"
        except ValueError:
            pass
    return sanitize_filename(stem, max_length=200)


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