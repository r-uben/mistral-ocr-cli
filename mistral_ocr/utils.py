"""Utility functions for Mistral OCR.

Output-shape concerns (output-root resolution, input-relative keying, the
per-document layout, page assembly, metadata, figure naming and the exit-code
policy) are owned by the shared ``ocr-output-contract`` package and live in
:mod:`mistral_ocr.processor`. What remains here is mistral-specific I/O: MIME
sniffing / data-URI construction for the API, base64 image decoding, PDF page
counting + splitting for the API's per-request page cap, and supported-file
discovery for batch runs.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ocr_output_contract import iter_input_files

# Canonical extension sets — used by processor.py and discovery.
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".avif"}
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS


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


def decode_base64_image(base64_string: str) -> bytes:
    """Decode a base64 (optionally data-URI-prefixed) image into raw bytes."""
    if "," in base64_string and base64_string.lstrip().startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    return base64.b64decode(base64_string)


def get_supported_files(directory: Path, output_root: Path) -> list[Path]:
    """Get all supported input files under ``directory``, excluding outputs.

    Discovery is delegated to the contract's :func:`iter_input_files`, which
    recurses ``directory`` and prunes everything at or under the RESOLVED
    ``output_root`` (so the engine never re-ingests its own ``.md``/figure
    outputs on a rerun). It targets the *real* output directory by resolved path,
    not any path component that merely happens to be named ``ocr`` — fixing the
    "files under any directory named 'ocr' silently skipped" bug that was acutely
    fatal under this user's own ``.../toolkits/ocr/...`` tree.
    """
    return list(iter_input_files(directory, output_root, suffixes=SUPPORTED_EXTENSIONS))


def get_pdf_page_count(file_path: Path) -> int:
    """Return the number of pages in a PDF."""
    from pypdf import PdfReader

    return len(PdfReader(str(file_path)).pages)


def split_pdf(
    file_path: Path,
    output_dir: Path,
    *,
    max_pages_per_chunk: int = 1000,
    max_pages: int | None = None,
) -> list[tuple[Path, int, int]]:
    """Split a PDF into page-bounded chunks.

    Returns a list of (chunk_path, start_page_index, page_count) tuples.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, int, int]] = []
    start = 0
    chunk_idx = 0

    while start < pages_to_process:
        end = min(start + max_pages_per_chunk, pages_to_process)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        chunk_path = output_dir / f"{file_path.stem}_chunk{chunk_idx + 1}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)

        chunks.append((chunk_path, start, end - start))
        start = end
        chunk_idx += 1

    return chunks


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"
