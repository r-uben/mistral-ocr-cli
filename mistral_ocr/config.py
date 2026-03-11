"""Configuration module for Mistral OCR."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for Mistral OCR."""

    api_key: str
    model: str = "mistral-ocr-latest"
    max_file_size_mb: int = 50
    include_images: bool = True
    save_original_images: bool = True
    table_format: str | None = None  # None, "markdown", or "html"
    extract_header: bool = False
    extract_footer: bool = False
    max_retries: int = 3
    retry_base_delay: float = 1.0
    dry_run: bool = False
    quiet: bool = False
    verbose: bool = False

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables. "
                "Please set it or create a .env file."
            )

        table_fmt = os.getenv("TABLE_FORMAT", "").lower() or None
        if table_fmt and table_fmt not in ("markdown", "html"):
            table_fmt = None

        try:
            max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        except ValueError as e:
            raise ValueError(
                f"MAX_FILE_SIZE_MB must be an integer, got: {os.getenv('MAX_FILE_SIZE_MB')!r}"
            ) from e
        if max_file_size_mb <= 0:
            raise ValueError(f"MAX_FILE_SIZE_MB must be positive, got: {max_file_size_mb}")

        try:
            max_retries = int(os.getenv("MAX_RETRIES", "3"))
        except ValueError as e:
            raise ValueError(
                f"MAX_RETRIES must be an integer, got: {os.getenv('MAX_RETRIES')!r}"
            ) from e
        if max_retries < 0:
            raise ValueError(f"MAX_RETRIES must be non-negative, got: {max_retries}")

        try:
            retry_base_delay = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
        except ValueError as e:
            raise ValueError(
                f"RETRY_BASE_DELAY must be a number, got: {os.getenv('RETRY_BASE_DELAY')!r}"
            ) from e
        if retry_base_delay < 0:
            raise ValueError(f"RETRY_BASE_DELAY must be non-negative, got: {retry_base_delay}")

        return cls(
            api_key=api_key,
            model=os.getenv("MISTRAL_MODEL", "mistral-ocr-latest"),
            max_file_size_mb=max_file_size_mb,
            include_images=os.getenv("INCLUDE_IMAGES", "true").lower() == "true",
            save_original_images=os.getenv("SAVE_ORIGINAL_IMAGES", "true").lower() == "true",
            table_format=table_fmt,
            extract_header=os.getenv("EXTRACT_HEADER", "false").lower() == "true",
            extract_footer=os.getenv("EXTRACT_FOOTER", "false").lower() == "true",
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
        )

    def validate_file_size(self, file_path: Path) -> None:
        """Validate that file size is within limits."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size "
                f"({self.max_file_size_mb} MB)"
            )
