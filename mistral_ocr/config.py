"""Configuration module for Mistral OCR."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for Mistral OCR."""
    
    api_key: str
    model: str = "mistral-ocr-latest"
    max_file_size_mb: int = 50
    max_pages: int = 1000
    output_format: str = "markdown"
    include_images: bool = True
    save_original_images: bool = True
    verbose: bool = False
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
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
        
        return cls(
            api_key=api_key,
            model=os.getenv("MISTRAL_MODEL", "mistral-ocr-latest"),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            max_pages=int(os.getenv("MAX_PAGES", "1000")),
            output_format=os.getenv("OUTPUT_FORMAT", "markdown"),
            include_images=os.getenv("INCLUDE_IMAGES", "true").lower() == "true",
            save_original_images=os.getenv("SAVE_ORIGINAL_IMAGES", "true").lower() == "true",
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