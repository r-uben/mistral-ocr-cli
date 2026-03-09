"""Mistral OCR CLI - A clean command-line tool for OCR processing using Mistral AI."""

__version__ = "1.0.0"
__author__ = "Ruben Fernandez-Fuertes"

from .config import Config
from .processor import OCRProcessor

__all__ = ["OCRProcessor", "Config"]