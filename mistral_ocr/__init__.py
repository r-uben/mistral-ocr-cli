"""Mistral OCR CLI - A clean command-line tool for OCR processing using Mistral AI."""

__version__ = "1.0.2"
__author__ = "Ruben Fernandez-Fuertes"

from .processor import OCRProcessor
from .config import Config

__all__ = ["OCRProcessor", "Config"]