"""Test basic imports and package structure."""


def test_imports():
    """Test that all modules can be imported."""
    from mistral_ocr import Config, OCRProcessor
    from mistral_ocr.cli import main
    from mistral_ocr.utils import (
        DOCUMENT_EXTENSIONS,
        IMAGE_EXTENSIONS,
        SUPPORTED_EXTENSIONS,
        encode_file_to_base64,
    )

    assert OCRProcessor is not None
    assert Config is not None
    assert main is not None
    assert encode_file_to_base64 is not None
    assert ".pdf" in DOCUMENT_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS
    assert SUPPORTED_EXTENSIONS == DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS


def test_version():
    """Test that version is defined."""
    from mistral_ocr import __version__

    assert __version__ == "1.1.1"
