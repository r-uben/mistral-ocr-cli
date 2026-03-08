from types import SimpleNamespace

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


def test_save_results_can_omit_markdown_metadata_block(tmp_path) -> None:
    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", include_metadata=False)

    response = SimpleNamespace(
        truncated_message="Document truncated to first 1 of 3 pages.",
        pages=[SimpleNamespace(index=0, markdown="Hello world", images=[])],
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    processor.save_results(
        {"file_path": tmp_path / "document.pdf", "response": response},
        output_dir,
    )

    markdown = (output_dir / "document.md").read_text()
    assert "# OCR Results" not in markdown
    assert "**Original File:**" not in markdown
    assert "**Note:**" not in markdown
    assert markdown.startswith("## Page 1\n\nHello world")


def test_save_results_can_omit_page_headings(tmp_path) -> None:
    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", include_page_headings=False)

    response = SimpleNamespace(
        pages=[
            SimpleNamespace(index=0, markdown="First page text", images=[]),
            SimpleNamespace(index=1, markdown="Second page text", images=[]),
        ]
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    processor.save_results(
        {"file_path": tmp_path / "document.pdf", "response": response},
        output_dir,
    )

    markdown = (output_dir / "document.md").read_text()
    assert "## Page 1" not in markdown
    assert "## Page 2" not in markdown
    assert "First page text" in markdown
    assert "Second page text" in markdown
