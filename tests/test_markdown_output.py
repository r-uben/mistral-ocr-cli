"""Tests for markdown output flags (--metadata, --page-headings)."""

from types import SimpleNamespace

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


def _make_processor(**config_kwargs):
    """Create an OCRProcessor without hitting the Mistral API."""
    config_kwargs.setdefault("save_original_images", False)
    proc = OCRProcessor.__new__(OCRProcessor)
    proc.config = Config(api_key="test", **config_kwargs)
    proc.errors = []
    proc.processed_files = []
    return proc


def _fake_response(*page_texts):
    return SimpleNamespace(
        pages=[
            SimpleNamespace(index=i, markdown=text, images=[]) for i, text in enumerate(page_texts)
        ]
    )


class TestIncludeMetadata:
    def test_metadata_included_by_default(self, tmp_path):
        proc = _make_processor()
        output = tmp_path / "out"
        output.mkdir()
        proc.save_results(
            {"file_path": tmp_path / "doc.pdf", "response": _fake_response("Hello")},
            output,
        )
        md = (output / "doc" / "doc.md").read_text()
        assert "# OCR Results" in md
        assert "**Original File:**" in md

    def test_metadata_omitted(self, tmp_path):
        proc = _make_processor(include_metadata=False)
        output = tmp_path / "out"
        output.mkdir()
        proc.save_results(
            {"file_path": tmp_path / "doc.pdf", "response": _fake_response("Hello")},
            output,
        )
        md = (output / "doc" / "doc.md").read_text()
        assert "# OCR Results" not in md
        assert "**Original File:**" not in md
        assert "Hello" in md


class TestIncludePageHeadings:
    def test_page_headings_included_by_default(self, tmp_path):
        proc = _make_processor()
        output = tmp_path / "out"
        output.mkdir()
        proc.save_results(
            {"file_path": tmp_path / "doc.pdf", "response": _fake_response("A", "B")},
            output,
        )
        md = (output / "doc" / "doc.md").read_text()
        assert "## Page 1" in md
        assert "## Page 2" in md

    def test_page_headings_omitted(self, tmp_path):
        proc = _make_processor(include_page_headings=False)
        output = tmp_path / "out"
        output.mkdir()
        proc.save_results(
            {"file_path": tmp_path / "doc.pdf", "response": _fake_response("A", "B")},
            output,
        )
        md = (output / "doc" / "doc.md").read_text()
        assert "## Page 1" not in md
        assert "## Page 2" not in md
        assert "A" in md
        assert "B" in md


class TestBothDisabled:
    def test_raw_text_only(self, tmp_path):
        proc = _make_processor(include_metadata=False, include_page_headings=False)
        output = tmp_path / "out"
        output.mkdir()
        proc.save_results(
            {"file_path": tmp_path / "doc.pdf", "response": _fake_response("Content here")},
            output,
        )
        md = (output / "doc" / "doc.md").read_text()
        assert "# OCR Results" not in md
        assert "## Page" not in md
        assert "Content here" in md
