"""Tests for PDF chunking and upload-based processing."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mistral_ocr.config import Config
from mistral_ocr.processor import MAX_PAGES_PER_REQUEST, OCRProcessor
from mistral_ocr.utils import get_pdf_page_count, split_pdf

# ---------------------------------------------------------------------------
# Utility tests (split_pdf, get_pdf_page_count)
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, num_pages: int) -> Path:
    """Create a minimal PDF with *num_pages* blank pages using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)
    return path


class TestGetPdfPageCount:
    def test_counts_pages(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 5)
        assert get_pdf_page_count(pdf) == 5

    def test_single_page(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "one.pdf", 1)
        assert get_pdf_page_count(pdf) == 1


class TestSplitPdf:
    def test_no_split_needed(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "small.pdf", 3)
        out = tmp_path / "chunks"
        chunks = split_pdf(pdf, out, max_pages_per_chunk=10)
        assert len(chunks) == 1
        chunk_path, start, count = chunks[0]
        assert start == 0
        assert count == 3
        assert chunk_path.exists()

    def test_splits_into_multiple_chunks(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "big.pdf", 7)
        out = tmp_path / "chunks"
        chunks = split_pdf(pdf, out, max_pages_per_chunk=3)
        assert len(chunks) == 3
        # chunk1: pages 0-2, chunk2: 3-5, chunk3: 6
        assert chunks[0] == (out / "big_chunk1.pdf", 0, 3)
        assert chunks[1] == (out / "big_chunk2.pdf", 3, 3)
        assert chunks[2] == (out / "big_chunk3.pdf", 6, 1)
        # Verify each chunk has the right page count
        assert get_pdf_page_count(chunks[0][0]) == 3
        assert get_pdf_page_count(chunks[1][0]) == 3
        assert get_pdf_page_count(chunks[2][0]) == 1

    def test_max_pages_limits_output(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "many.pdf", 10)
        out = tmp_path / "chunks"
        chunks = split_pdf(pdf, out, max_pages_per_chunk=5, max_pages=7)
        assert len(chunks) == 2
        assert chunks[0][2] == 5  # first chunk: 5 pages
        assert chunks[1][2] == 2  # second chunk: 2 pages (7-5)

    def test_max_pages_none_processes_all(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "all.pdf", 4)
        out = tmp_path / "chunks"
        chunks = split_pdf(pdf, out, max_pages_per_chunk=100, max_pages=None)
        assert len(chunks) == 1
        assert chunks[0][2] == 4


# ---------------------------------------------------------------------------
# Config: MAX_PAGES env var
# ---------------------------------------------------------------------------


class TestConfigMaxPages:
    def test_default_is_none(self) -> None:
        config = Config(api_key="test")
        assert config.max_pages is None

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "key", "MAX_PAGES": "50"})
    def test_from_env(self) -> None:
        config = Config.from_env()
        assert config.max_pages == 50

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "key", "MAX_PAGES": "abc"})
    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="MAX_PAGES must be an integer"):
            Config.from_env()

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "key", "MAX_PAGES": "0"})
    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="MAX_PAGES must be positive"):
            Config.from_env()

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "key", "MAX_PAGES": "-5"})
    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="MAX_PAGES must be positive"):
            Config.from_env()


# ---------------------------------------------------------------------------
# Processor: _process_pdf routing
# ---------------------------------------------------------------------------


def _make_processor(**overrides: object) -> OCRProcessor:
    defaults = {"api_key": "test-key", "include_images": False, "save_original_images": False}
    defaults.update(overrides)
    config = Config(**defaults)
    with patch("mistral_ocr.processor.Mistral"):
        return OCRProcessor(config)


def _fake_page(index: int) -> SimpleNamespace:
    return SimpleNamespace(index=index, markdown=f"Page {index + 1} text")


class TestProcessPdf:
    """Test _process_pdf routing logic with mocked upload."""

    @patch("mistral_ocr.processor.get_pdf_page_count", return_value=5)
    def test_small_pdf_uploads_directly(self, mock_count: MagicMock) -> None:
        proc = _make_processor()
        proc._upload_and_process = MagicMock(
            return_value=SimpleNamespace(pages=[_fake_page(i) for i in range(5)])
        )
        result = proc._process_pdf(Path("test.pdf"))
        proc._upload_and_process.assert_called_once_with(Path("test.pdf"))
        assert len(result.pages) == 5

    @patch("mistral_ocr.processor.get_pdf_page_count", return_value=5)
    def test_max_pages_triggers_chunking(self, mock_count: MagicMock) -> None:
        proc = _make_processor(max_pages=3)
        proc._process_pdf_chunked = MagicMock(
            return_value=SimpleNamespace(pages=[_fake_page(i) for i in range(3)])
        )
        result = proc._process_pdf(Path("test.pdf"))
        proc._process_pdf_chunked.assert_called_once_with(Path("test.pdf"), 5)
        assert len(result.pages) == 3

    @patch("mistral_ocr.processor.get_pdf_page_count", return_value=1500)
    def test_large_pdf_triggers_chunking(self, mock_count: MagicMock) -> None:
        proc = _make_processor()
        proc._process_pdf_chunked = MagicMock(
            return_value=SimpleNamespace(pages=[_fake_page(i) for i in range(1500)])
        )
        result = proc._process_pdf(Path("test.pdf"))
        proc._process_pdf_chunked.assert_called_once_with(Path("test.pdf"), 1500)
        assert len(result.pages) == 1500


class TestProcessPdfChunked:
    """Test _process_pdf_chunked with real split_pdf but mocked API calls."""

    def test_reassembles_pages_with_correct_indices(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 5)
        proc = _make_processor()

        def fake_upload(chunk_path: Path) -> SimpleNamespace:
            from mistral_ocr.utils import get_pdf_page_count as gpc

            n = gpc(chunk_path)
            return SimpleNamespace(pages=[_fake_page(i) for i in range(n)])

        proc._upload_and_process = MagicMock(side_effect=fake_upload)
        result = proc._process_pdf_chunked(pdf, total_pages=5)
        # Should have 5 pages with indices 0-4
        assert len(result.pages) == 5
        assert [p.index for p in result.pages] == [0, 1, 2, 3, 4]

    def test_truncation_note_when_max_pages(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 10)
        proc = _make_processor(max_pages=3)

        def fake_upload(chunk_path: Path) -> SimpleNamespace:
            from mistral_ocr.utils import get_pdf_page_count as gpc

            n = gpc(chunk_path)
            return SimpleNamespace(pages=[_fake_page(i) for i in range(n)])

        proc._upload_and_process = MagicMock(side_effect=fake_upload)
        result = proc._process_pdf_chunked(pdf, total_pages=10)
        assert len(result.pages) == 3
        assert hasattr(result, "truncated")
        assert "3 of 10" in result.truncated

    def test_no_truncation_note_when_all_pages(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", 3)
        proc = _make_processor()

        def fake_upload(chunk_path: Path) -> SimpleNamespace:
            from mistral_ocr.utils import get_pdf_page_count as gpc

            n = gpc(chunk_path)
            return SimpleNamespace(pages=[_fake_page(i) for i in range(n)])

        proc._upload_and_process = MagicMock(side_effect=fake_upload)
        result = proc._process_pdf_chunked(pdf, total_pages=3)
        assert not hasattr(result, "truncated")


class TestBuildOcrKwargs:
    def test_basic_kwargs(self) -> None:
        proc = _make_processor()
        doc = {"type": "file", "file_id": "abc"}
        kwargs = proc._build_ocr_kwargs(doc)
        assert kwargs["model"] == "mistral-ocr-latest"
        assert kwargs["document"] == doc
        assert kwargs["include_image_base64"] is False

    def test_with_table_format(self) -> None:
        proc = _make_processor(table_format="html")
        kwargs = proc._build_ocr_kwargs({"type": "file", "file_id": "x"})
        assert kwargs["table_format"] == "html"

    def test_with_headers_footers(self) -> None:
        proc = _make_processor(extract_header=True, extract_footer=True)
        kwargs = proc._build_ocr_kwargs({"type": "file", "file_id": "x"})
        assert kwargs["extract_header"] is True
        assert kwargs["extract_footer"] is True


class TestMaxPagesPerRequest:
    def test_constant_value(self) -> None:
        assert MAX_PAGES_PER_REQUEST == 1000
