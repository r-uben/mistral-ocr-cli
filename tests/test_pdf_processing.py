from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfReader, PdfWriter

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


def make_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as file:
        writer.write(file)


class StubFiles:
    def __init__(self) -> None:
        self.uploaded_page_counts = {}
        self.deleted = []

    def upload(self, file, purpose):
        data = file["content"].read()
        file_id = f"file_{len(self.uploaded_page_counts)}"
        self.uploaded_page_counts[file_id] = len(PdfReader(BytesIO(data)).pages)
        return SimpleNamespace(id=file_id)

    def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)


class StubOCR:
    def __init__(self, files: StubFiles) -> None:
        self.files = files
        self.calls = []

    def process(self, **kwargs):
        self.calls.append(kwargs)
        page_count = self.files.uploaded_page_counts[kwargs["document"]["file_id"]]
        return SimpleNamespace(
            pages=[
                SimpleNamespace(index=index, markdown=f"page {index + 1}", images=[])
                for index in range(page_count)
            ]
        )


class StubClient:
    def __init__(self) -> None:
        self.files = StubFiles()
        self.ocr = StubOCR(self.files)


def test_process_file_uploads_pdf_chunks_and_reassembles_pages(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    make_pdf(pdf_path, pages=3)

    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", max_pages=0)
    processor.client = StubClient()
    processor.errors = []
    processor.processed_files = []

    monkeypatch.setattr("mistral_ocr.processor.PDF_REQUEST_PAGE_LIMIT", 2)

    result = processor.process_file(pdf_path)

    assert result is not None
    assert list(processor.client.files.uploaded_page_counts.values()) == [2, 1]
    assert [page.index for page in result["response"].pages] == [0, 1, 2]
    assert processor.client.files.deleted == ["file_0", "file_1"]


def test_process_file_respects_max_pages_limit(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    make_pdf(pdf_path, pages=5)

    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", max_pages=3)
    processor.client = StubClient()
    processor.errors = []
    processor.processed_files = []

    monkeypatch.setattr("mistral_ocr.processor.PDF_REQUEST_PAGE_LIMIT", 2)

    result = processor.process_file(pdf_path)

    assert result is not None
    assert list(processor.client.files.uploaded_page_counts.values()) == [2, 1]
    assert [page.index for page in result["response"].pages] == [0, 1, 2]
    assert result["response"].truncated_message == "Document truncated to first 3 of 5 pages."
