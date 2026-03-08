from pathlib import Path

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


class StubOCR:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def process(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"ok": True, "kwargs": kwargs}


class StubClient:
    def __init__(self, failures):
        self.ocr = StubOCR(failures)


def test_process_file_retries_transient_errors(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test")
    processor.client = StubClient(
        [
            RuntimeError("429 Too Many Requests"),
            RuntimeError("503 Service Unavailable"),
        ]
    )
    processor.errors = []
    processor.processed_files = []

    monkeypatch.setattr("mistral_ocr.processor.time.sleep", lambda _: None)

    result = processor.process_file(pdf_path)

    assert result is not None
    assert result["success"] is True
    assert processor.client.ocr.calls == 3


def test_process_file_does_not_retry_non_retryable_errors(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test")
    processor.client = StubClient([RuntimeError("invalid api key")])
    processor.errors = []
    processor.processed_files = []

    result = processor.process_file(pdf_path)

    assert result is None
    assert processor.client.ocr.calls == 1
    assert len(processor.errors) == 1
