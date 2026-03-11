"""Tests for retry logic in OCRProcessor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


@pytest.fixture
def processor():
    """Create an OCRProcessor with a mocked Mistral client."""
    config = Config(api_key="test-key", max_retries=2, retry_base_delay=0.01)
    with patch("mistral_ocr.processor.Mistral") as mock_mistral_cls:
        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client
        proc = OCRProcessor(config)
    return proc


class TestIsRetryable:
    def test_timeout_error(self):
        assert OCRProcessor._is_retryable(TimeoutError("timed out")) is True

    def test_connection_error(self):
        assert OCRProcessor._is_retryable(ConnectionError("refused")) is True

    def test_os_error(self):
        assert OCRProcessor._is_retryable(OSError("network unreachable")) is True

    def test_value_error_not_retryable(self):
        assert OCRProcessor._is_retryable(ValueError("bad input")) is False

    def test_http_429(self):
        """HTTP 429 responses should be retryable."""
        err = Exception("rate limited")
        err.response = SimpleNamespace(status_code=429)
        assert OCRProcessor._is_retryable(err) is True

    def test_http_500(self):
        err = Exception("server error")
        err.response = SimpleNamespace(status_code=500)
        assert OCRProcessor._is_retryable(err) is True

    def test_http_400_not_retryable(self):
        err = Exception("bad request")
        err.response = SimpleNamespace(status_code=400)
        assert OCRProcessor._is_retryable(err) is False

    def test_rate_limit_error_by_name(self):
        """SDK exception types identified by class name."""

        class RateLimitError(Exception):
            pass

        assert OCRProcessor._is_retryable(RateLimitError()) is True

    def test_internal_server_error_by_name(self):
        class InternalServerError(Exception):
            pass

        assert OCRProcessor._is_retryable(InternalServerError()) is True


class TestCallWithRetry:
    def test_succeeds_first_try(self, processor):
        processor.client.ocr.process.return_value = {"pages": []}
        result = processor._call_with_retry(model="m", document={})
        assert result == {"pages": []}
        assert processor.client.ocr.process.call_count == 1

    def test_retries_on_transient_error(self, processor):
        processor.client.ocr.process.side_effect = [
            ConnectionError("reset"),
            {"pages": []},
        ]
        result = processor._call_with_retry(model="m", document={})
        assert result == {"pages": []}
        assert processor.client.ocr.process.call_count == 2

    def test_exhausts_retries(self, processor):
        processor.client.ocr.process.side_effect = ConnectionError("down")
        with pytest.raises(ConnectionError, match="down"):
            processor._call_with_retry(model="m", document={})
        # max_retries=2 means 3 attempts total
        assert processor.client.ocr.process.call_count == 3

    def test_no_retry_on_non_transient(self, processor):
        processor.client.ocr.process.side_effect = ValueError("bad doc")
        with pytest.raises(ValueError, match="bad doc"):
            processor._call_with_retry(model="m", document={})
        assert processor.client.ocr.process.call_count == 1

    def test_forwards_all_kwargs(self, processor):
        processor.client.ocr.process.return_value = {"pages": []}
        processor._call_with_retry(
            model="m",
            document={"type": "document_url"},
            include_image_base64=True,
            table_format="html",
            extract_header=True,
            extract_footer=True,
        )
        processor.client.ocr.process.assert_called_once_with(
            model="m",
            document={"type": "document_url"},
            include_image_base64=True,
            table_format="html",
            extract_header=True,
            extract_footer=True,
        )


class TestProcessFileRetry:
    def test_process_file_retries_transient_api_error(self, processor, tmp_path):
        """Integration: process_file retries on transient errors."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        mock_response = SimpleNamespace(pages=[])
        processor.client.ocr.process.side_effect = [
            ConnectionError("transient"),
            mock_response,
        ]

        result = processor.process_file(pdf)
        assert result is not None
        assert result["success"] is True
        assert processor.client.ocr.process.call_count == 2
