"""Tests for CLI config overrides and process() orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mistral_ocr.cli import main
from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


@pytest.fixture
def runner():
    return CliRunner()


def _mock_processor():
    """Return a mock OCRProcessor that does nothing."""
    mock = MagicMock(spec=OCRProcessor)
    mock.errors = []
    mock.process.return_value = None
    return mock


# ---------------------------------------------------------------------------
# CLI config override tests
# ---------------------------------------------------------------------------


class TestCliOverrides:
    """CLI flags override env/config defaults only when explicitly passed."""

    def test_model_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--model", "custom-model"])
            config = mock_cls.call_args[0][0]
            assert config.model == "custom-model"

    def test_no_images_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--no-images"])
            config = mock_cls.call_args[0][0]
            assert config.include_images is False

    def test_table_format_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--table-format", "html"])
            config = mock_cls.call_args[0][0]
            assert config.table_format == "html"

    def test_extract_headers_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--extract-headers"])
            config = mock_cls.call_args[0][0]
            assert config.extract_header is True

    def test_extract_footers_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--extract-footers"])
            config = mock_cls.call_args[0][0]
            assert config.extract_footer is True

    def test_workers_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--workers", "4"])
            config = mock_cls.call_args[0][0]
            assert config.max_workers == 4

    def test_max_pages_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--max-pages", "100"])
            config = mock_cls.call_args[0][0]
            assert config.max_pages == 100

    def test_no_metadata_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--no-metadata"])
            config = mock_cls.call_args[0][0]
            assert config.include_metadata is False

    def test_no_page_headings_override(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf), "--no-page-headings"])
            config = mock_cls.call_args[0][0]
            assert config.include_page_headings is False

    def test_defaults_not_overridden(self, runner, tmp_path, monkeypatch):
        """When no CLI flags are passed, config keeps env/default values."""
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        monkeypatch.setenv("INCLUDE_IMAGES", "false")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            runner.invoke(main, [str(pdf)])
            config = mock_cls.call_args[0][0]
            # Env says false, CLI didn't override — should stay false
            assert config.include_images is False


# ---------------------------------------------------------------------------
# CLI error handling
# ---------------------------------------------------------------------------


class TestCliErrors:
    def test_missing_api_key(self, runner, tmp_path, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        result = runner.invoke(main, [str(pdf)])
        assert result.exit_code == 1
        assert "MISTRAL_API_KEY" in result.output

    def test_nonexistent_path(self, runner, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        result = runner.invoke(main, ["/nonexistent/path"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_api_key_via_flag(self, runner, tmp_path, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            result = runner.invoke(main, [str(pdf), "--api-key", "my-key"])
            assert result.exit_code == 0

    def test_version_flag(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.2.0" in result.output

    def test_verbose_flag(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_cls.return_value = _mock_processor()
            result = runner.invoke(main, [str(pdf), "--verbose"])
            assert result.exit_code == 0
            config = mock_cls.call_args[0][0]
            assert config.verbose is True


# ---------------------------------------------------------------------------
# process() orchestration
# ---------------------------------------------------------------------------


class TestProcessOrchestration:
    """Test the process() method routing and skip logic."""

    def _make_real_processor(self, **overrides):
        defaults = {"api_key": "test", "save_original_images": False}
        defaults.update(overrides)
        proc = OCRProcessor.__new__(OCRProcessor)
        proc.config = Config(**defaults)
        proc.client = MagicMock()
        proc.errors = []
        proc.processed_files = []
        import threading

        proc._lock = threading.Lock()
        return proc

    def test_single_file_success(self, tmp_path):
        proc = self._make_real_processor()
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        response = SimpleNamespace(pages=[SimpleNamespace(index=0, markdown="Hello", images=[])])
        proc.client.ocr.process.return_value = response

        proc.process(img)
        assert len(proc.processed_files) == 1
        assert len(proc.errors) == 0
        # Check output was created
        out_dir = tmp_path / "mistral_ocr_output"
        assert (out_dir / "doc" / "doc.md").exists()

    def test_single_file_skip_already_processed(self, tmp_path):
        proc = self._make_real_processor()
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        # First process
        response = SimpleNamespace(pages=[SimpleNamespace(index=0, markdown="Hello", images=[])])
        proc.client.ocr.process.return_value = response
        proc.process(img)

        # Second process — should skip
        proc.client.ocr.process.reset_mock()
        proc.process(img)
        proc.client.ocr.process.assert_not_called()

    def test_single_file_reprocess(self, tmp_path):
        proc = self._make_real_processor()
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        response = SimpleNamespace(pages=[SimpleNamespace(index=0, markdown="Hello", images=[])])
        proc.client.ocr.process.return_value = response
        proc.process(img)

        # Reprocess with flag
        proc.process(img, reprocess=True)
        assert proc.client.ocr.process.call_count == 2

    def test_single_file_failure(self, tmp_path):
        proc = self._make_real_processor()
        img = tmp_path / "doc.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        proc.client.ocr.process.side_effect = RuntimeError("API down")
        proc.process(img)
        assert len(proc.errors) == 1
        assert len(proc.processed_files) == 0

    def test_nonexistent_path_raises(self, tmp_path):
        proc = self._make_real_processor()
        with pytest.raises(ValueError, match="does not exist"):
            proc.process(tmp_path / "nope.png")

    def test_directory_processing(self, tmp_path):
        proc = self._make_real_processor()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        response = SimpleNamespace(pages=[SimpleNamespace(index=0, markdown="text", images=[])])
        proc.client.ocr.process.return_value = response
        proc.process(input_dir)
        assert len(proc.processed_files) == 2
