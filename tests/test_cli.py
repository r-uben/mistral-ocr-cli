"""Tests for CLI flags (dry-run, quiet)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mistral_ocr.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def _make_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


class TestDryRun:
    def test_dry_run_single_file(self, runner, tmp_path, monkeypatch):
        _make_env(monkeypatch)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")

        result = runner.invoke(main, [str(pdf), "--dry-run"])
        assert result.exit_code == 0
        assert "doc.pdf" in result.output
        assert "dry run" in result.output

    def test_dry_run_directory(self, runner, tmp_path, monkeypatch):
        _make_env(monkeypatch)
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "b.png").write_bytes(b"\x89PNG")
        (tmp_path / "c.txt").write_bytes(b"text")  # not supported

        result = runner.invoke(main, [str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "a.pdf" in result.output
        assert "b.png" in result.output
        assert "c.txt" not in result.output
        assert "2 file(s) would be processed" in result.output

    def test_dry_run_empty_directory(self, runner, tmp_path, monkeypatch):
        _make_env(monkeypatch)
        result = runner.invoke(main, [str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "No supported files" in result.output

    def test_dry_run_does_not_call_api(self, runner, tmp_path, monkeypatch):
        _make_env(monkeypatch)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")

        with patch("mistral_ocr.cli.OCRProcessor") as mock:
            result = runner.invoke(main, [str(pdf), "--dry-run"])
            assert result.exit_code == 0
            mock.assert_not_called()


class TestQuiet:
    def test_quiet_suppresses_output(self, runner, tmp_path, monkeypatch):
        _make_env(monkeypatch)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")

        with patch("mistral_ocr.cli.OCRProcessor") as mock_cls:
            mock_proc = mock_cls.return_value
            mock_proc.errors = []
            mock_proc.process.return_value = None

            result = runner.invoke(main, [str(pdf), "--quiet"])
            assert result.exit_code == 0
            # Quiet mode: no banner, no completion message
            assert "Mistral OCR" not in result.output
            assert "Processing complete" not in result.output
