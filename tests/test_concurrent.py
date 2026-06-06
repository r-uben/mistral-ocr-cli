"""Tests for concurrent file processing via the RunOutcome contract."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


def _make_ocr_response(text="OCR text"):
    return SimpleNamespace(
        pages=[SimpleNamespace(index=0, markdown=text, images=[])],
    )


def _make_processor(max_workers=1):
    proc = OCRProcessor.__new__(OCRProcessor)
    proc.config = Config(
        api_key="test",
        max_workers=max_workers,
        include_images=False,
    )
    proc.client = MagicMock()
    proc.client.ocr.process.return_value = _make_ocr_response()
    proc._lock = threading.Lock()
    return proc


class TestConcurrentProcessing:
    def test_sequential_processes_all_files(self, tmp_path):
        """Workers=1 processes files sequentially; RunOutcome tallies them all."""
        proc = _make_processor(max_workers=1)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        outcome = proc.process(input_dir)
        assert outcome.completed == 3
        assert outcome.failed == 0
        assert outcome.exit_code == 0
        assert len(outcome.outputs) == 3

    def test_concurrent_processes_all_files(self, tmp_path):
        """Workers>1 processes files concurrently and gets the same results."""
        proc = _make_processor(max_workers=3)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        outcome = proc.process(input_dir)
        assert outcome.completed == 3
        assert outcome.failed == 0
        assert outcome.exit_code == 0

    def test_concurrent_handles_failures(self, tmp_path):
        """One failing file → recorded failed, others complete, nonzero exit."""
        from mistral_ocr.processor import OCRResult

        proc = _make_processor(max_workers=2)
        original = proc.process_file

        def patched(file_path):
            if file_path.name == "b.png":
                return OCRResult(file_path=file_path, pages=[], error="test error")
            return original(file_path)

        proc.process_file = patched

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        outcome = proc.process(input_dir)
        assert outcome.completed == 2
        assert outcome.failed == 1
        assert outcome.exit_code != 0

    def test_concurrent_uses_multiple_threads(self, tmp_path):
        """Concurrent mode dispatches OCR calls off the main thread."""
        seen_threads = set()

        def track_thread(**kwargs):
            seen_threads.add(threading.current_thread().name)
            return _make_ocr_response()

        proc = _make_processor(max_workers=3)
        proc.client.ocr.process.side_effect = track_thread

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        outcome = proc.process(input_dir)
        assert outcome.completed == 3
        # Worker threads (not MainThread) handled the OCR calls.
        assert any(t != "MainThread" for t in seen_threads)


class TestWorkersConfig:
    def test_default_workers(self):
        config = Config(api_key="test")
        assert config.max_workers == 1

    def test_workers_from_env(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        monkeypatch.setenv("MAX_WORKERS", "4")
        config = Config.from_env()
        assert config.max_workers == 4

    def test_workers_minimum_is_one(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        monkeypatch.setenv("MAX_WORKERS", "0")
        config = Config.from_env()
        assert config.max_workers == 1
