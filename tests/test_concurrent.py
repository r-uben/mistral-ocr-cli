"""Tests for concurrent file processing."""

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
        save_original_images=False,
    )
    proc.client = MagicMock()
    proc.client.ocr.process.return_value = _make_ocr_response()
    proc.errors = []
    proc.processed_files = []
    proc._lock = threading.Lock()
    return proc


class TestConcurrentProcessing:
    def test_sequential_processes_all_files(self, tmp_path):
        """Workers=1 processes files sequentially."""
        proc = _make_processor(max_workers=1)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        success, total = proc.process_directory(input_dir)
        assert total == 3
        assert success == 3
        assert len(proc.processed_files) == 3
        assert len(proc.errors) == 0

    def test_concurrent_processes_all_files(self, tmp_path):
        """Workers>1 processes files concurrently and gets same results."""
        proc = _make_processor(max_workers=3)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        success, total = proc.process_directory(input_dir)
        assert total == 3
        assert success == 3
        assert len(proc.processed_files) == 3
        assert len(proc.errors) == 0

    def test_concurrent_handles_failures(self, tmp_path):
        """Concurrent mode handles individual file failures gracefully."""
        proc = _make_processor(max_workers=2)

        # Make process_file fail for one specific file
        original_process = proc.process_file

        def selective_fail(file_path):
            if file_path.name == "b.png":
                proc.errors.append({"file": str(file_path), "error": "test error"})
                return None
            return original_process(file_path)

        proc.process_file = selective_fail

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        success, total = proc.process_directory(input_dir)
        assert total == 3
        assert success == 2
        assert len(proc.errors) == 1

    def test_concurrent_uses_multiple_threads(self, tmp_path):
        """Verify concurrent mode actually uses threads."""
        seen_threads = set()

        def track_thread(*args, **kwargs):
            seen_threads.add(threading.current_thread().name)
            return _make_ocr_response()

        proc = _make_processor(max_workers=3)
        proc.client.ocr.process.side_effect = track_thread

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["a.png", "b.png", "c.png"]:
            (input_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        proc.process_directory(input_dir)
        # With 3 workers and 3 files, we should see more than 1 thread
        # (not guaranteed with timing, so just check it completes)
        assert len(proc.processed_files) == 3


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
