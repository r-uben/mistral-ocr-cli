"""Tests for the OCR 4 blocks / confidence feature (issue #23).

Covers the four things that can go wrong independently of the API: the kwargs
we send, the fingerprint we key the cache on, the sidecar we write, and the
promise that the markdown body is untouched by any of it.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from ocr_output_contract import doc_dir_for

from mistral_ocr.config import Config
from mistral_ocr.processor import BLOCKS_FILENAME, OCRProcessor


def _config(**kwargs) -> Config:
    return Config(api_key="test-key", **kwargs)


def _processor(**kwargs) -> OCRProcessor:
    with patch("mistral_ocr.processor.Mistral"):
        return OCRProcessor(_config(**kwargs))


def _page(index: int = 0, *, blocks=None, confidence=None) -> SimpleNamespace:
    """A minimal stand-in for OCRPageObject."""
    page = SimpleNamespace(
        index=index,
        markdown=f"Body text for page {index + 1}.",
        images=[],
        dimensions=None,
        tables=None,
        hyperlinks=None,
        header=None,
        footer=None,
    )
    if blocks is not None:
        page.blocks = blocks
    if confidence is not None:
        page.confidence_scores = confidence
    return page


def _block(block_type: str = "text", **extra) -> dict:
    """A block in the shape the live API actually returns.

    Bounding boxes come back as four scalar corner fields, not a ``bbox`` array —
    verified against a real OCR 4.1 response.
    """
    return {
        "type": block_type,
        "top_left_x": 43,
        "top_left_y": 55,
        "bottom_right_x": 125,
        "bottom_right_y": 69,
        "content": "Some content",
        **extra,
    }


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


class TestOcrKwargs:
    def test_flags_absent_by_default(self):
        kwargs = _processor()._build_ocr_kwargs({"type": "file", "file_id": "f"})
        assert "include_blocks" not in kwargs
        assert "confidence_scores_granularity" not in kwargs

    def test_include_blocks_forwarded(self):
        kwargs = _processor(include_blocks=True)._build_ocr_kwargs({})
        assert kwargs["include_blocks"] is True

    def test_granularity_forwarded(self):
        kwargs = _processor(confidence_scores_granularity="block")._build_ocr_kwargs({})
        assert kwargs["confidence_scores_granularity"] == "block"

    @pytest.mark.parametrize("granularity", ["page", "block", "word"])
    def test_every_granularity_forwarded(self, granularity):
        kwargs = _processor(confidence_scores_granularity=granularity)._build_ocr_kwargs({})
        assert kwargs["confidence_scores_granularity"] == granularity


# ---------------------------------------------------------------------------
# Fingerprint: the cache-invalidation contract
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_defaults_do_not_change_the_fingerprint(self):
        """The regression guard for the whole feature.

        ``run_fingerprint`` hashes the entire ``extra`` dict, so emitting the new
        keys at their defaults would invalidate every previously completed
        document and force a full re-OCR of existing corpora at $4/1k pages.
        This asserts the exact digest a pre-feature run produced.
        """
        baseline = "fp:e1ece34158a0d63d4464880bc4bcc517184904c858d19c70dab0735df2caa39b"
        assert _processor()._run_fingerprint() == baseline

    def test_include_blocks_changes_the_fingerprint(self):
        assert _processor(include_blocks=True)._run_fingerprint() != _processor()._run_fingerprint()

    def test_granularity_changes_the_fingerprint(self):
        assert (
            _processor(confidence_scores_granularity="word")._run_fingerprint()
            != _processor()._run_fingerprint()
        )

    def test_granularity_levels_are_distinct(self):
        assert (
            _processor(confidence_scores_granularity="page")._run_fingerprint()
            != _processor(confidence_scores_granularity="word")._run_fingerprint()
        )

    def test_toggling_back_off_restores_the_original_fingerprint(self):
        """Off must be indistinguishable from never-enabled, or turning the
        feature off would strand every document behind a fingerprint miss."""
        assert (
            _processor(include_blocks=False)._run_fingerprint() == _processor()._run_fingerprint()
        )


# ---------------------------------------------------------------------------
# Capture + sidecar
# ---------------------------------------------------------------------------


class TestSidecar:
    def _run(self, tmp_path: Path, proc: OCRProcessor, pages) -> Path:
        result = proc._parse_response(tmp_path / "doc.pdf", SimpleNamespace(pages=pages), 0.0)
        proc.save_results(result, tmp_path / "out", "doc.pdf")
        return doc_dir_for(tmp_path / "out", "doc.pdf")

    def test_no_sidecar_when_disabled(self, tmp_path):
        doc_dir = self._run(tmp_path, _processor(), [_page(0, blocks=[_block()])])
        assert not (doc_dir / BLOCKS_FILENAME).exists()

    def test_sidecar_written_with_blocks(self, tmp_path):
        blocks = [_block("title", content="Hi")]
        doc_dir = self._run(tmp_path, _processor(include_blocks=True), [_page(0, blocks=blocks)])

        payload = json.loads((doc_dir / BLOCKS_FILENAME).read_text())
        written = payload["pages"][0]["blocks"][0]
        assert payload["pages"][0]["page"] == 1
        assert written["type"] == "title"
        assert (written["top_left_x"], written["bottom_right_y"]) == (43, 69)

    def test_page_numbers_match_the_body_headers(self, tmp_path):
        """Chunked PDFs re-index pages globally; the sidecar must follow, or a
        consumer cannot join it to the ``## Page N`` headers."""
        pages = [_page(i, blocks=[_block(content=str(i))]) for i in (0, 1, 2)]
        doc_dir = self._run(tmp_path, _processor(include_blocks=True), pages)

        payload = json.loads((doc_dir / BLOCKS_FILENAME).read_text())
        assert [p["page"] for p in payload["pages"]] == [1, 2, 3]

        body = (doc_dir / "doc.md").read_text()
        for n in (1, 2, 3):
            assert f"## Page {n}" in body

    def test_confidence_scores_captured(self, tmp_path):
        proc = _processor(confidence_scores_granularity="block")
        doc_dir = self._run(tmp_path, proc, [_page(0, confidence={"block": [0.98, 0.91]})])

        payload = json.loads((doc_dir / BLOCKS_FILENAME).read_text())
        assert payload["pages"][0]["confidence_scores"] == {"block": [0.98, 0.91]}

    def test_base64_payloads_are_stripped(self, tmp_path):
        """Image blocks may echo the bytes the figures pipeline already wrote,
        and the API's image ids are chunk-local. Neither belongs in the sidecar."""
        blocks = [_block("image", image_base64="AAAA" * 100)]
        doc_dir = self._run(tmp_path, _processor(include_blocks=True), [_page(0, blocks=blocks)])

        raw = (doc_dir / BLOCKS_FILENAME).read_text()
        assert "AAAA" not in raw
        payload = json.loads(raw)
        assert payload["pages"][0]["blocks"][0]["top_left_x"] == 43

    def test_stale_sidecar_removed_when_disabled(self, tmp_path):
        """A re-run with blocks off must not leave a sidecar describing a
        different run sitting next to fresh metadata."""
        blocks = [_block()]
        doc_dir = self._run(tmp_path, _processor(include_blocks=True), [_page(0, blocks=blocks)])
        assert (doc_dir / BLOCKS_FILENAME).exists()

        self._run(tmp_path, _processor(), [_page(0, blocks=blocks)])
        assert not (doc_dir / BLOCKS_FILENAME).exists()

    def test_absent_blocks_write_no_sidecar(self, tmp_path):
        """A model that ignores the flag must not yield a file full of nulls."""
        doc_dir = self._run(tmp_path, _processor(include_blocks=True), [_page(0)])
        assert not (doc_dir / BLOCKS_FILENAME).exists()


# ---------------------------------------------------------------------------
# The body-stays-clean contract
# ---------------------------------------------------------------------------


class TestBodyUnchanged:
    def test_body_is_byte_identical_with_and_without_blocks(self, tmp_path):
        blocks = [_block("title", content="Heading")]

        def body_for(proc, out_name):
            pages = [_page(0, blocks=blocks), _page(1, blocks=blocks)]
            result = proc._parse_response(tmp_path / "doc.pdf", SimpleNamespace(pages=pages), 0.0)
            proc.save_results(result, tmp_path / out_name, "doc.pdf")
            return (doc_dir_for(tmp_path / out_name, "doc.pdf") / "doc.md").read_bytes()

        assert body_for(_processor(include_blocks=True), "on") == body_for(_processor(), "off")

    def test_no_blocks_json_inlined_in_the_body(self, tmp_path):
        blocks = [_block("table", content="cell")]
        proc = _processor(include_blocks=True, confidence_scores_granularity="word")
        result = proc._parse_response(
            tmp_path / "doc.pdf",
            SimpleNamespace(pages=[_page(0, blocks=blocks, confidence={"word": [0.5]})]),
            0.0,
        )
        proc.save_results(result, tmp_path / "out", "doc.pdf")

        body = (doc_dir_for(tmp_path / "out", "doc.pdf") / "doc.md").read_text()
        for token in ("top_left_x", "confidence", '"type"', "0.5"):
            assert token not in body
