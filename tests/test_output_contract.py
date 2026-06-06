"""Engine-level conformance: mistral's REAL output vs the shared contract.

The contract *primitives* (path/key computation, page assembly, metadata
writers, the exit-code policy) are unit-tested inside the ``ocr-output-contract``
package itself, so they are NOT re-tested here.

What stays here is the engine-side proof: run mistral's actual processor with a
mocked Mistral API (no key, no network) over real inputs and assert the produced
output tree conforms to the family-wide contract via the package's reusable
:func:`ocr_output_contract.conformance.assert_conforms` harness — including the
cases that motivated the canon rollout:

* multi-page document → ONE ``<stem>/<stem>.md`` with ``## Page 1`` + ``## Page 2``
  (the API's per-page list is fed into ``assemble_pages``, never dumped whole);
* a failure → recorded ``status="failed"`` with a nonzero exit (no silent drop),
  replacing mistral's old list-based ``errors`` schema;
* a nested batch with same-basename inputs → both survive under input-relative
  keys (the old basename/path keying lost one);
* embedded images → ``figures/figure_<N>_page<P>.png`` with resolving links.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from ocr_output_contract import (
    METADATA_FILENAME,
    UNREADABLE_CHECKSUM,
    doc_dir_for,
    markdown_path_for,
)
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor

# A valid 1x1 PNG so Pillow can open + re-encode an embedded figure.
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00"
    b"\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _page(index: int, markdown: str, images: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(index=index, markdown=markdown, images=images or [])


def _b64_image(image_id: str = "fig.png") -> SimpleNamespace:
    import base64

    return SimpleNamespace(image_base64=base64.b64encode(_PNG_1x1).decode(), id=image_id)


class FakeMistralProcessor:
    """Build an OCRProcessor whose ocr.process call is a deterministic stub.

    No Mistral client is constructed and no key is needed: the client attribute
    is a stub whose ``ocr.process`` returns whatever ``responder`` yields.
    """

    @staticmethod
    def make(responder, **config_kwargs) -> OCRProcessor:
        proc = OCRProcessor.__new__(OCRProcessor)
        proc.config = Config(api_key="test", **config_kwargs)
        proc._lock = threading.Lock()

        class _Client:
            class ocr:  # noqa: N801 - mirrors the SDK attribute path
                @staticmethod
                def process(**kwargs):
                    return responder(kwargs)

            class files:  # noqa: N801
                @staticmethod
                def upload(**kwargs):
                    return SimpleNamespace(id="upload-1")

                @staticmethod
                def delete(**kwargs):
                    return None

        proc.client = _Client()
        return proc


def test_multipage_conforms_no_whole_doc_dump(tmp_path: Path) -> None:
    """A multi-page response → one <stem>/<stem>.md with both pages present."""
    img = tmp_path / "sample.png"  # image input → single API call, multi-page resp
    img.write_bytes(_PNG_1x1)

    def responder(_kwargs):
        return SimpleNamespace(pages=[_page(0, "PAGE-1-CONTENT"), _page(1, "PAGE-2-CONTENT")])

    proc = FakeMistralProcessor.make(responder, include_images=False)
    outcome = proc.process(img, output_path=tmp_path / "out")
    assert outcome.exit_code == 0

    assert_conforms(
        tmp_path / "out",
        [ExpectedDoc(rel_key="sample.png", pages=2, status="completed")],
        require_failures_nonzero_exit=outcome.exit_code != 0,
    )

    doc_dir = doc_dir_for(tmp_path / "out", "sample.png")
    body = markdown_path_for(doc_dir, "sample.png").read_text()
    assert "## Page 1" in body and "## Page 2" in body
    # Both pages survived — not just the first (no whole-doc dump / data loss).
    assert "PAGE-1-CONTENT" in body and "PAGE-2-CONTENT" in body


def test_failure_recorded_and_nonzero_exit(tmp_path: Path) -> None:
    """An API failure → status=failed recorded, nonzero exit (no silent drop)."""
    img = tmp_path / "broken.png"
    img.write_bytes(_PNG_1x1)

    def responder(_kwargs):
        raise RuntimeError("API down")

    proc = FakeMistralProcessor.make(responder, include_images=False)
    outcome = proc.process(img, output_path=tmp_path / "out")
    assert outcome.exit_code != 0

    assert_conforms(
        tmp_path / "out",
        [ExpectedDoc(rel_key="broken.png", status="failed")],
        require_failures_nonzero_exit=True,
    )
    doc_dir = doc_dir_for(tmp_path / "out", "broken.png")
    doc_meta = json.loads((doc_dir / METADATA_FILENAME).read_text())
    assert doc_meta["status"] == "failed"
    assert "error" in doc_meta


def test_empty_response_is_failure(tmp_path: Path) -> None:
    """An API call returning zero pages is a FAILURE, not a 0-byte success."""
    img = tmp_path / "blank.png"
    img.write_bytes(_PNG_1x1)

    proc = FakeMistralProcessor.make(lambda _k: SimpleNamespace(pages=[]), include_images=False)
    outcome = proc.process(img, output_path=tmp_path / "out")
    assert outcome.exit_code != 0

    assert_conforms(
        tmp_path / "out",
        [ExpectedDoc(rel_key="blank.png", status="failed")],
        require_failures_nonzero_exit=True,
    )


def test_nested_batch_no_basename_collision(tmp_path: Path) -> None:
    """Two same-basename inputs in different subdirs both survive (rel-key)."""
    root = tmp_path / "in"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "intro.png").write_bytes(_PNG_1x1)
    (root / "b" / "intro.png").write_bytes(_PNG_1x1)

    proc = FakeMistralProcessor.make(
        lambda _k: SimpleNamespace(pages=[_page(0, "content")]), include_images=False
    )
    outcome = proc.process(root, output_path=tmp_path / "out")
    assert outcome.exit_code == 0

    assert_conforms(
        tmp_path / "out",
        [
            ExpectedDoc(rel_key="a/intro.png", pages=1, status="completed"),
            ExpectedDoc(rel_key="b/intro.png", pages=1, status="completed"),
        ],
    )


def test_embedded_figures_conform(tmp_path: Path) -> None:
    """Embedded images → figures/figure_<N>_page<P>.png with resolving links."""
    img = tmp_path / "withfig.png"
    img.write_bytes(_PNG_1x1)

    def responder(_kwargs):
        return SimpleNamespace(pages=[_page(0, "see figure", images=[_b64_image()])])

    proc = FakeMistralProcessor.make(responder, include_images=True)
    outcome = proc.process(img, output_path=tmp_path / "out")
    assert outcome.exit_code == 0

    assert_conforms(
        tmp_path / "out",
        [ExpectedDoc(rel_key="withfig.png", pages=1, status="completed", figures=[(1, 1)])],
    )
    doc_dir = doc_dir_for(tmp_path / "out", "withfig.png")
    body = markdown_path_for(doc_dir, "withfig.png").read_text()
    assert "figures/figure_1_page1.png" in body


def test_embedded_image_placeholders_resolve_on_disk(tmp_path: Path) -> None:
    """Byte-faithful real Mistral output → every inline image link resolves.

    Round-2 HIGH blocker: real Mistral ``page.markdown`` carries inline
    placeholders like ``![img-0.jpeg](img-0.jpeg)`` whose target is the API's
    ``page.images[i].id``. The old code kept ``page.markdown`` verbatim AND
    appended a separate ``./figures/...`` link, leaving the original ``img-0.jpeg``
    as a DANGLING local link that the v0.1.2 conformance harness resolves on disk
    and rejects ("dangling inline image link"). This fixture reproduces that real
    shape (markdown placeholder + matching image id); the processor must rewrite
    the placeholder in place to the canonical figure file so the produced body
    has NO dangling link and assert_conforms passes.
    """
    img = tmp_path / "paper.png"
    img.write_bytes(_PNG_1x1)

    def responder(_kwargs):
        # The REAL Mistral shape: an inline ![img-0.jpeg](img-0.jpeg) placeholder
        # whose target equals page.images[0].id. (The old happy-path fixture had
        # no placeholder, which is exactly why it could not catch this bug.)
        return SimpleNamespace(
            pages=[
                _page(
                    0,
                    "Intro text.\n\n![img-0.jpeg](img-0.jpeg)\n\nTrailing text.",
                    images=[_b64_image("img-0.jpeg")],
                )
            ]
        )

    proc = FakeMistralProcessor.make(responder, include_images=True)
    outcome = proc.process(img, output_path=tmp_path / "out")
    assert outcome.exit_code == 0

    # The conformance harness resolves EVERY inline ![..](target) on disk; this
    # would have FAILED before the placeholder-rewrite fix.
    assert_conforms(
        tmp_path / "out",
        [ExpectedDoc(rel_key="paper.png", pages=1, status="completed", figures=[(1, 1)])],
    )

    body = markdown_path_for(doc_dir_for(tmp_path / "out", "paper.png"), "paper.png").read_text()
    # The placeholder was rewritten IN PLACE to the canonical figure file...
    assert "![img-0.jpeg](./figures/figure_1_page1.png)" in body
    # ...and the dangling local link is GONE (no duplicate appended link either).
    assert "(img-0.jpeg)" not in body
    assert body.count("figure_1_page1.png") == 1
    assert (doc_dir_for(tmp_path / "out", "paper.png") / "figures" / "figure_1_page1.png").exists()


def test_unwritten_placeholder_is_stripped_not_dangling(tmp_path: Path) -> None:
    """An inline placeholder with no extractable image is stripped, not dangling.

    If the API emits an ``![missing.jpeg](missing.jpeg)`` placeholder but no
    decodable image bytes for it, there is no figure to point at. The body must
    not ship a dangling local link; the placeholder is stripped (alt text kept)
    so conformance still passes.
    """
    img = tmp_path / "ghost.png"
    img.write_bytes(_PNG_1x1)

    def responder(_kwargs):
        # Placeholder present in markdown but NO images list → nothing to resolve.
        return SimpleNamespace(
            pages=[_page(0, "Body.\n\n![missing.jpeg](missing.jpeg)\n\nEnd.", images=[])]
        )

    proc = FakeMistralProcessor.make(responder, include_images=True)
    outcome = proc.process(img, output_path=tmp_path / "out")
    assert outcome.exit_code == 0

    assert_conforms(
        tmp_path / "out",
        [ExpectedDoc(rel_key="ghost.png", pages=1, status="completed")],
    )
    body = markdown_path_for(doc_dir_for(tmp_path / "out", "ghost.png"), "ghost.png").read_text()
    assert "(missing.jpeg)" not in body


def test_unreadable_input_recorded_failed_batch_continues(tmp_path: Path) -> None:
    """SYS-02: an input unreadable at the pre-check is failed, batch continues.

    safe_checksum returns None for an unreadable file instead of raising an
    OSError that would abort the whole directory run with zero output. The bad
    file is recorded status=failed (nonzero exit) and the good files still
    process and conform.
    """
    import os
    import stat

    root = tmp_path / "in"
    root.mkdir()
    good = root / "good.png"
    bad = root / "bad.png"
    good.write_bytes(_PNG_1x1)
    bad.write_bytes(_PNG_1x1)
    # Make the input unreadable so safe_checksum cannot hash it.
    os.chmod(bad, 0)
    # chmod(0) does not deny reads when running as root (e.g. some CI); capture
    # the effective readability so the sentinel assertion only fires when the
    # input is genuinely unreadable.
    bad_was_unreadable = not os.access(bad, os.R_OK)

    try:
        proc = FakeMistralProcessor.make(
            lambda _k: SimpleNamespace(pages=[_page(0, "content")]), include_images=False
        )
        outcome = proc.process(root, output_path=tmp_path / "out")

        # The batch did NOT abort: good file completed, bad file failed.
        assert outcome.completed == 1
        assert outcome.failed == 1
        assert outcome.exit_code != 0
        assert "bad.png" in outcome.failures

        # The unreadable input is durably recorded status=failed in the root index.
        root_index = json.loads((tmp_path / "out" / "metadata.json").read_text())
        assert root_index["files"]["bad.png"]["status"] == "failed"
        assert root_index["files"]["good.png"]["status"] == "completed"
        # v0.1.3: the failure record must carry a VALID ``sha256:`` checksum
        # (failure_checksum), never the old bare ``"sha256:"`` (no digest). When
        # the input is genuinely unreadable it is the UNREADABLE_CHECKSUM sentinel.
        bad_checksum = root_index["files"]["bad.png"]["checksum"]
        assert bad_checksum.startswith("sha256:")
        assert bad_checksum != "sha256:"
        if bad_was_unreadable:
            assert bad_checksum == UNREADABLE_CHECKSUM
    finally:
        os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)


def test_fingerprint_keys_extract_flags_via_extra(tmp_path: Path) -> None:
    """A re-run toggling an OCR-3 extract flag reprocesses (extra-keyed fp).

    The output-affecting flags (table_format, extract_header/footer,
    include_images) are passed to run_fingerprint via the v0.1.2 ``extra`` dict,
    so flipping --extract-headers on a re-run changes the fingerprint and forces
    reprocessing rather than serving a stale result keyed only on the checksum.
    """
    img = tmp_path / "fp2.png"
    img.write_bytes(_PNG_1x1)
    calls = {"n": 0}

    def responder(_kwargs):
        calls["n"] += 1
        return SimpleNamespace(pages=[_page(0, "content")])

    out = tmp_path / "out"
    FakeMistralProcessor.make(responder, include_images=False).process(img, output_path=out)
    assert calls["n"] == 1

    # Same input + checksum, but extract_header now on → different fingerprint.
    proc2 = FakeMistralProcessor.make(responder, include_images=False, extract_header=True)
    proc2.process(img, output_path=out)
    assert calls["n"] == 2  # reprocessed, not silently reused


def test_checksum_resume_skip_emits_output_path(tmp_path: Path) -> None:
    """A checksum-resumed (skipped-but-valid) doc still reports its .md path.

    The quiet scripting contract is one .md path per line on stdout, *including*
    docs skipped on resume. The skip branch previously called add(COMPLETED)
    without an output_path, so a rerun emitted nothing. This asserts the path is
    reported AND that the API is not called again on the second run.
    """
    img = tmp_path / "resume.png"
    img.write_bytes(_PNG_1x1)

    calls = {"n": 0}

    def responder(_kwargs):
        calls["n"] += 1
        return SimpleNamespace(pages=[_page(0, "content")])

    out = tmp_path / "out"
    proc1 = FakeMistralProcessor.make(responder, include_images=False)
    first = proc1.process(img, output_path=out)
    assert first.exit_code == 0
    assert calls["n"] == 1
    expected_md = str(markdown_path_for(doc_dir_for(out, "resume.png"), "resume.png"))
    assert first.outputs == [expected_md]

    # Second run: a fresh processor (fresh RootIndex load from disk) resumes.
    proc2 = FakeMistralProcessor.make(responder, include_images=False)
    second = proc2.process(img, output_path=out)
    assert second.exit_code == 0
    # The API was NOT called again (checksum + fingerprint + on-disk skip).
    assert calls["n"] == 1
    # ...yet the skipped-but-valid doc STILL reports its .md path for scripting.
    assert second.outputs == [expected_md]


def test_changed_fingerprint_reprocesses(tmp_path: Path) -> None:
    """A re-run under a different run config (table_format) reprocesses, not skips."""
    img = tmp_path / "fp.png"
    img.write_bytes(_PNG_1x1)

    calls = {"n": 0}

    def responder(_kwargs):
        calls["n"] += 1
        return SimpleNamespace(pages=[_page(0, "content")])

    out = tmp_path / "out"
    FakeMistralProcessor.make(responder, include_images=False).process(img, output_path=out)
    assert calls["n"] == 1

    # Same input + checksum, but a different OCR-3 config → different fingerprint.
    proc2 = FakeMistralProcessor.make(responder, include_images=False, table_format="markdown")
    proc2.process(img, output_path=out)
    assert calls["n"] == 2  # reprocessed, not silently reused


def test_persistence_failure_does_not_abort_batch(tmp_path: Path) -> None:
    """A save/metadata I/O failure on one doc is recorded failed; batch continues.

    Persistence failures previously escaped to the CLI and aborted the whole
    batch with no status=failed record. They must instead be folded into a
    failed RunOutcome entry while the remaining files still process.
    """
    root = tmp_path / "in"
    root.mkdir()
    for name in ("a.png", "boom.png", "c.png"):
        (root / name).write_bytes(_PNG_1x1)

    proc = FakeMistralProcessor.make(
        lambda _k: SimpleNamespace(pages=[_page(0, "content")]), include_images=False
    )

    # Make save_results raise for exactly one document; the others persist fine.
    real_save = proc.save_results

    def flaky_save(result, output_root, rel_key):
        if Path(rel_key).name == "boom.png":
            raise OSError("disk full")
        return real_save(result, output_root, rel_key)

    proc.save_results = flaky_save

    outcome = proc.process(root, output_path=tmp_path / "out")

    # Batch did NOT abort: the two good files completed, the bad one is failed.
    assert outcome.completed == 2
    assert outcome.failed == 1
    assert outcome.exit_code != 0
    assert "boom.png" in outcome.failures

    # The failure is durably recorded (status=failed) in the root index, and the
    # failed placeholder path is NOT echoed to the quiet stdout list.
    root_index = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert root_index["files"]["boom.png"]["status"] == "failed"
    failed_md = str(markdown_path_for(doc_dir_for(tmp_path / "out", "boom.png"), "boom.png"))
    assert failed_md not in outcome.outputs
    assert len(outcome.outputs) == 2
