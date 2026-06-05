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


def _b64_image() -> SimpleNamespace:
    import base64

    return SimpleNamespace(image_base64=base64.b64encode(_PNG_1x1).decode(), id="fig.png")


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

    body = (tmp_path / "out" / "sample" / "sample.md").read_text()
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
    doc_meta = json.loads((tmp_path / "out" / "broken" / "metadata.json").read_text())
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
    body = (tmp_path / "out" / "withfig" / "withfig.md").read_text()
    assert "figures/figure_1_page1.png" in body
