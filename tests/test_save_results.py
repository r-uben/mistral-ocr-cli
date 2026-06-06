"""Tests for mistral-specific page rendering and figure extraction.

The output *shape* (layout, page assembly, metadata, figure naming) is owned by
``ocr-output-contract`` and is exercised end-to-end in
``tests/test_output_contract.py`` via the package's conformance harness. What
stays here is the mistral-owned glue: how one API page's OCR 3 extras
(dimensions, header/footer, hyperlinks) fold into a single page body, and how
embedded base64 images become canonically-named PNG figures with resolving
links.

These call the real :class:`OCRProcessor` methods directly with a stub config
(no Mistral client), so they need no API key.
"""

from types import SimpleNamespace

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor, OCRResult


def _proc(**config_kwargs):
    """An OCRProcessor with a stub config and no live Mistral client."""
    proc = OCRProcessor.__new__(OCRProcessor)
    proc.config = Config(api_key="test", **config_kwargs)
    return proc


def _page(index=0, markdown="text", **kwargs):
    kwargs.setdefault("images", [])
    return SimpleNamespace(index=index, markdown=markdown, **kwargs)


# ---------------------------------------------------------------------------
# Page body rendering (_render_page_markdown): OCR 3 extras folded inline
# ---------------------------------------------------------------------------


class TestPageDimensions:
    def test_dimensions_rendered(self):
        page = _page(dimensions=SimpleNamespace(width=612, height=792))
        body = _proc()._render_page_markdown(page)
        assert "612 x 792" in body

    def test_no_dimensions_when_absent(self):
        body = _proc()._render_page_markdown(_page())
        assert "Page size" not in body


class TestHeaderFooter:
    def test_header_rendered(self):
        body = _proc()._render_page_markdown(_page(header="Chapter 1"))
        assert "> **Header:** Chapter 1" in body

    def test_footer_rendered(self):
        body = _proc()._render_page_markdown(_page(footer="Page 1 of 10"))
        assert "> **Footer:** Page 1 of 10" in body

    def test_no_header_when_empty(self):
        body = _proc()._render_page_markdown(_page(header=""))
        assert "**Header:**" not in body


class TestHyperlinks:
    def test_hyperlinks_rendered(self):
        link = SimpleNamespace(text="Example", url="https://example.com")
        body = _proc()._render_page_markdown(_page(hyperlinks=[link]))
        assert "**Hyperlinks:**" in body
        assert "[Example](https://example.com)" in body

    def test_hyperlink_without_text(self):
        link = SimpleNamespace(text="", url="https://example.com")
        body = _proc()._render_page_markdown(_page(hyperlinks=[link]))
        assert "[https://example.com](https://example.com)" in body

    def test_hyperlink_with_href_fallback(self):
        link = SimpleNamespace(text="Link", href="https://test.com")  # no 'url' attr
        body = _proc()._render_page_markdown(_page(hyperlinks=[link]))
        assert "[Link](https://test.com)" in body


class TestPageBodyIsCleanBody:
    def test_body_has_no_page_header(self):
        """The page body must NOT carry its own ## Page header.

        The contract's assemble_pages adds the canonical ``## Page N`` header,
        so a doubled header would violate the marker count. The renderer emits
        only the page body.
        """
        body = _proc()._render_page_markdown(_page(markdown="hello"))
        assert "## Page" not in body
        assert body.strip() == "hello"


def _table(tid="tbl-0", content="| A | B |\n|---|---|\n| 1 | 2 |", fmt="markdown"):
    return SimpleNamespace(id=tid, content=content, format_=fmt)


class TestStructuredTables:
    """Regression guard: page.tables must NOT be silently discarded.

    With --table-format=markdown|html Mistral returns tables in the structured
    page.tables field (NOT inline in page.markdown), so a renderer that ignores
    page.tables makes --table-format a silent no-op and drops OCR-3 table
    extraction. These assert the structured tables land in the rendered body.
    """

    def test_table_appended_when_no_placeholder(self):
        page = _page(markdown="Body text", tables=[_table()])
        body = _proc(table_format="markdown")._render_page_markdown(page)
        assert "Body text" in body
        # The table content survived (not dropped).
        assert "| A | B |" in body and "| 1 | 2 |" in body

    def test_table_replaces_inline_placeholder(self):
        """A markdown placeholder referencing the table id is replaced in place."""
        page = _page(
            markdown="See table below:\n\n[tbl-0](tbl-0)\n\nrest",
            tables=[_table(tid="tbl-0", content="| X |\n|---|\n| 9 |")],
        )
        body = _proc(table_format="markdown")._render_page_markdown(page)
        assert "[tbl-0](tbl-0)" not in body  # placeholder consumed
        assert "| X |" in body and "| 9 |" in body
        assert "rest" in body

    def test_html_table_passed_through(self):
        page = _page(
            markdown="Body",
            tables=[_table(content="<table><tr><td>v</td></tr></table>", fmt="html")],
        )
        body = _proc(table_format="html")._render_page_markdown(page)
        assert "<table><tr><td>v</td></tr></table>" in body

    def test_multiple_tables_all_preserved(self):
        page = _page(
            markdown="Body",
            tables=[_table(tid="t1", content="TABLE-ONE"), _table(tid="t2", content="TABLE-TWO")],
        )
        body = _proc(table_format="markdown")._render_page_markdown(page)
        assert "TABLE-ONE" in body and "TABLE-TWO" in body

    def test_no_tables_is_noop(self):
        """When --table-format is unset the API inlines tables; render is a no-op."""
        page = _page(markdown="inline | table | here")
        body = _proc()._render_page_markdown(page)
        assert body.strip() == "inline | table | here"

    def test_empty_tables_list_is_noop(self):
        page = _page(markdown="just text", tables=[])
        body = _proc(table_format="markdown")._render_page_markdown(page)
        assert body.strip() == "just text"


# ---------------------------------------------------------------------------
# Figure extraction (_save_figures): canonical figure_<N>_page<P>.png naming
# ---------------------------------------------------------------------------

# A valid 1x1 PNG so Pillow can open + re-encode it.
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00"
    b"\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _result_with_image(tmp_path, page_no=1, image_bytes=_PNG_1x1, image_id="img-0.jpeg"):
    fp = tmp_path / "doc.png"
    fp.write_bytes(_PNG_1x1)
    return OCRResult(
        file_path=fp,
        pages=["text"],
        # page_images now carries (image_id, raw_bytes) pairs so the inline
        # placeholder ![..](image_id) can be rewritten to the canonical name.
        page_images={page_no: [(image_id, image_bytes)]},
    )


class TestFigures:
    def test_image_saved_with_canonical_name_and_link(self, tmp_path):
        proc = _proc(include_images=True)
        doc_dir = tmp_path / "doc"
        doc_dir.mkdir()
        result = _result_with_image(tmp_path, page_no=1)

        saved = proc._save_figures(result, doc_dir)

        # Canonical naming: figure_<N>_page<P>.png (NOT page1_img1.png).
        assert (doc_dir / "figures" / "figure_1_page1.png").exists()
        assert not (doc_dir / "figures" / "page1_img1.png").exists()
        # Returns a saved-figure record carrying the API image id + canonical
        # number/page for the page that produced it.
        rec = saved[1][0]
        assert rec.image_id == "img-0.jpeg"
        assert rec.figure_number == 1
        assert rec.page_number == 1

    def test_figure_numbering_is_global_with_source_page(self, tmp_path):
        """Figures are numbered globally across the doc, tagged by source page."""
        proc = _proc(include_images=True)
        doc_dir = tmp_path / "doc"
        doc_dir.mkdir()
        result = OCRResult(
            file_path=tmp_path / "doc.png",
            pages=["a", "b"],
            page_images={
                1: [("img-0.jpeg", _PNG_1x1)],
                2: [("img-1.jpeg", _PNG_1x1), ("img-2.jpeg", _PNG_1x1)],
            },
        )
        (tmp_path / "doc.png").write_bytes(_PNG_1x1)

        proc._save_figures(result, doc_dir)
        figs = sorted(p.name for p in (doc_dir / "figures").iterdir())
        assert figs == [
            "figure_1_page1.png",
            "figure_2_page2.png",
            "figure_3_page2.png",
        ]

    def test_no_figures_when_images_disabled(self, tmp_path):
        proc = _proc(include_images=False)
        doc_dir = tmp_path / "doc"
        doc_dir.mkdir()
        result = _result_with_image(tmp_path, page_no=1)

        links = proc._save_figures(result, doc_dir)
        assert links == {}
        assert not (doc_dir / "figures").exists()


# ---------------------------------------------------------------------------
# Truncation note (folded into the body by save_results, recorded in metadata)
# ---------------------------------------------------------------------------


class TestTruncationNote:
    def test_note_prepended_to_body(self, tmp_path):
        proc = _proc(include_images=False, verbose=False)
        fp = tmp_path / "doc.png"
        fp.write_bytes(_PNG_1x1)
        result = OCRResult(
            file_path=fp,
            pages=["page one"],
            note="Processed 50 of 200 pages (--max-pages)",
        )
        md_path = proc.save_results(result, tmp_path / "out", "doc.png")
        body = md_path.read_text()
        assert "> **Note:** Processed 50 of 200 pages" in body
        assert "## Page 1" in body  # contract still owns page headers
