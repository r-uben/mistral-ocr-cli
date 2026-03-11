"""Tests for save_results rendering: OCR 3 features, images, truncation, originals."""

import base64
from types import SimpleNamespace

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor


def _make_processor(**config_kwargs):
    proc = OCRProcessor.__new__(OCRProcessor)
    config_kwargs.setdefault("save_original_images", False)
    proc.config = Config(api_key="test", **config_kwargs)
    proc.errors = []
    proc.processed_files = []
    return proc


def _page(index=0, markdown="text", **kwargs):
    return SimpleNamespace(index=index, markdown=markdown, images=[], **kwargs)


def _result(tmp_path, response, name="doc.pdf"):
    fp = tmp_path / name
    fp.write_bytes(b"%PDF-1.4 fake")
    return {"file_path": fp, "response": response}


# ---------------------------------------------------------------------------
# Original file copy
# ---------------------------------------------------------------------------


class TestSaveOriginals:
    def test_copies_original_when_enabled(self, tmp_path):
        proc = _make_processor(save_original_images=True)
        out = tmp_path / "out"
        out.mkdir()
        res = _result(tmp_path, SimpleNamespace(pages=[_page()]))
        proc.save_results(res, out)
        assert (out / "doc" / "doc.pdf").exists()

    def test_skips_copy_when_disabled(self, tmp_path):
        proc = _make_processor(save_original_images=False)
        out = tmp_path / "out"
        out.mkdir()
        res = _result(tmp_path, SimpleNamespace(pages=[_page()]))
        proc.save_results(res, out)
        assert not (out / "doc" / "doc.pdf").exists()

    def test_original_link_in_metadata(self, tmp_path):
        proc = _make_processor(save_original_images=True, include_metadata=True)
        out = tmp_path / "out"
        out.mkdir()
        res = _result(tmp_path, SimpleNamespace(pages=[_page()]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "**Original:**" in md
        assert "doc.pdf" in md


# ---------------------------------------------------------------------------
# Truncation note
# ---------------------------------------------------------------------------


class TestTruncationNote:
    def test_truncation_note_rendered(self, tmp_path):
        proc = _make_processor(include_metadata=True)
        out = tmp_path / "out"
        out.mkdir()
        response = SimpleNamespace(
            pages=[_page()], truncated="Processed 50 of 200 pages (--max-pages)"
        )
        res = _result(tmp_path, response)
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "**Note:** Processed 50 of 200" in md

    def test_no_truncation_note_when_absent(self, tmp_path):
        proc = _make_processor(include_metadata=True)
        out = tmp_path / "out"
        out.mkdir()
        res = _result(tmp_path, SimpleNamespace(pages=[_page()]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "**Note:**" not in md


# ---------------------------------------------------------------------------
# Page dimensions (OCR 3)
# ---------------------------------------------------------------------------


class TestPageDimensions:
    def test_dimensions_rendered(self, tmp_path):
        proc = _make_processor(include_page_headings=True)
        out = tmp_path / "out"
        out.mkdir()
        dims = SimpleNamespace(width=612, height=792)
        page = _page(dimensions=dims)
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "612 x 792" in md

    def test_no_dimensions_when_absent(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        res = _result(tmp_path, SimpleNamespace(pages=[_page()]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "Page size" not in md


# ---------------------------------------------------------------------------
# Headers and footers (OCR 3)
# ---------------------------------------------------------------------------


class TestHeaderFooter:
    def test_header_rendered(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        page = _page(header="Chapter 1")
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "> **Header:** Chapter 1" in md

    def test_footer_rendered(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        page = _page(footer="Page 1 of 10")
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "> **Footer:** Page 1 of 10" in md

    def test_no_header_when_empty(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        page = _page(header="")
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "**Header:**" not in md


# ---------------------------------------------------------------------------
# Tables (OCR 3)
# ---------------------------------------------------------------------------


class TestTables:
    def test_table_saved_as_markdown(self, tmp_path):
        proc = _make_processor(table_format="markdown")
        out = tmp_path / "out"
        out.mkdir()
        table = SimpleNamespace(content="| A | B |\n|---|---|\n| 1 | 2 |")
        page = _page(tables=[table])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        table_path = out / "doc" / "tables" / "page1_table1.md"
        assert table_path.exists()
        assert "| A | B |" in table_path.read_text()
        md = (out / "doc" / "doc.md").read_text()
        assert "[Table 1](./tables/page1_table1.md)" in md

    def test_table_saved_as_html(self, tmp_path):
        proc = _make_processor(table_format="html")
        out = tmp_path / "out"
        out.mkdir()
        table = SimpleNamespace(content="<table><tr><td>1</td></tr></table>")
        page = _page(tables=[table])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        table_path = out / "doc" / "tables" / "page1_table1.html"
        assert table_path.exists()
        assert "<table>" in table_path.read_text()

    def test_table_falls_back_to_markdown_attr(self, tmp_path):
        proc = _make_processor(table_format="markdown")
        out = tmp_path / "out"
        out.mkdir()
        table = SimpleNamespace(markdown="| X |")  # no 'content' attr
        page = _page(tables=[table])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        assert "| X |" in (out / "doc" / "tables" / "page1_table1.md").read_text()


# ---------------------------------------------------------------------------
# Hyperlinks (OCR 3)
# ---------------------------------------------------------------------------


class TestHyperlinks:
    def test_hyperlinks_rendered(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        link = SimpleNamespace(text="Example", url="https://example.com")
        page = _page(hyperlinks=[link])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "**Hyperlinks:**" in md
        assert "[Example](https://example.com)" in md

    def test_hyperlink_without_text(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        link = SimpleNamespace(text="", url="https://example.com")
        page = _page(hyperlinks=[link])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "[https://example.com](https://example.com)" in md

    def test_hyperlink_with_href_fallback(self, tmp_path):
        proc = _make_processor()
        out = tmp_path / "out"
        out.mkdir()
        link = SimpleNamespace(text="Link", href="https://test.com")  # no 'url' attr
        page = _page(hyperlinks=[link])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        md = (out / "doc" / "doc.md").read_text()
        assert "[Link](https://test.com)" in md


# ---------------------------------------------------------------------------
# Figures / images
# ---------------------------------------------------------------------------


class TestFigures:
    def test_image_saved_and_linked(self, tmp_path):
        proc = _make_processor(include_images=True)
        out = tmp_path / "out"
        out.mkdir()
        # 1x1 red PNG as base64
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        img = SimpleNamespace(image_base64=b64, id="fig1.png")
        page = SimpleNamespace(index=0, markdown="text", images=[img])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        assert (out / "doc" / "figures" / "page1_img1.png").exists()
        md = (out / "doc" / "doc.md").read_text()
        assert "![Image 1](./figures/page1_img1.png)" in md

    def test_no_figures_when_images_disabled(self, tmp_path):
        proc = _make_processor(include_images=False)
        out = tmp_path / "out"
        out.mkdir()
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        img = SimpleNamespace(image_base64=b64, id="fig1.png")
        page = SimpleNamespace(index=0, markdown="text", images=[img])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        assert not (out / "doc" / "figures").exists()
        md = (out / "doc" / "doc.md").read_text()
        assert "![Image" not in md

    def test_image_default_extension(self, tmp_path):
        proc = _make_processor(include_images=True)
        out = tmp_path / "out"
        out.mkdir()
        b64 = base64.b64encode(b"\x89PNG" + b"\x00" * 50).decode()
        img = SimpleNamespace(image_base64=b64, id="no_ext")  # no extension in id
        page = SimpleNamespace(index=0, markdown="text", images=[img])
        res = _result(tmp_path, SimpleNamespace(pages=[page]))
        proc.save_results(res, out)
        assert (out / "doc" / "figures" / "page1_img1.png").exists()
