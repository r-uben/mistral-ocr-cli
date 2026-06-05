"""Tests for utility functions (mistral-specific I/O only).

Output-shape helpers (output-root resolution, keying, layout, metadata, figure
naming) now live in the ``ocr-output-contract`` package and are exercised via
``test_output_contract.py``, not here.
"""

import base64

import pytest

from mistral_ocr.utils import (
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    create_data_uri,
    decode_base64_image,
    encode_file_to_base64,
    format_file_size,
    get_mime_type,
    get_supported_files,
)


class TestExtensionSets:
    def test_supported_is_union(self):
        assert SUPPORTED_EXTENSIONS == DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS
        assert ".pdf" in DOCUMENT_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS


class TestEncodeFileToBase64:
    def test_encodes_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello")
        result = encode_file_to_base64(f)
        assert base64.b64decode(result) == b"hello"


class TestGetMimeType:
    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".pdf", "application/pdf"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".png", "image/png"),
            (".webp", "image/webp"),
            (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ],
    )
    def test_known_types(self, suffix, expected, tmp_path):
        f = tmp_path / f"file{suffix}"
        f.write_bytes(b"")
        assert get_mime_type(f) == expected

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "file.qqq_unknown_ext"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_mime_type(f)


class TestCreateDataUri:
    def test_creates_uri(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG")
        uri = create_data_uri(f)
        assert uri.startswith("data:image/png;base64,")


class TestDecodeBase64Image:
    def test_decodes_raw(self):
        raw = base64.b64encode(b"pixels").decode()
        assert decode_base64_image(raw) == b"pixels"

    def test_strips_data_uri_prefix(self):
        raw = base64.b64encode(b"pixels").decode()
        uri = f"data:image/png;base64,{raw}"
        assert decode_base64_image(uri) == b"pixels"


class TestGetSupportedFiles:
    def test_finds_supported_files(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"")
        (tmp_path / "img.png").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        files = get_supported_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"doc.pdf", "img.png"}

    def test_excludes_canonical_output_dir(self, tmp_path):
        """The default output dir name (ocr/) is excluded from discovery."""
        out = tmp_path / "ocr"
        out.mkdir()
        (out / "result.pdf").write_bytes(b"")
        (tmp_path / "input.pdf").write_bytes(b"")
        files = get_supported_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "input.pdf"

    def test_excludes_absolute_paths(self, tmp_path):
        sub = tmp_path / "output"
        sub.mkdir()
        (sub / "file.pdf").write_bytes(b"")
        (tmp_path / "input.pdf").write_bytes(b"")
        files = get_supported_files(tmp_path, exclude_paths=[sub])
        assert len(files) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "deep.jpg").write_bytes(b"")
        files = get_supported_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "deep.jpg"


class TestFormatFileSize:
    @pytest.mark.parametrize(
        "size,expected",
        [
            (500, "500.00 B"),
            (1024, "1.00 KB"),
            (1024 * 1024, "1.00 MB"),
            (1024 * 1024 * 1024, "1.00 GB"),
        ],
    )
    def test_formatting(self, size, expected):
        assert format_file_size(size) == expected
