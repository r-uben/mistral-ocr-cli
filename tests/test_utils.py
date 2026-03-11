"""Tests for utility functions."""

import base64

import pytest

from mistral_ocr.utils import (
    create_data_uri,
    determine_output_path,
    encode_file_to_base64,
    format_file_size,
    get_mime_type,
    get_supported_files,
    load_metadata,
    make_unique_basename,
    sanitize_filename,
    save_base64_image,
    save_metadata,
)


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


class TestSaveBase64Image:
    def test_saves_image(self, tmp_path):
        data = base64.b64encode(b"fake image data").decode()
        out = tmp_path / "subdir" / "img.png"
        save_base64_image(data, out)
        assert out.read_bytes() == b"fake image data"

    def test_strips_data_uri_prefix(self, tmp_path):
        """save_base64_image should handle raw base64 (no prefix)."""
        raw = base64.b64encode(b"pixels").decode()
        out = tmp_path / "img.png"
        save_base64_image(raw, out)
        assert out.read_bytes() == b"pixels"


class TestGetSupportedFiles:
    def test_finds_supported_files(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"")
        (tmp_path / "img.png").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        files = get_supported_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"doc.pdf", "img.png"}

    def test_excludes_output_dir(self, tmp_path):
        out = tmp_path / "mistral_ocr_output"
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


class TestDetermineOutputPath:
    def test_default_output(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"")
        result = determine_output_path(f)
        assert result.name == "mistral_ocr_output"
        assert result.parent == tmp_path

    def test_custom_output(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"")
        custom = tmp_path / "custom_out"
        result = determine_output_path(f, output_path=custom)
        assert result == custom
        assert custom.exists()

    def test_creates_nested_output(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"")
        nested = tmp_path / "a" / "b" / "c"
        result = determine_output_path(f, output_path=nested)
        assert result == nested
        assert nested.is_dir()

    def test_output_path_is_file_raises(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"")
        existing_file = tmp_path / "block"
        existing_file.write_bytes(b"")
        with pytest.raises(ValueError, match="not a directory"):
            determine_output_path(f, output_path=existing_file)

    def test_timestamp(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"")
        result = determine_output_path(f, add_timestamp=True)
        assert "mistral_ocr_output_" in result.name


class TestMetadata:
    def test_load_missing(self, tmp_path):
        meta = load_metadata(tmp_path)
        assert meta["files_processed"] == []
        assert meta["total_files"] == 0

    def test_save_and_load(self, tmp_path):
        save_metadata(tmp_path, [{"file": "/a.pdf", "size": 100}], 1.5, [])
        meta = load_metadata(tmp_path)
        assert meta["total_files"] == 1
        assert meta["files_processed"][0]["file"] == "/a.pdf"
        assert meta["processing_time_seconds"] == 1.5

    def test_incremental_save(self, tmp_path):
        save_metadata(tmp_path, [{"file": "/a.pdf", "size": 100}], 1.0, [])
        save_metadata(
            tmp_path,
            [{"file": "/b.pdf", "size": 200}],
            2.0,
            [],
            base_processing_time=1.0,
        )
        meta = load_metadata(tmp_path)
        assert meta["total_files"] == 2
        assert meta["processing_time_seconds"] == 3.0

    def test_corrupt_metadata(self, tmp_path):
        (tmp_path / "metadata.json").write_text("not json{{{")
        meta = load_metadata(tmp_path)
        assert meta["files_processed"] == []


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


class TestMakeUniqueBasename:
    def test_simple(self, tmp_path):
        f = tmp_path / "report.pdf"
        assert make_unique_basename(f) == "report"

    def test_with_base_dir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "report.pdf"
        result = make_unique_basename(f, base_dir=tmp_path)
        assert result == "sub__report"


class TestSanitizeFilename:
    def test_removes_invalid_chars(self):
        assert sanitize_filename("file<>:name") == "file___name"

    def test_truncates(self):
        result = sanitize_filename("a" * 300, max_length=50)
        assert len(result) <= 50

    def test_no_truncation_by_default(self):
        long = "a" * 300
        assert sanitize_filename(long) == long
