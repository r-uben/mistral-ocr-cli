from mistral_ocr.utils import determine_output_path


def test_determine_output_path_creates_explicit_directory(tmp_path) -> None:
    output_path = tmp_path / "custom-output"

    resolved = determine_output_path(tmp_path / "document.pdf", output_path)

    assert resolved == output_path
    assert output_path.is_dir()
