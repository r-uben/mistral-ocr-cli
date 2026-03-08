import base64
from types import SimpleNamespace

from mistral_ocr.config import Config
from mistral_ocr.processor import OCRProcessor
from mistral_ocr.utils import get_image_base64_data


def test_get_image_base64_data_strips_data_uri_prefix() -> None:
    image = SimpleNamespace(image_base64="data:image/png;base64,Zm9v")

    assert get_image_base64_data(image) == "Zm9v"


def test_get_image_base64_data_supports_legacy_base64_field() -> None:
    image = SimpleNamespace(base64="YmFy")

    assert get_image_base64_data(image) == "YmFy"


def test_save_results_writes_images_from_image_base64(tmp_path) -> None:
    processor = OCRProcessor.__new__(OCRProcessor)
    processor.config = Config(api_key="test", include_images=True)

    image_bytes = b"fake-image-bytes"
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    response = SimpleNamespace(
        pages=[
            SimpleNamespace(
                index=0,
                markdown="Hello world",
                images=[
                    SimpleNamespace(
                        image_base64=f"data:image/png;base64,{image_data}",
                    )
                ],
            )
        ]
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    processor.save_results(
        {"file_path": tmp_path / "document.pdf", "response": response},
        output_dir,
    )

    assert (output_dir / "document.md").exists()
    assert (output_dir / "images" / "page1_img1.png").read_bytes() == image_bytes
