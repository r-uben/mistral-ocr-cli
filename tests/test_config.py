"""Tests for configuration module."""

import pytest

from mistral_ocr.config import Config


def test_config_defaults():
    """Test Config dataclass default values."""
    config = Config(api_key="test-key")
    assert config.api_key == "test-key"
    assert config.model == "mistral-ocr-latest"
    assert config.max_file_size_mb == 50
    assert config.include_images is True
    assert config.save_original_images is True
    assert config.table_format is None
    assert config.extract_header is False
    assert config.extract_footer is False
    assert config.max_retries == 3
    assert config.retry_base_delay == 1.0
    assert config.verbose is False


def test_config_custom_values():
    """Test Config with custom values."""
    config = Config(
        api_key="key",
        model="custom-model",
        max_file_size_mb=100,
        include_images=False,
        table_format="html",
        extract_header=True,
        max_retries=5,
        retry_base_delay=2.0,
    )
    assert config.model == "custom-model"
    assert config.max_file_size_mb == 100
    assert config.include_images is False
    assert config.table_format == "html"
    assert config.extract_header is True
    assert config.max_retries == 5
    assert config.retry_base_delay == 2.0


def test_config_from_env(monkeypatch, tmp_path):
    """Test Config.from_env loads from environment."""
    monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
    monkeypatch.setenv("MISTRAL_MODEL", "custom-model")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "100")
    monkeypatch.setenv("INCLUDE_IMAGES", "false")
    monkeypatch.setenv("TABLE_FORMAT", "html")
    monkeypatch.setenv("EXTRACT_HEADER", "true")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("RETRY_BASE_DELAY", "2.5")

    config = Config.from_env()
    assert config.api_key == "env-key"
    assert config.model == "custom-model"
    assert config.max_file_size_mb == 100
    assert config.include_images is False
    assert config.table_format == "html"
    assert config.extract_header is True
    assert config.max_retries == 5
    assert config.retry_base_delay == 2.5


def test_config_from_env_missing_key(monkeypatch):
    """Test Config.from_env raises without API key."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        Config.from_env()


def test_config_from_env_invalid_table_format(monkeypatch):
    """Test Config.from_env ignores invalid table format."""
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    monkeypatch.setenv("TABLE_FORMAT", "invalid")
    config = Config.from_env()
    assert config.table_format is None


def test_validate_file_size(tmp_path):
    """Test file size validation."""
    config = Config(api_key="key", max_file_size_mb=1)

    # Create a small file (should pass)
    small_file = tmp_path / "small.pdf"
    small_file.write_bytes(b"x" * 100)
    config.validate_file_size(small_file)  # no exception

    # Create a file exceeding limit
    big_file = tmp_path / "big.pdf"
    big_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
    with pytest.raises(ValueError, match="exceeds maximum"):
        config.validate_file_size(big_file)
