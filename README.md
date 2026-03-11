# Mistral OCR CLI

[![CI](https://github.com/r-uben/mistral-ocr-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/r-uben/mistral-ocr-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/mistral-ocr-cli.svg)](https://badge.fury.io/py/mistral-ocr-cli)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR processing using Mistral AI's OCR API. Process PDFs and images to extract text, tables, equations, and images.

> **Disclaimer:** This is an unofficial, community-created tool. Not affiliated with or endorsed by [Mistral AI](https://mistral.ai). For official tools, see the [Mistral OCR docs](https://docs.mistral.ai/capabilities/OCR/).

## Installation

Requires Python 3.11+ and a [Mistral API key](https://console.mistral.ai/).

```bash
pip install mistral-ocr-cli
```

Or from source:

```bash
git clone https://github.com/r-uben/mistral-ocr-cli.git
cd mistral-ocr-cli
uv sync
```

## Quick start

```bash
# Set your API key
export MISTRAL_API_KEY="your_key_here"

# Process a single file
mistral-ocr document.pdf

# Process a directory
mistral-ocr ./documents --output-path ./results

# Preview what would be processed (no API calls)
mistral-ocr ./documents --dry-run

# Process 4 files concurrently
mistral-ocr ./documents --workers 4
```

## Options

```
Usage: mistral-ocr INPUT_PATH [OPTIONS]

Arguments:
  INPUT_PATH                        Path to input file or directory (required)

Options:
  -o, --output-path PATH            Output directory (default: <input_dir>/mistral_ocr_output/)
  --api-key TEXT                    Mistral API key (or set MISTRAL_API_KEY env var)
  --model TEXT                      OCR model (default: mistral-ocr-latest)
  --env-file PATH                   Path to .env file

  --include-images/--no-images      Extract embedded images (default: True)
  --save-originals/--no-save-originals  Copy original files to output (default: True)
  --metadata/--no-metadata          Include markdown header block (default: True)
  --page-headings/--no-page-headings  Include "## Page N" headings (default: True)

  --table-format [markdown|html]    Extract tables separately (OCR 3+)
  --extract-headers/--no-extract-headers  Extract page headers (OCR 3+)
  --extract-footers/--no-extract-footers  Extract page footers (OCR 3+)

  -w, --workers N                   Concurrent workers for batch processing (default: 1)
  --reprocess                       Reprocess already-processed files
  --add-timestamp/--no-timestamp    Timestamp output folder name (default: False)
  --dry-run                         List files without calling the API
  -q, --quiet                       Suppress all output except errors
  -v, --verbose                     Enable verbose/debug output
  --log-file PATH                   Write logs to file
  --version                         Show version
  --help                            Show this message
```

## Output structure

```
mistral_ocr_output/
├── document_name/
│   ├── document_name.pdf       # original copy (if --save-originals)
│   ├── document_name.md        # OCR markdown
│   ├── figures/                # extracted images
│   │   ├── page1_img1.png
│   │   └── page2_img1.png
│   └── tables/                 # extracted tables (if --table-format)
│       └── page1_table1.md
├── another_document/
│   └── ...
└── metadata.json               # processing stats, file list, errors
```

Use `--no-metadata` and `--no-page-headings` for cleaner markdown output without the header block and page separators.

## Configuration

All CLI options can also be set via environment variables or a `.env` file:

| CLI flag | Environment variable | Default |
|----------|---------------------|---------|
| `--api-key` | `MISTRAL_API_KEY` | (required) |
| `--model` | `MISTRAL_MODEL` | `mistral-ocr-latest` |
| `--include-images` | `INCLUDE_IMAGES` | `true` |
| `--save-originals` | `SAVE_ORIGINAL_IMAGES` | `true` |
| `--metadata` | `INCLUDE_METADATA` | `true` |
| `--page-headings` | `INCLUDE_PAGE_HEADINGS` | `true` |
| `--table-format` | `TABLE_FORMAT` | (none) |
| `--extract-headers` | `EXTRACT_HEADER` | `false` |
| `--extract-footers` | `EXTRACT_FOOTER` | `false` |
| `--workers` | `MAX_WORKERS` | `1` |
| `--verbose` | `VERBOSE` | `false` |
| | `MAX_FILE_SIZE_MB` | `50` |
| | `MAX_RETRIES` | `3` |
| | `RETRY_BASE_DELAY` | `1.0` |

CLI flags override environment variables when explicitly passed.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy mistral_ocr/ --ignore-missing-imports
```

## Limitations

- Maximum file size: 50 MB (configurable via `MAX_FILE_SIZE_MB`)
- Supported formats: PDF, DOCX, PPTX, JPG, JPEG, PNG, WEBP, GIF, BMP, TIFF

## Pricing

Mistral OCR API: ~$1 per 1,000 pages. See [Mistral pricing](https://mistral.ai/products/pricing/) for current rates.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Legal

"Mistral AI" and "Mistral" are trademarks of Mistral AI. This project is not affiliated with or endorsed by Mistral AI. Use of Mistral AI's OCR API is subject to Mistral AI's [Terms of Service](https://mistral.ai/terms/).
