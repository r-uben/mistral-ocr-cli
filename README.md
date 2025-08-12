# Mistral OCR CLI

[![PyPI version](https://badge.fury.io/py/mistral-ocr-cli.svg)](https://badge.fury.io/py/mistral-ocr-cli)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful command-line tool for OCR processing using Mistral AI's state-of-the-art OCR API. Process PDFs and images to extract text, tables, equations, and images with unprecedented accuracy.

## ⚠️ Disclaimer

**This is an unofficial, community-created tool** that uses Mistral AI's OCR API. This project is not affiliated with, officially maintained, or endorsed by Mistral AI.

- **Official Mistral OCR**: https://mistral.ai/news/mistral-ocr
- **Official Documentation**: https://docs.mistral.ai/capabilities/OCR/
- **Mistral AI Platform**: https://console.mistral.ai/

For official tools and support, please visit [Mistral AI's website](https://mistral.ai).

## Features

- 📄 **Multi-format Support**: Process PDFs and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- 📊 **Advanced Extraction**: Extract text, tables, equations, and images
- 📁 **Batch Processing**: Process single files or entire directories
- 🎯 **Smart Output**: Preserves document structure in markdown format
- 🖼️ **Image Extraction**: Optionally extract and save embedded images
- 📈 **Progress Tracking**: Real-time progress bars for batch operations
- 🔧 **Flexible Configuration**: Environment variables or command-line options

## Installation

### Prerequisites

- Python 3.9 or higher
- Mistral API key from [Mistral Console](https://console.mistral.ai/)

### Install from PyPI (Recommended)

```bash
pip install mistral-ocr-cli
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/r-uben/mistral-ocr-cli.git
cd mistral-ocr-cli

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

## Configuration

### 1. Set up your Mistral API key

Create a `.env` file in your project root (or copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
MISTRAL_API_KEY=your_actual_api_key_here
```

### 2. Alternative: Export as environment variable

```bash
export MISTRAL_API_KEY="your_actual_api_key_here"
```

## Usage

### Basic Usage

Process a single PDF file (output saved to `mistral_ocr_output/` in the same directory):

```bash
mistral-ocr document.pdf
```

### Specify Output Directory

```bash
mistral-ocr document.pdf --output-path ./results
```

### Process Entire Directory

```bash
mistral-ocr ./documents --output-path ./extracted
```

### Command-Line Options

```
Usage: mistral-ocr INPUT_PATH [OPTIONS]

Arguments:
  INPUT_PATH                  Path to input file or directory (required)

Options:
  -o, --output-path PATH      Output directory (default: <input_dir>/mistral_ocr_output/)
  --api-key TEXT             Mistral API key (or set MISTRAL_API_KEY env var)
  --model TEXT               OCR model (default: mistral-ocr-latest)
  --env-file PATH            Path to .env file
  --include-images/--no-images  Extract images (default: True)
  -v, --verbose              Enable verbose output
  --version                  Show version
  --help                     Show this message
```

## Examples

### Process a single image

```bash
mistral-ocr photo.jpg
```

### Process multiple PDFs with custom output

```bash
mistral-ocr ./reports --output-path ./extracted_text --verbose
```

### Use a different .env file

```bash
mistral-ocr document.pdf --env-file .env.production
```

### Process without extracting images

```bash
mistral-ocr document.pdf --no-images
```

### Pass API key directly (not recommended for production)

```bash
mistral-ocr doc.pdf --api-key "your_api_key_here"
```

## Output Structure

The tool creates the following output structure:

```
mistral_ocr_output/
├── document1.md           # Extracted text in markdown format
├── document1_images/      # Extracted images (if enabled)
│   ├── page1_img1.png
│   └── page1_img2.png
├── document2.md
├── document2_images/
│   └── ...
└── metadata.json         # Processing statistics and errors
```

### Markdown Output

Each processed document generates a markdown file containing:
- Document metadata (source, processing time)
- Extracted text with preserved formatting
- Tables rendered in markdown format
- Mathematical equations
- Image references (if image extraction is enabled)

### Metadata File

The `metadata.json` file contains:
- List of processed files
- Processing time
- File sizes
- Output paths
- Any errors encountered

## Limitations

- Maximum file size: 50 MB
- Maximum pages per document: 1,000
- Supported formats: PDF, JPG, JPEG, PNG, WEBP, GIF, BMP, TIFF

## Pricing

Mistral OCR API pricing: $1 per 1,000 pages ($0.001 per page)

## Development

### Run tests

```bash
poetry run pytest
```

### Format code

```bash
poetry run black mistral_ocr/
poetry run ruff check mistral_ocr/
```

### Type checking

```bash
poetry run mypy mistral_ocr/
```

## Troubleshooting

### API Key Not Found

If you get an error about missing API key:
1. Ensure `.env` file exists and contains `MISTRAL_API_KEY=your_key`
2. Or export it: `export MISTRAL_API_KEY="your_key"`
3. Or pass it directly: `mistral-ocr --api-key "your_key" ...`

### File Size Error

If a file exceeds 50 MB:
- Consider splitting large PDFs into smaller parts
- Compress images before processing

### Installation Issues

If the command is not found after installation:
1. Ensure the package is installed: `pip show mistral-ocr`
2. Check your PATH includes pip's script directory
3. Try reinstalling with: `pip install -e .`

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions about this CLI tool, please open an issue on [GitHub](https://github.com/r-uben/mistral-ocr-cli/issues)

For questions about Mistral AI's OCR API, please refer to [Mistral's official documentation](https://docs.mistral.ai) or contact their support.

## Legal

"Mistral AI" and "Mistral" are trademarks of Mistral AI. This project is not affiliated with or endorsed by Mistral AI. The use of Mistral AI's OCR API is subject to Mistral AI's [Terms of Service](https://mistral.ai/terms/).