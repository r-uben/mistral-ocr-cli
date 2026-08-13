# Mistral OCR CLI

[![CI](https://github.com/r-uben/mistral-ocr-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/r-uben/mistral-ocr-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/mistral-ocr-cli.svg)](https://badge.fury.io/py/mistral-ocr-cli)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR processing using Mistral AI's OCR API. Process PDFs and images to extract text, tables, equations, and images.

> **Disclaimer:** This is an unofficial, community-created tool. Not affiliated with or endorsed by [Mistral AI](https://mistral.ai). For official tools, see the [Mistral Document AI docs](https://docs.mistral.ai/capabilities/document_ai/basic_ocr).

## Choosing an OCR tool

This is one of five OCR CLI tools with a shared design: clean Markdown output, batch processing, and figure extraction. Pick based on your constraints:

| Tool | Engine | Runs | Cost | Best for |
|------|--------|------|------|----------|
| [deepseek-ocr-cli](https://github.com/r-uben/deepseek-ocr-cli) | DeepSeek vision | Local (Ollama / vLLM) | Free | General-purpose local OCR with multi-backend flexibility |
| [gemini-ocr-cli](https://github.com/r-uben/gemini-ocr-cli) | Google Gemini | Cloud API | Free tier / Pay-per-use | Fast cloud OCR with concurrent processing |
| [marker-ocr-cli](https://github.com/r-uben/marker-ocr-cli) | Marker (Surya + Texify) | Local | Free | Academic papers with equations, tables, complex layouts |
| **mistral-ocr-cli** (this repo) | Mistral OCR API | Cloud API | $4/1k pages | Structured extraction (tables, headers, footers) |
| [nougat-ocr-cli](https://github.com/r-uben/nougat-ocr-cli) | Meta Nougat | Local (GPU) | Free | Academic papers, GPU-accelerated batch processing |

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

# Process a directory (default output root is ./documents/ocr/)
mistral-ocr ./documents -o ./results

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
  -o, --output-dir PATH             Output root (default: <input-parent>/ocr/). Never required.
  --api-key TEXT                    Mistral API key (or set MISTRAL_API_KEY env var)
  --model TEXT                      OCR model (default: mistral-ocr-latest)
  --env-file PATH                   Path to .env file

  --include-images/--no-images      Extract embedded figures (default: True)

  --table-format [markdown|html]    Request tables inline in a given format (OCR 3+)
  --extract-headers/--no-extract-headers  Extract page headers (OCR 3+)
  --extract-footers/--no-extract-footers  Extract page footers (OCR 3+)

  --include-blocks/--no-include-blocks    Write per-page blocks to blocks.json (OCR 4+)
  --confidence-scores-granularity [page|block|word]  Confidence scores in blocks.json (OCR 4+)

  --max-pages N                     Max PDF pages to process (default: all pages)
  -w, --workers N                   Concurrent workers for batch processing (default: 1)
  --reprocess                       Re-OCR files already recorded completed (checksum-based)
  --dry-run                         List files without calling the API
  -q, --quiet                       Suppress output except the written .md paths (for scripting)
  -v, --verbose                     Enable verbose/debug output
  --log-file PATH                   Write logs to file
  --version                         Show version
  --help                            Show this message
```

> Output writing is owned by the shared
> [`ocr-output-contract`](https://github.com/r-uben/ocr-output-contract) package, so
> mistral's output structure is byte-identical to every sibling OCR engine CLI. The
> markdown body is always clean (`## Page N` headers, no header block, no YAML
> frontmatter); all provenance lives in the JSON sidecars. The `--save-originals`,
> `--metadata`, `--page-headings` and `--add-timestamp` flags are deprecated no-ops kept
> only for invocation compatibility.

## Output structure

Default output root is `<input-parent>/ocr/` (`-o` overrides verbatim; never required).
Each source document gets one aggregated folder, mirroring the input subtree so
same-basename inputs in different directories never collide:

```
ocr/
├── document_name/
│   ├── document_name.md        # all pages, joined under "## Page N" headers (clean body)
│   ├── figures/                # extracted embedded images (normalised to PNG)
│   │   ├── figure_1_page1.png
│   │   └── figure_2_page2.png
│   ├── metadata.json           # per-document sidecar: status/checksum/model/backend/...
│   └── blocks.json             # only with --include-blocks (see below)
├── sub/dir/another_document/
│   └── ...
└── metadata.json               # root index, keyed by input-relative path
```

Resume is content-aware: a file recorded `completed` is skipped only when its SHA-256
checksum still matches, so editing a file in place forces a re-OCR. Failures are recorded
with `status="failed"`, and any file/page failure drives a nonzero exit (uniform across
single-file and batch runs).

### Blocks and confidence scores (OCR 4+)

`--include-blocks` requests per-page `blocks[]` — structural type, bounding box,
content and reading order — and `--confidence-scores-granularity {page,block,word}`
requests confidence scores. Both are **off by default**, and both write to a
`blocks.json` sidecar rather than into the markdown:

```json
{
  "version": 1,
  "backend": "mistral-api",
  "model": "mistral-ocr-latest",
  "pages": [
    {
      "page": 1,
      "blocks": [
        {
          "type": "text",
          "top_left_x": 43, "top_left_y": 55,
          "bottom_right_x": 125, "bottom_right_y": 69,
          "content": "Migration parity test - page 1"
        }
      ],
      "confidence_scores": {"average_page_confidence_score": 0.97}
    }
  ]
}
```

Block types are `text`, `title`, `list`, `table`, `image`, `equation`, `caption`,
`code`, `references`, `aside_text`, `header`, `footer`, `signature`.

Three properties worth relying on:

- **The markdown body never changes.** Turning blocks on produces a byte-identical
  `.md`; the sidecar is purely additive. Blocks live outside `metadata.json`
  because `DocMetadata` is a closed field set owned by the shared output contract.
- **`page` matches the body's `## Page N` headers**, including for large PDFs split
  across requests, so the sidecar joins cleanly to the text.
- **Enabling either flag forces a re-OCR** of affected documents, since it changes
  the run fingerprint. Leaving both off keeps the fingerprint exactly as it was
  before this feature existed, so upgrading does **not** silently reprocess an
  existing corpus at $4/1k pages. Turning blocks off again removes the sidecar.

Embedded base64 payloads are stripped from blocks — figures are already written to
`figures/`, and the API's image ids are request-local, so they would not resolve.

## Configuration

All CLI options can also be set via environment variables or a `.env` file:

| CLI flag | Environment variable | Default |
|----------|---------------------|---------|
| `--api-key` | `MISTRAL_API_KEY` | (required) |
| `--model` | `MISTRAL_MODEL` | `mistral-ocr-latest` |
| `--include-images` | `INCLUDE_IMAGES` | `true` |
| `--table-format` | `TABLE_FORMAT` | (none) |
| `--extract-headers` | `EXTRACT_HEADER` | `false` |
| `--extract-footers` | `EXTRACT_FOOTER` | `false` |
| `--include-blocks` | `INCLUDE_BLOCKS` | `false` |
| `--confidence-scores-granularity` | `CONFIDENCE_SCORES_GRANULARITY` | (off) |
| `--max-pages` | `MAX_PAGES` | (all pages) |
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

Mistral OCR API: **$4 per 1,000 pages** ($2 per 1,000 via the Batch API).

The default model, `mistral-ocr-latest`, is an alias for OCR 4.1 — confirmed
against the models endpoint, where it reports aliases `mistral-ocr-4` and
`mistral-ocr-4-1`. Pricing moved to $4/1k with the OCR 4 release; earlier OCR 3
rates no longer apply to the default model.

Note that the OCR response echoes back whichever model string you pass, so a run
made with the alias records `mistral-ocr-latest` and cannot later be attributed
to a specific OCR 4 point release. Pass `--model mistral-ocr-4-1` (or
`mistral-ocr-4-0`) explicitly if you need the exact version pinned in
`metadata.json` and in the resume fingerprint.

See [Mistral pricing](https://mistral.ai/pricing/api) for current rates.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Legal

"Mistral AI" and "Mistral" are trademarks of Mistral AI. This project is not affiliated with or endorsed by Mistral AI. Use of Mistral AI's OCR API is subject to Mistral AI's [Terms of Service](https://mistral.ai/terms/).
