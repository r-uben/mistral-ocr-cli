"""Core OCR processing using the Mistral OCR API, emitting canonical output.

Mistral is a *cloud document-OCR API*: a file is uploaded once and the API
returns a per-page result list (each page carrying ``.markdown`` plus optional
OCR 3 fields — page dimensions, header/footer, tables, hyperlinks, embedded
images). This module owns *how OCR happens* (the API round-trip, large-PDF
chunking + global page re-indexing, and folding the per-page OCR 3 extras into
each page's markdown). The shared ``ocr-output-contract`` package owns *where
the bytes go and what the metadata looks like*, so mistral's output is
byte-structure-identical to every sibling engine.

The per-page text list this module produces is fed straight into the contract's
:func:`assemble_pages`, so every page of a document lands in ONE
``<root>/<rel/dir>/<stem>/<stem>.md`` under ``## Page N`` headers (the API's
whole-doc response is never dumped as a single blob). Embedded images are
normalised to PNG and named ``figures/figure_<N>_page<P>.png`` with resolving
links. Provenance lives only in the dual-level ``metadata.json`` sidecars — the
markdown body is clean (no ``# OCR Results`` header, no YAML frontmatter).

Audit fixes baked in here:

* **HIGH (idempotency)** — resume is now content-aware via the package's
  :meth:`RootIndex.is_completed` (status==completed AND SHA256 match), replacing
  the old resolved-path-equality skip that served stale output on in-place edits.
* **Metadata** — the non-standard ``files_processed``/``errors`` list schema is
  gone wholesale; the dual-level ``DocMetadata`` + ``RootIndex`` carries
  ``version``/``checksum``/``model``/``backend`` and records failures
  (``status=failed``).
* **Retry** — the catch-all that retried every ``SDKError`` (incl. permanent
  4xx) now excludes non-429 4xx, so auth/validation errors fail fast.
* **MAX_FILE_SIZE_MB** — now enforced for PDFs too, not only the non-PDF branch.
* **Exit policy** — a :class:`RunOutcome` drives a uniform nonzero exit on any
  failure, across single-file and batch.
"""

from __future__ import annotations

import io
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mistralai import Mistral
from ocr_output_contract import (
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    figure_filename,
    figure_markdown_link,
    figures_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .config import Config
from .utils import (
    DOCUMENT_EXTENSIONS,
    create_data_uri,
    decode_base64_image,
    format_file_size,
    get_pdf_page_count,
    get_supported_files,
    split_pdf,
)

logger = logging.getLogger(__name__)

# Shared console instance — CLI sets .quiet on this directly
console = Console()

# Dedicated stderr console for per-file failures. It is NEVER muted by --quiet
# (the CLI only sets console.quiet), so SYS-02's "failures still emitted to
# stderr under --quiet" holds while stdout stays a clean one-path-per-line list
# of successful outputs for scripting.
err_console = Console(stderr=True)

#: Backend identifier recorded in metadata. Mistral is a cloud OCR API.
BACKEND = "mistral-api"

# Mistral API limit: max pages per single OCR request
MAX_PAGES_PER_REQUEST = 1000

#: HTTP statuses that are transient and worth retrying. Everything else in the
#: 4xx range (401 auth, 400/422 validation, ...) is permanent and must fail fast.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class OCRResult:
    """Result of OCR'ing one source document.

    ``pages`` holds the per-page markdown in order (already including any folded
    OCR 3 extras). ``page_images`` maps a 1-indexed page number to the list of
    raw image-bytes extracted from that page. ``status`` maps to the contract
    enum: a document that could not be opened/processed at all is ``FAILED``;
    one whose API call returned zero pages is also ``FAILED``.
    """

    file_path: Path
    pages: list[str]
    page_images: dict[int, list[bytes]] = field(default_factory=dict)
    processing_time: float = 0.0
    error: str | None = None
    #: Non-fatal note (e.g. ``--max-pages`` truncation), recorded in metadata.
    note: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def status(self) -> Status:
        """``completed`` when the document yielded >=1 page; else ``failed``.

        The Mistral API does not surface per-page success/failure (it returns a
        page list or raises), so there is no ``partial`` state for a single doc:
        either the call produced pages (completed) or it did not (failed).
        """
        if self.pages and not self.error:
            return Status.COMPLETED
        return Status.FAILED


class OCRProcessor:
    """OCR processor using the Mistral AI API, routed through the contract."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        try:
            self.client = Mistral(api_key=config.api_key)
        except (ValueError, TypeError, RuntimeError) as e:
            console.print(f"[red]Failed to initialize Mistral client: {e}[/red]")
            raise
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # API round-trip (mistral-owned: how OCR happens)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is transient and worth retrying.

        Fix (audit MEDIUM): a permanent 4xx ``SDKError`` (401 auth, 400/422
        validation) is NOT retried. Only the explicit transient statuses, the
        SDK's typed rate-limit/server errors, and network-level errors retry.
        An ``SDKError`` whose status we can read is gated on that status; one
        whose status we cannot read is treated as non-retryable (fail fast)
        rather than blindly retried.
        """
        # SDK-typed rate limit / server errors
        for exc_name in ("RateLimitError", "InternalServerError", "ServiceUnavailableError"):
            if type(error).__name__ == exc_name:
                return True
        # httpx-level / SDK HTTP status errors: only the transient set retries.
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status is None:
            status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status in _RETRYABLE_STATUSES
        # Network-level transient errors retry; an SDKError with no readable
        # status is permanent (fail fast), NOT a blanket retry — that catch-all
        # was the old bug that retried permanent 4xx.
        return isinstance(error, (TimeoutError, ConnectionError, OSError))

    def _call_with_retry(self, **ocr_kwargs: object) -> object:
        """Call ocr.process with exponential backoff on transient errors."""
        max_attempts = self.config.max_retries + 1
        base_delay = self.config.retry_base_delay

        for attempt in range(max_attempts):
            try:
                return self.client.ocr.process(**ocr_kwargs)
            except Exception as e:
                is_last = attempt == max_attempts - 1
                if is_last or not self._is_retryable(e):
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
        # Unreachable, but keeps mypy happy
        raise RuntimeError("Retry loop exited unexpectedly")

    def _build_ocr_kwargs(self, document: dict) -> dict:
        """Build common OCR API kwargs."""
        ocr_kwargs: dict = {
            "model": self.config.model,
            "document": document,
            "include_image_base64": self.config.include_images,
        }
        if self.config.table_format:
            ocr_kwargs["table_format"] = self.config.table_format
        if self.config.extract_header:
            ocr_kwargs["extract_header"] = True
        if self.config.extract_footer:
            ocr_kwargs["extract_footer"] = True
        return ocr_kwargs

    def _upload_and_process(self, file_path: Path) -> object:
        """Upload a file via Mistral files API and process with OCR.

        The uploaded file is always deleted in a ``finally`` block so transient
        OCR files do not accumulate in the user's Mistral account/quota.
        """
        with open(file_path, "rb") as f:
            uploaded = self.client.files.upload(
                file={"file_name": file_path.name, "content": f},
                purpose="ocr",
            )
        try:
            document = {"type": "file", "file_id": uploaded.id}
            return self._call_with_retry(**self._build_ocr_kwargs(document))
        finally:
            try:
                self.client.files.delete(file_id=uploaded.id)
            except Exception:
                logger.debug("Failed to delete uploaded file %s", uploaded.id)

    def _process_pdf(self, file_path: Path) -> object:
        """Process a PDF, chunking if needed for the API page limit.

        Fix (audit LOW): the configured ``MAX_FILE_SIZE_MB`` is enforced here for
        PDFs too (the old code only validated the non-PDF branch).
        """
        self.config.validate_file_size(file_path)

        page_count = get_pdf_page_count(file_path)
        max_pages = self.config.max_pages
        effective_pages = min(page_count, max_pages) if max_pages else page_count

        if effective_pages <= MAX_PAGES_PER_REQUEST:
            if max_pages and page_count > max_pages:
                logger.debug("Truncating %d-page PDF to %d pages", page_count, max_pages)
                return self._process_pdf_chunked(file_path, page_count)
            logger.debug("Uploading PDF directly (%d pages)", page_count)
            return self._upload_and_process(file_path)

        logger.debug(
            "PDF has %d pages (processing %d), splitting into chunks of %d",
            page_count,
            effective_pages,
            MAX_PAGES_PER_REQUEST,
        )
        return self._process_pdf_chunked(file_path, page_count)

    def _process_pdf_chunked(self, file_path: Path, total_pages: int) -> object:
        """Split a PDF into chunks, process each, and reassemble pages.

        Pages are re-indexed to their position in the original document and the
        original page objects are passed through to preserve all OCR 3 fields
        (tables, headers, footers, hyperlinks, dimensions).
        """
        from types import SimpleNamespace

        max_pages = self.config.max_pages

        with tempfile.TemporaryDirectory(prefix="mistral_ocr_") as tmp:
            chunks = split_pdf(
                file_path,
                Path(tmp),
                max_pages_per_chunk=MAX_PAGES_PER_REQUEST,
                max_pages=max_pages,
            )

            all_pages = []
            for chunk_path, start_page, chunk_count in chunks:
                logger.debug(
                    "Processing chunk: pages %d-%d", start_page + 1, start_page + chunk_count
                )
                response = self._upload_and_process(chunk_path)
                for local_idx, page in enumerate(getattr(response, "pages", [])):
                    page.index = start_page + local_idx
                    all_pages.append(page)

        result = SimpleNamespace(pages=all_pages)
        if max_pages and total_pages > max_pages:
            result.truncated = f"Processed {max_pages} of {total_pages} pages (--max-pages)"
        return result

    def _call_api(self, file_path: Path) -> object:
        """Run the Mistral OCR API on one file (PDF chunked, others via data URI)."""
        if not hasattr(self.client, "ocr"):
            raise AttributeError(
                "OCR endpoint not available in Mistral client. "
                "Please ensure you have the latest mistralai package "
                "and OCR access enabled for your API key."
            )
        logger.debug("Sending to Mistral OCR API (model=%s)...", self.config.model)

        if file_path.suffix.lower() == ".pdf":
            return self._process_pdf(file_path)

        # Images and other documents: validate size, use data URI.
        self.config.validate_file_size(file_path)
        data_uri = create_data_uri(file_path)
        if file_path.suffix.lower() in DOCUMENT_EXTENSIONS:
            document = {"type": "document_url", "document_url": data_uri}
        else:
            document = {"type": "image_url", "image_url": data_uri}
        return self._call_with_retry(**self._build_ocr_kwargs(document))

    # ------------------------------------------------------------------
    # Page parsing: API page list -> per-page markdown + figure bytes
    # ------------------------------------------------------------------

    def _render_page_markdown(self, page: Any) -> str:
        """Fold one API page's OCR 3 extras into a single markdown body.

        The contract supplies the ``## Page N`` header at assembly time, so this
        produces only the page *body*: optional page dimensions, header/footer
        quotes, the OCR text, the structured tables, and hyperlinks. Embedded
        image links are appended later (after figures are saved with canonical
        names).

        Tables: when ``--table-format=markdown|html`` is set, Mistral OCR returns
        tables in the structured ``page.tables`` field, NOT inline in
        ``page.markdown`` (the markdown may carry a placeholder referencing the
        table ``id``). Each structured table is rendered into the body so the
        ``--table-format`` flag is not a silent no-op: a placeholder is replaced
        in place when present, otherwise the table is appended.
        """
        parts: list[str] = []

        dims = getattr(page, "dimensions", None)
        if dims:
            w = getattr(dims, "width", None)
            h = getattr(dims, "height", None)
            if w and h:
                parts.append(f"*Page size: {w} x {h}*")

        header = getattr(page, "header", None)
        if header:
            parts.append(f"> **Header:** {header}")

        markdown = getattr(page, "markdown", None) or ""
        markdown, appended_tables = self._render_tables(markdown, getattr(page, "tables", None))
        if markdown:
            parts.append(markdown)
        if appended_tables:
            parts.append(appended_tables)

        hyperlinks = getattr(page, "hyperlinks", None)
        if hyperlinks:
            lines = ["**Hyperlinks:**"]
            for link in hyperlinks:
                text = getattr(link, "text", "") or ""
                url = getattr(link, "url", "") or getattr(link, "href", "") or ""
                if url:
                    lines.append(f"- [{text or url}]({url})")
            if len(lines) > 1:
                parts.append("\n".join(lines))

        footer = getattr(page, "footer", None)
        if footer:
            parts.append(f"> **Footer:** {footer}")

        return "\n\n".join(parts).strip()

    @staticmethod
    def _render_one_table(table: Any) -> str:
        """Render one structured OCR-3 table object into markdown body text.

        ``content`` already carries the table in the requested ``--table-format``
        (a markdown table or an HTML ``<table>``); HTML is passed through verbatim
        (markdown renderers accept inline HTML), markdown likewise. Empty content
        renders to an empty string.
        """
        content: str = str(getattr(table, "content", "") or "")
        return content.strip()

    @classmethod
    def _render_tables(cls, markdown: str, tables: Any) -> tuple[str, str]:
        """Fold structured ``page.tables`` into the page body.

        Returns ``(markdown, appended)`` where ``markdown`` has any in-body table
        placeholder (a markdown link/text referencing the table ``id``) replaced
        by the rendered table, and ``appended`` is the concatenation of tables
        that had no placeholder to anchor them (so a populated ``tables`` field is
        never silently discarded). When ``--table-format`` is NOT set the API
        inlines tables in ``page.markdown`` and ``tables`` is empty, so this is a
        no-op and the existing inline content is preserved.
        """
        if not tables:
            return markdown, ""

        appended: list[str] = []
        for table in tables:
            rendered = cls._render_one_table(table)
            if not rendered:
                continue
            table_id = getattr(table, "id", None)
            placeholder = cls._table_placeholder(markdown, table_id)
            if placeholder is not None:
                markdown = markdown.replace(placeholder, rendered)
            else:
                appended.append(rendered)
        return markdown, "\n\n".join(appended)

    @staticmethod
    def _table_placeholder(markdown: str, table_id: Any) -> str | None:
        """Return the exact placeholder substring in ``markdown`` for a table id.

        Mistral may embed a table placeholder such as ``[tbl-0.html](tbl-0.html)``
        or a bare ``tbl-0.html`` token referencing the table ``id``. Returns the
        matched substring to replace, or ``None`` if no placeholder is present.
        """
        if not table_id or not markdown:
            return None
        tid = str(table_id)
        linked = f"[{tid}]({tid})"
        if linked in markdown:
            return linked
        if tid in markdown:
            return tid
        return None

    @staticmethod
    def _page_image_bytes(page: Any) -> list[bytes]:
        """Decode the embedded base64 images on one API page into raw bytes."""
        out: list[bytes] = []
        for image in getattr(page, "images", None) or []:
            b64 = getattr(image, "image_base64", None) or getattr(image, "base64", None)
            if not b64:
                continue
            try:
                out.append(decode_base64_image(b64))
            except Exception as exc:
                logger.warning("Failed to decode embedded image: %s", exc)
        return out

    def _parse_response(self, file_path: Path, response: Any, start: float) -> OCRResult:
        """Turn an API response into an OCRResult (per-page text + figure bytes)."""
        pages_obj = getattr(response, "pages", None) or []
        pages: list[str] = []
        page_images: dict[int, list[bytes]] = {}
        for page in pages_obj:
            page_no = getattr(page, "index", len(pages)) + 1
            pages.append(self._render_page_markdown(page))
            if self.config.include_images:
                imgs = self._page_image_bytes(page)
                if imgs:
                    page_images[page_no] = imgs
        note = getattr(response, "truncated", None) if hasattr(response, "truncated") else None
        return OCRResult(
            file_path=file_path,
            pages=pages,
            page_images=page_images,
            processing_time=time.time() - start,
            note=note,
        )

    def process_file(self, file_path: Path) -> OCRResult:
        """OCR a single file into an OCRResult (no output written)."""
        start = time.time()
        try:
            response = self._call_api(file_path)
            result = self._parse_response(file_path, response, start)
            if not result.pages:
                result.error = "Empty response from Mistral OCR (no pages returned)"
            return result
        except Exception as e:
            logger.error("Error processing %s: %s", file_path.name, e)
            logger.debug("Traceback for %s", file_path.name, exc_info=True)
            return OCRResult(
                file_path=file_path,
                pages=[],
                processing_time=time.time() - start,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Output writing (all routed through the ocr-output-contract package)
    # ------------------------------------------------------------------

    def save_results(self, result: OCRResult, output_root: Path, rel_key: str) -> Path:
        """Write the aggregated markdown + figures for one document.

        Layout is owned entirely by the contract:
        ``<output_root>/<rel/dir>/<stem>/<stem>.md`` plus a ``figures/`` folder.
        Figures are normalised to PNG, named ``figure_<N>_page<P>.png``, and
        linked from the page that produced them. The body is clean (no
        ``# OCR Results`` header, no frontmatter).
        """
        doc_dir = doc_dir_for(output_root, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_path_for(doc_dir, rel_key)

        if result.pages:
            page_links = self._save_figures(result, doc_dir)
            pages = [
                self._append_links(text, page_links.get(idx, []))
                for idx, text in enumerate(result.pages, start=1)
            ]
            body = assemble_pages(pages)
            if result.note:
                body = f"> **Note:** {result.note}\n\n" + body
        else:
            body = "*[OCR Failed]*\n"

        markdown_path.write_text(body, encoding="utf-8")
        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")
        return markdown_path

    @staticmethod
    def _append_links(text: str, links: list[str]) -> str:
        """Append figure markdown links to a page body."""
        if not links:
            return text
        joined = "\n\n".join(links)
        return f"{text}\n\n{joined}" if text else joined

    def _save_figures(self, result: OCRResult, doc_dir: Path) -> dict[int, list[str]]:
        """Persist extracted images as PNG; return per-page resolving md links.

        Figures are numbered globally (``figure_1``, ``figure_2``, ...) across the
        whole document and tagged with their source page, matching the canonical
        ``figure_<N>_page<P>.png`` naming.
        """
        if not (result.page_images and self.config.include_images):
            return {}

        figures_dir = figures_dir_for(doc_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        links: dict[int, list[str]] = {}
        figure_counter = 0
        for page_no in sorted(result.page_images):
            for raw in result.page_images[page_no]:
                figure_counter += 1
                filename = figure_filename(figure_counter, page_no)
                img_path = figures_dir / filename
                try:
                    image = Image.open(io.BytesIO(raw))
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")
                    image.save(img_path, format="PNG")
                    links.setdefault(page_no, []).append(
                        figure_markdown_link(figure_counter, page_no)
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to save figure %d (page %d): %s", figure_counter, page_no, exc
                    )
                    figure_counter -= 1
        return links

    def _run_fingerprint(self) -> str:
        """Fingerprint of the run config that affects *what output is produced*.

        Beyond model + backend, mistral's OCR-3 toggles (table format, header /
        footer extraction, embedded-image extraction) change the produced
        markdown, so they are folded into the contract's ``task`` selector. A
        re-run under a different ``--table-format`` / ``--extract-headers`` /
        ``--model`` therefore reprocesses instead of silently reusing a cached
        result keyed only on the input checksum.
        """
        task = (
            f"table={self.config.table_format or ''}"
            f";header={int(self.config.extract_header)}"
            f";footer={int(self.config.extract_footer)}"
            f";images={int(self.config.include_images)}"
        )
        fingerprint: str = run_fingerprint(model=self.config.model, backend=BACKEND, task=task)
        return fingerprint

    def _build_doc_metadata(
        self, result: OCRResult, markdown_path: Path, output_root: Path
    ) -> DocMetadata:
        """Assemble the per-document metadata record from a result."""
        status = result.status
        error = result.error if status is not Status.COMPLETED else None
        return DocMetadata(
            status=status,
            checksum=sha256_checksum(result.file_path),
            model=self.config.model,
            backend=BACKEND,
            processing_time=result.processing_time,
            timestamp=utc_timestamp(),
            output_path=str(markdown_path.relative_to(output_root)),
            pages=result.page_count,
            error=error,
            fingerprint=self._run_fingerprint(),
        )

    def _persist(
        self, result: OCRResult, output_root: Path, rel_key: str, index: RootIndex
    ) -> tuple[DocMetadata, Path]:
        """Write markdown, figures, and BOTH metadata levels for one document.

        Always writes output (markdown + per-doc + root metadata) regardless of
        OCR success, so failures are recorded with ``status=failed`` per the canon.

        Persistence is also failure-isolated: if a save / metadata-write / index
        record raises (e.g. a disk-full / permission / OSError), the exception is
        NOT allowed to escape and abort the whole batch. Instead it is folded into
        a ``status=failed`` record (best-effort persisted, both metadata levels),
        and a failed :class:`DocMetadata` is returned so the caller marks this one
        document failed and the batch continues — uniform per-file failure
        accounting that covers I/O failures during persistence, not just OCR.
        """
        markdown_path = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
        try:
            markdown_path = self.save_results(result, output_root, rel_key)
            meta = self._build_doc_metadata(result, markdown_path, output_root)
            doc_dir = doc_dir_for(output_root, rel_key)
            write_doc_metadata(doc_dir, rel_key, meta)
            with self._lock:
                index.record(rel_key, meta)
            return meta, markdown_path
        except Exception as exc:
            logger.error("Failed to persist output for %s: %s", rel_key, exc)
            logger.debug("Persistence traceback for %s", rel_key, exc_info=True)
            failed_meta = self._build_failed_persist_metadata(
                result, markdown_path, output_root, exc
            )
            self._best_effort_record_failure(output_root, rel_key, index, failed_meta)
            return failed_meta, markdown_path

    def _build_failed_persist_metadata(
        self, result: OCRResult, markdown_path: Path, output_root: Path, exc: Exception
    ) -> DocMetadata:
        """Build a ``status=failed`` record for a persistence I/O failure."""
        try:
            output_rel = str(markdown_path.relative_to(output_root))
        except ValueError:
            output_rel = str(markdown_path)
        try:
            checksum = sha256_checksum(result.file_path)
        except OSError:
            checksum = "sha256:"
        return DocMetadata(
            status=Status.FAILED,
            checksum=checksum,
            model=self.config.model,
            backend=BACKEND,
            processing_time=result.processing_time,
            timestamp=utc_timestamp(),
            output_path=output_rel,
            pages=result.page_count,
            error=f"persistence failed: {exc}",
            fingerprint=self._run_fingerprint(),
        )

    def _best_effort_record_failure(
        self, output_root: Path, rel_key: str, index: RootIndex, meta: DocMetadata
    ) -> None:
        """Try to persist a failed record without ever re-raising.

        The original persistence already failed, so these writes may also fail;
        each is isolated so the batch still continues and the in-memory
        :class:`RunOutcome` (driven by the returned meta) remains authoritative
        even when nothing reaches disk.
        """
        try:
            write_doc_metadata(doc_dir_for(output_root, rel_key), rel_key, meta)
        except Exception:
            logger.debug("Best-effort per-doc failure write also failed for %s", rel_key)
        try:
            with self._lock:
                index.record(rel_key, meta)
        except Exception:
            logger.debug("Best-effort root-index failure write also failed for %s", rel_key)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        reprocess: bool = False,
    ) -> RunOutcome:
        """Process an input path (file or directory). Returns a RunOutcome.

        Output goes to ``resolve_output_root(input_path, output_path)`` — default
        ``<input-parent>/ocr/``; ``-o`` overrides; never required. The returned
        :class:`RunOutcome` carries the uniform exit policy: nonzero if any file
        failed, across both single-file and batch runs.
        """
        if input_path.is_file():
            return self._process_single_file(input_path, output_path, reprocess)
        if input_path.is_dir():
            return self._process_directory(input_path, output_path, reprocess)
        raise ValueError(f"Input path does not exist: {input_path}")

    def _process_single_file(
        self, file_path: Path, output_path: Path | None, reprocess: bool
    ) -> RunOutcome:
        """Process a single file. Scan root is the file's parent (rel key = name)."""
        outcome = RunOutcome()
        output_root = resolve_output_root(file_path, output_path)
        output_root.mkdir(parents=True, exist_ok=True)
        rel_key = relative_key(file_path, file_path.parent)
        index = RootIndex(output_root)

        # Fix (audit HIGH): content-aware skip — status==completed AND checksum
        # match AND output still on disk AND a matching run fingerprint (model /
        # OCR-3 toggles) — so an in-place edit, a deleted output, or a config
        # change all force reprocessing instead of serving stale output.
        if not reprocess and index.is_completed(
            rel_key, sha256_checksum(file_path), fingerprint=self._run_fingerprint()
        ):
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            # Emit the existing .md path so the quiet scripting contract still
            # reports a path for skipped-but-valid docs (one .md path per line).
            doc_dir = doc_dir_for(output_root, rel_key)
            outcome.add(Status.COMPLETED, output_path=str(markdown_path_for(doc_dir, rel_key)))
            return outcome

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        with self._progress() as progress:
            if progress is not None:
                progress.add_task(f"OCR {file_path.name}", total=None)
            result = self.process_file(file_path)

        meta, markdown_path = self._persist(result, output_root, rel_key, index)
        # Only successful outputs land in outcome.outputs (the quiet stdout
        # scripting list). A failed doc's placeholder .md path is NOT echoed to
        # stdout as if it succeeded; the failure goes to stderr instead.
        if meta.status is Status.COMPLETED:
            outcome.add(Status.COMPLETED, output_path=str(markdown_path))
            console.print("\n[green]Success[/green]")
            console.print(f"[dim]Time: {result.processing_time:.2f}s[/dim]")
        else:
            outcome.add(meta.status, detail=rel_key)
            console.print(f"\n[red]Failed:[/red] {meta.error}")
            err_console.print(f"FAILED {rel_key}: {meta.error}")
        return outcome

    def _process_directory(
        self, dir_path: Path, output_path: Path | None, reprocess: bool
    ) -> RunOutcome:
        """Process all supported files in a directory, keyed on input-rel paths."""
        outcome = RunOutcome()
        output_root = resolve_output_root(dir_path, output_path)

        # Discovery (contract iter_input_files) excludes the resolved output root
        # so prior outputs are never re-ingested; it does NOT skip arbitrary dirs
        # merely named 'ocr'.
        files = get_supported_files(dir_path, output_root)
        if not files:
            console.print("[yellow]No supported files found in the directory.[/yellow]")
            return outcome

        output_root.mkdir(parents=True, exist_ok=True)
        index = RootIndex(output_root)

        files_to_process: list[tuple[Path, str]] = []
        for f in files:
            rel_key = relative_key(f, dir_path)
            if not reprocess and index.is_completed(
                rel_key, sha256_checksum(f), fingerprint=self._run_fingerprint()
            ):
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {rel_key}[/dim]")
                # Emit the existing .md path so quiet scripting still reports a
                # path for skipped-but-valid docs.
                doc_dir = doc_dir_for(output_root, rel_key)
                outcome.add(Status.COMPLETED, output_path=str(markdown_path_for(doc_dir, rel_key)))
            else:
                files_to_process.append((f, rel_key))

        if not files_to_process:
            console.print("[green]All files already processed.[/green]")
            console.print("[dim]Use --reprocess to force reprocessing.[/dim]")
            return outcome

        workers = self.config.max_workers
        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        if workers > 1:
            console.print(f"[blue]Using {workers} concurrent workers[/blue]")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        start_time = time.time()
        with self._progress(total=len(files_to_process)) as progress:
            task = (
                progress.add_task("Processing files...", total=len(files_to_process))
                if progress is not None
                else None
            )

            if workers <= 1:
                for file_path, rel_key in files_to_process:
                    if progress is not None and task is not None:
                        size = format_file_size(file_path.stat().st_size)
                        progress.update(task, description=f"Processing {rel_key} ({size})...")
                    result = self.process_file(file_path)
                    self._record(result, output_root, rel_key, index, outcome)
                    if progress is not None and task is not None:
                        progress.update(task, advance=1)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(self.process_file, fp): (fp, rk)
                        for fp, rk in files_to_process
                    }
                    for future in as_completed(futures):
                        _fp, rel_key = futures[future]
                        result = future.result()
                        self._record(result, output_root, rel_key, index, outcome)
                        if progress is not None and task is not None:
                            progress.update(task, advance=1)

        total = outcome.completed + outcome.failed + outcome.partial
        console.print(f"\n[green]Completed:[/green] {outcome.completed}/{total} files")
        if outcome.has_failures:
            console.print(f"[red]Failures:[/red] {outcome.failed} failed")
        console.print(f"[dim]Total time: {time.time() - start_time:.2f}s[/dim]")
        return outcome

    def _record(
        self,
        result: OCRResult,
        output_root: Path,
        rel_key: str,
        index: RootIndex,
        outcome: RunOutcome,
    ) -> None:
        """Persist one result and fold its status into the run outcome.

        Only successful docs contribute their .md path to ``outcome.outputs`` (the
        quiet stdout scripting list); failures are routed to stderr instead, so a
        scripting consumer never mistakes a failed placeholder for a success.
        """
        meta, markdown_path = self._persist(result, output_root, rel_key, index)
        if meta.status is Status.COMPLETED:
            outcome.add(Status.COMPLETED, output_path=str(markdown_path))
            console.print(f"  [green]OK[/green] {rel_key} ({result.processing_time:.1f}s)")
        else:
            outcome.add(meta.status, detail=rel_key)
            console.print(f"  [red]FAILED[/red] {rel_key}: {meta.error}")
            err_console.print(f"FAILED {rel_key}: {meta.error}")

    def _progress(self, total: int | None = None) -> Progress | _NullProgress:
        """A rich Progress, or a no-op context when quiet."""
        if self.config.quiet:
            return _NullProgress()
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )


class _NullProgress:
    """A context manager standing in for rich.Progress under --quiet."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False
