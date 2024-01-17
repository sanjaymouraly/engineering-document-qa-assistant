"""PDF text and table extraction with graceful degradation.

Pipeline per page:
  1. ``pypdf`` text layer (fast, preserves reading order reasonably well).
  2. If the pypdf text layer looks empty/thin, retry with ``pdfplumber``
     (word-position aware, better for two-column and table-heavy pages).
  3. Extracted tables (pdfplumber) are rendered to pipe-delimited text so
     numeric table values become searchable and quotable.
  4. If configured and Tesseract is installed, pages that still look like
     scans are OCR'd; pages that cannot be read are flagged ``needs_ocr``.

Printed page labels (front-matter roman numerals, "P-3" style numbering, …)
are captured when present so the UI can show both physical and printed pages.
"""
from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass, field

from ..logging_setup import get_logger

log = get_logger("ingest.extract")

_TABLE_MARK = "TABLE (page {page}):"
_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Whitespace normalization used everywhere text is compared or stored."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")          # soft hyphens
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


@dataclass
class PageResult:
    page_number: int                  # 1-based physical page
    text: str = ""
    printed_label: str = ""
    extraction_method: str = "none"   # pypdf | pdfplumber | ocr | none
    needs_ocr: bool = False
    tables: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ExtractionResult:
    pages: list[PageResult] = field(default_factory=list)
    encrypted: bool = False
    error: str | None = None
    ocr_used: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def needs_ocr_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.needs_ocr]

    @property
    def ocr_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.extraction_method == "ocr"]

    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


def looks_like_scan(text: str) -> bool:
    """A page with almost no alphanumeric characters cannot be indexed."""
    alnum = sum(ch.isalnum() for ch in text)
    return alnum < 20


# --------------------------------------------------------------------- OCR
def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page_image(page, min_size_px: int = 1200) -> str:
    """OCR a pdfplumber page via pytesseract (best-effort, optional extra)."""
    try:
        import pytesseract
        from PIL import Image  # noqa: F401  (validated at import time)
    except Exception:
        return ""
    try:
        img = page.to_image(resolution=220).original
        w, h = img.size
        if min(w, h) < min_size_px and w * h < 4_000_000:
            scale = min_size_px / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        return normalize_text(pytesseract.image_to_string(img))
    except Exception as exc:  # pragma: no cover - depends on external binary
        log.warning("OCR failed: %s", exc)
        return ""


# ----------------------------------------------------------------- extraction
def _pdfplumber_table_text(table: list[list[str | None]]) -> str:
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
            for row in table if row]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return "\n".join(" | ".join(r) for r in rows)


def extract_pdf(path_or_bytes, *, enable_ocr: bool = False,
                max_pages: int = 1500,
                max_chars: int = 6_000_000) -> ExtractionResult:
    """Extract pages, labels and tables from a PDF file."""
    import pypdf
    import pdfplumber

    result = ExtractionResult()
    if isinstance(path_or_bytes, (str, bytes, bytearray)):
        if isinstance(path_or_bytes, (bytes, bytearray)):
            src: io.BytesIO | str = io.BytesIO(bytes(path_or_bytes))
        else:
            src = str(path_or_bytes)
    else:
        src = path_or_bytes  # already a file-like object

    try:
        reader = pypdf.PdfReader(src, strict=False)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    result.encrypted = True
                    result.error = ("The PDF is encrypted with a user password. "
                                    "Remove the password and upload again.")
                    return result
            except Exception:
                result.encrypted = True
                result.error = "The PDF is encrypted and could not be opened."
                return result
        info = reader.metadata or {}
        result.metadata = {
            "title": (info.title or "").strip(),
            "author": (info.author or "").strip(),
            "producer": (info.producer or "").strip(),
        }
        labels: list[str] = []
        try:
            labels = list(reader.page_labels or [])
        except Exception:
            labels = []
        n = min(len(reader.pages), max_pages)
        texts: list[str] = []
        for i in range(n):
            try:
                texts.append(normalize_text(reader.pages[i].extract_text() or ""))
            except Exception as exc:
                log.warning("pypdf page %d failed: %s", i + 1, exc)
                texts.append("")
    except Exception as exc:
        result.error = f"Could not parse PDF ({type(exc).__name__}: {exc}). " \
                       "The file may be corrupt or not a PDF at all."
        return result

    # Second pass with pdfplumber: fallback text + tables.
    plumber_pages_text: list[str] = []
    plumber_tables: list[list[str]] = []
    try:
        with pdfplumber.open(path_or_bytes) as pdf:
            for i, page in enumerate(pdf.pages[:n]):
                try:
                    plumber_pages_text.append(normalize_text(page.extract_text() or ""))
                except Exception:
                    plumber_pages_text.append("")
                tables_txt: list[str] = []
                try:
                    for t in (page.find_tables() or [])[:6]:
                        txt = _pdfplumber_table_text(t.extract())
                        if txt:
                            tables_txt.append(_TABLE_MARK.format(page=i + 1) + "\n" + txt)
                except Exception as exc:
                    log.debug("table extraction failed p%d: %s", i + 1, exc)
                plumber_tables.append(tables_txt)
    except Exception as exc:  # pdfplumber is a fallback; keep pypdf results
        log.warning("pdfplumber pass failed: %s", exc)
        plumber_pages_text = [""] * n
        plumber_tables = [[] for _ in range(n)]

    total_chars = 0
    ocr_enabled = enable_ocr and tesseract_available()
    if enable_ocr and not ocr_enabled:
        log.warning("OCR requested but tesseract binary/pytesseract not found; "
                    "affected pages will be flagged needs_ocr instead.")

    for i in range(n):
        pr = PageResult(page_number=i + 1)
        if i < len(labels) and labels[i] not in ("", str(i + 1)):
            pr.printed_label = str(labels[i])
        text = plumber_pages_text[i] if i < len(plumber_pages_text) else ""
        pypdf_text = texts[i] if i < len(texts) else ""
        if len(pypdf_text) >= len(text):
            pr.text, pr.extraction_method = pypdf_text, "pypdf"
        elif text:
            pr.text, pr.extraction_method = text, "pdfplumber"
        if i < len(plumber_tables):
            pr.tables = plumber_tables[i]
        if looks_like_scan(pr.text) and i < len(plumber_pages_text):
            # pypdf thin but pdfplumber richer?
            alt = plumber_pages_text[i]
            if len(alt) > len(pr.text):
                pr.text, pr.extraction_method = alt, "pdfplumber"
        if looks_like_scan(pr.text) and ocr_enabled:
            try:
                with pdfplumber.open(path_or_bytes) as pdf:
                    ocr_text = ocr_page_image(pdf.pages[i])
                if ocr_text and not looks_like_scan(ocr_text):
                    pr.text, pr.extraction_method, pr.needs_ocr = \
                        ocr_text, "ocr", False
                    result.ocr_used = True
            except Exception as exc:  # pragma: no cover
                log.warning("OCR pass failed on page %d: %s", i + 1, exc)
        if looks_like_scan(pr.text):
            pr.needs_ocr = True
            pr.text = pr.text or ""
        if total_chars + len(pr.text) > max_chars:
            log.warning("Extraction stopped at page %d: char budget "
                        "(ENGRAG_MAX_EXTRACTED_CHARS_PER_DOC) exceeded.", i + 1)
            result.error = (f"Extraction truncated at page {i + 1} because the "
                            f"per-document character budget ({max_chars:,}) was "
                            "reached. Adjust ENGRAG_MAX_EXTRACTED_CHARS_PER_DOC "
                            "if this is intentional.")
            result.pages.append(pr)
            break
        total_chars += len(pr.text)
        result.pages.append(pr)

    if result.page_count == 0 and result.error is None:
        result.error = "The PDF contains no pages."
    return result
