"""Server-side page rendering with source-term highlighting.

Renders the stored PDF page to PNG via pypdfium2 and overlays highlight
boxes on words that match the cited quote (positions from pdfplumber,
cached per page).  Works offline in any environment — no browser PDF plugin,
no CDN, and it renders identically in the sandboxed Streamlit preview.
"""
from __future__ import annotations

import io
import json
import re
import threading
from pathlib import Path

from .logging_setup import get_logger

log = get_logger("rendering")

_LOCK = threading.Lock()


MISSING_RENDERER_MSG = (
    "Page rendering (and its source highlighting) cannot import pypdfium2 and Pillow even though "
    "both are declared in this project's base dependency set — a partial or trimmed install is the "
    'usual cause. Reinstall with:  pip install -e ".[dev]"   (or:  pip install pypdfium2 Pillow). '
    "Text search, Q&A and verification do not need them."
)


def _import_render_deps():
    """Import the optional render deps, or exit with an actionable message."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image, ImageDraw  # noqa: F401 (Image proves Pillow is present)
    except ImportError as exc:      # pragma: no cover - exercised in tests with a stubbed module
        raise SystemExit(MISSING_RENDERER_MSG) from exc
    return pdfium, Image, ImageDraw


def _word_cache_path(renders_dir: Path, doc_id: int, page: int) -> Path:
    return renders_dir / f"words_{doc_id}_{page}.json"


def _render_cache_path(renders_dir: Path, doc_id: int, page: int, scale: float) -> Path:
    return renders_dir / f"page_{doc_id}_{page}_{scale:.2f}.png"


def page_words(stored_pdf: Path, page: int) -> list[dict]:
    """Words with boxes (PDF points, origin top-left) for a 1-based page."""
    import pdfplumber
    words: list[dict] = []
    with pdfplumber.open(str(stored_pdf)) as pdf:
        if page < 1 or page > len(pdf.pages):
            return []
        for w in pdf.pages[page - 1].extract_words():
            words.append({"t": w["text"], "x0": round(w["x0"], 1),
                          "x1": round(w["x1"], 1), "top": round(w["top"], 1),
                          "bottom": round(w["bottom"], 1)})
    return words


def get_page_words(renders_dir: Path, stored_pdf: Path, doc_id: int, page: int) -> list[dict]:
    cache = _word_cache_path(renders_dir, doc_id, page)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    words = page_words(stored_pdf, page)
    try:
        cache.write_text(json.dumps(words), encoding="utf-8")
    except OSError:
        pass
    return words


def highlight_boxes(words: list[dict], terms: list[str]) -> list[dict]:
    """Bounding boxes for citation terms on one page.

    A cited term is usually spread over several extracted words ("132/33 kV" is
    two words, a table cell may be three), so each term is normalised into
    tokens and matched against *consecutive* words that sit on the same visual
    line.  Word-by-word equality would silently skip exactly the numeric terms
    the highlight is there for.
    """
    def clean(tok: str) -> str:
        return re.sub(r"[^a-z0-9°²Ω%.\-]", "", tok.lower())

    pats: list[list[str]] = []
    for t in terms:
        toks = [x for x in (clean(w) for w in re.split(r"[\s,;]+", str(t).strip())) if x]
        if toks:
            pats.append(toks)
    if not pats or not words:
        return []
    keys = [clean(w["t"]) for w in words]
    boxes: list[dict] = []
    seen: set[tuple] = set()
    for pat in pats:
        n = len(pat)
        for i in range(len(keys) - n + 1):
            if keys[i:i + n] != pat:
                continue
            span = words[i:i + n]
            if any(abs(w["top"] - span[0]["top"]) > 2.5 for w in span[1:]):
                continue          # phrase split across lines: do not draw a band
            key = (round(span[0]["top"], 1), round(span[-1]["bottom"], 1),
                   round(min(w["x0"] for w in span), 1),
                   round(max(w["x1"] for w in span), 1))
            if key in seen:
                continue
            seen.add(key)
            boxes.append({"x0": key[2] - 1.2, "x1": key[3] + 1.2,
                          "top": key[0] - 1.0, "bottom": key[1] + 1.0})
    boxes.sort(key=lambda b: (b["top"], b["x0"]))
    return boxes[:40]


def render_page_png(stored_pdf: Path, doc_id: int, page: int, *,
                    renders_dir: Path, scale: float = 1.6,
                    max_pixels: int = 22_000_000,
                    highlight_terms: list[str] | None = None,
                    page_words_list: list[dict] | None = None,
                    use_cache: bool = True) -> bytes | None:
    """Render one physical page (1-based) to PNG; None if out of range."""
    renders_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _render_cache_path(renders_dir, doc_id, page, scale)
    if use_cache and not highlight_terms and cache_file.exists():
        return cache_file.read_bytes()
    pdfium, _Image, _ImageDraw = _import_render_deps()
    with _LOCK:
        pdf = pdfium.PdfDocument(str(stored_pdf))
        try:
            n = len(pdf)
            if page < 1 or page > n:
                return None
            # pixels per PDF point: the render applies ``scale``, so word boxes
            # (measured in points by pdfplumber) must be multiplied by it too
            pl = float(scale)
            p = pdf[page - 1]
            img = p.render(scale=scale, rotation=0).to_pil()
        finally:
            pdf.close()
    if img.width * img.height > max_pixels:
        factor = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((int(img.width * factor), int(img.height * factor)))
        pl = scale * factor
    if highlight_terms:
        try:
            words = page_words_list if page_words_list is not None else \
                get_page_words(renders_dir, stored_pdf, doc_id, page)
            draw = _ImageDraw.Draw(img, "RGBA")
            for box in highlight_boxes(words, highlight_terms):
                draw.rectangle([box["x0"] * pl, box["top"] * pl,
                                box["x1"] * pl, box["bottom"] * pl],
                               fill=(255, 224, 70, 96),
                               outline=(196, 141, 0, 220))
        except Exception as exc:
            log.warning("highlighting failed (rendering page plain): %s", exc)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG", optimize=True)
    data = buf.getvalue()
    if use_cache and not highlight_terms:
        try:
            cache_file.write_bytes(data)
        except OSError:
            pass
    return data
