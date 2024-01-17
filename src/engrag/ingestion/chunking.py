"""Page-aware chunking.

Chunks keep exact physical page spans (``page_start``/``page_end``), so a
citation to a chunk is always a citation to a real page range in the stored
PDF.  Extracted tables become dedicated ``table`` chunks that are never split
mid-row, which matters for engineering spec lookups.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .extract import normalize_text

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])")
_PAGE_MARK = re.compile(r"\[page \d+\]")


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    kind: str = "text"          # text | table
    char_count: int = 0
    text_sha1: str = ""


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_long_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized block at sentence boundaries, hard-split as last resort."""
    sentences = [s for s in _SENT_SPLIT.split(block) if s.strip()]
    if len(sentences) <= 1:
        sentences = [block[i:i + max_chars] for i in range(0, len(block), max_chars)]
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur.strip())
    return [o for o in out if o]


def chunk_pages(pages: list[dict], *, min_chars: int = 400,
                target_chars: int = 1500, max_chars: int = 2400,
                overlap_chars: int = 160) -> list[Chunk]:
    """Build chunks from ``[{page_number, text, tables}, ...]`` page dicts.

    Paragraphs and tables are appended in page order.  A chunk never exceeds
    ``max_chars`` unless a single unsplittable block forces it; tables are
    emitted as their own chunks with the table marker kept intact.
    """
    chunks: list[Chunk] = []
    cur_text: list[str] = []
    cur_len = 0
    cur_start = cur_end = 0
    carry = ""

    def seed() -> list[str]:
        return [carry] if carry else []

    def flush() -> None:
        """Emit the pending chunk and carry a short context tail forward."""
        nonlocal cur_text, cur_len, cur_start, cur_end, carry
        if not cur_text:
            return
        text = normalize_text("\n\n".join(cur_text))
        if text:
            chunks.append(Chunk(text=text, page_start=cur_start, page_end=cur_end,
                                kind="text", char_count=len(text),
                                text_sha1=_sha1(text)))
            if overlap_chars > 0:
                tail = text[-(overlap_chars * 3):]
                sents = _SENT_SPLIT.split(tail)
                carry = " ".join(sents[-2:])[-overlap_chars:] if len(sents) > 1 else ""
        cur_text, cur_len, cur_start, cur_end = [], 0, 0, 0

    for page in pages:
        pno = int(page["page_number"])
        # Long narrative blocks first, then the page's tables as own chunks.
        body_blocks: list[str] = []
        text = page.get("text") or ""
        text = _PAGE_MARK.sub("", text)
        for raw in re.split(r"\n\s*\n", text):
            block = normalize_text(raw)
            if len(block) >= 40:          # skip decoration / bare headers
                body_blocks.append(block)
        for block in body_blocks:
            if len(block) > max_chars:
                flush()
                parts = split_long_block(block, max_chars)
                for i, part in enumerate(parts):
                    chunks.append(Chunk(text=part, page_start=pno, page_end=pno,
                                        kind="text", char_count=len(part),
                                        text_sha1=_sha1(part)))
                continue
            if cur_len and cur_len + len(block) + 2 > target_chars:
                flush()
            if not cur_text:
                cur_text = seed()
                cur_len = sum(len(t) + 2 for t in cur_text)
                cur_start = pno
            cur_end = pno
            cur_text.append(block)
            cur_len += len(block) + 2
            if cur_len >= target_chars + 400:   # respect max even without breaks
                flush()
        # tables: dedicated chunks; split only between rows
        for table in page.get("tables") or []:
            table = table.strip()
            if not table:
                continue
            if cur_len >= min_chars:
                flush()
            rows = table.split("\n")
            header = rows[0] if rows[0].startswith("TABLE") else ""
            body_rows = rows[1:] if header else rows
            piece = [header] if header else []
            piece_len = len(header)
            for row in body_rows:
                if piece_len + len(row) + 1 > max_chars and len(piece) > (1 if header else 0):
                    t = "\n".join(piece)
                    chunks.append(Chunk(text=t, page_start=pno, page_end=pno,
                                        kind="table", char_count=len(t),
                                        text_sha1=_sha1(t)))
                    piece = [header] if header else []
                    piece_len = len(header)
                piece.append(row)
                piece_len += len(row) + 1
            if len(piece) > (1 if header else 0):
                t = "\n".join(piece)
                chunks.append(Chunk(text=t, page_start=pno, page_end=pno,
                                    kind="table", char_count=len(t),
                                    text_sha1=_sha1(t)))
    flush()
    return chunks


def chunk_document(pages: list[dict], *, min_chars: int = 400,
                   target_chars: int = 1500, max_chars: int = 2400,
                   overlap_chars: int = 160) -> list[dict]:
    """Public helper returning plain dicts ready for ``db.store_chunks``."""
    chunks = chunk_pages(pages, min_chars=min_chars, target_chars=target_chars,
                         max_chars=max_chars, overlap_chars=overlap_chars)
    return [{"page_start": c.page_start, "page_end": c.page_end, "kind": c.kind,
             "text": c.text, "text_sha1": c.text_sha1} for c in chunks]
