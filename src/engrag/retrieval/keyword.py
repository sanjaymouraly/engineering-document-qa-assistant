"""Keyword retrieval via SQLite FTS5 (deterministic, no model downloads).

FTS5's BM25 ``rank`` is a relevance signal, not a calibrated probability: we
map it to ``[0, 1]`` monotonically and always report it as
``score_kind='bm25_normalized'`` so nothing downstream pretends it is a
likelihood.  A pure-Python fallback (trigram-ish inverted scoring) keeps
keyword search working even on SQLite builds without FTS5.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..db import Database
from ..logging_setup import get_logger

log = get_logger("retrieval.keyword")

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "what", "which", "how", "many", "does", "do", "did", "be", "by",
    "at", "as", "it", "its", "this", "that", "these", "those", "from", "was",
    "were", "can", "could", "should", "would", "will", "shall", "may", "might",
    "please", "tell", "me", "about", "according", "per", "vs", "versus",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.,/-][A-Za-z0-9]+)*")
_QUOTE_RE = re.compile(r'[“”„»]([^“”„»]+)[“”„«]')
_NUM_RE = re.compile(r"\d")

# Words that must never appear inside an FTS5 MATCH string unescaped.
_FTS_SPECIAL = re.compile(r'(["*()^{}\-:])')


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP]


def extract_quoted_phrases(query: str) -> list[str]:
    out = []
    for m in _QUOTE_RE.finditer(query):
        phrase = m.group(1).strip()
        if len(phrase) >= 3:
            out.append(phrase)
    for m in re.finditer(r'"([^"]{3,})"', query):
        out.append(m.group(1).strip())
    return out


def build_fts_query(query: str) -> str:
    """Compose a safe FTS5 MATCH expression.

    OR-joined terms keep recall high for natural-language questions (AND would
    silently return nothing when a user adds one stray word); BM25 ranking
    still favours documents matching more terms.
    """
    tokens = content_tokens(query)
    seen: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
        if len(seen) >= 24:
            break
    parts = [f'"{_FTS_SPECIAL.sub(r"\\\1", t)}"' for t in seen]
    for phrase in extract_quoted_phrases(query)[:3]:
        clean = re.sub(r"\s+", " ", phrase)
        parts.append(f'"{_FTS_SPECIAL.sub(r"\\\1", clean)}"^5')
    if not parts:
        return ""
    return " OR ".join(parts)


@dataclass
class KeywordHit:
    chunk_id: int
    bm25_rank: float      # raw sqlite rank (lower = better)
    score: float          # normalized to (0, 1] for ordering only


class KeywordRetriever:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ API
    def search(self, query: str, *, limit: int = 20,
               document_ids: list[int] | None = None,
               collection_id: int | None = None) -> list[KeywordHit]:
        if self.db.fts5:
            return self._fts_search(query, limit, document_ids, collection_id)
        return self._fallback_search(query, limit, document_ids, collection_id)

    def _filter_sql(self, document_ids, collection_id):
        conds, params = [], []
        if document_ids:
            conds.append("c.document_id IN (" + ",".join("?" * len(document_ids)) + ")")
            params.extend(document_ids)
        if collection_id is not None:
            conds.append("d.collection_id = ?")
            params.append(collection_id)
        # Only serve fully processed documents: failed/needs-ocr docs stay searchable
        # but pending/processing rows must not leak half-indexed content.
        conds.append("d.status IN ('ready','needs_ocr','failed')")
        return (" AND " + " AND ".join(conds)) if conds else "", params

    def _fts_search(self, query, limit, document_ids, collection_id) -> list[KeywordHit]:
        match = build_fts_query(query)
        if not match:
            return []
        where, params = self._filter_sql(document_ids, collection_id)
        sql = f"""
            SELECT m.chunk_id AS chunk_id, rank AS bm25
              FROM chunks_fts f
              JOIN fts_map m ON m.fts_rowid = f.rowid
              JOIN document_chunks c ON c.id = m.chunk_id
              JOIN documents d ON d.id = c.document_id
             WHERE chunks_fts MATCH ?{where}
             ORDER BY rank
             LIMIT ?
        """
        try:
            rows = self.db.query(sql, (match, *params, limit))
        except Exception as exc:  # malformed MATCH despite escaping
            log.warning("FTS query failed (%s); falling back to token search", exc)
            return self._fallback_search(query, limit, document_ids, collection_id)
        hits: list[KeywordHit] = []
        for r in rows:
            raw = float(r["bm25"])          # negative; more negative = better
            mag = max(0.0, -raw)
            hits.append(KeywordHit(
                chunk_id=int(r["chunk_id"]), bm25_rank=raw,
                score=1.0 - math.exp(-mag / 4.0)))
        return hits

    def _fallback_search(self, query, limit, document_ids, collection_id):
        """No FTS5 (rare builds): score rows in Python. Correct, not fast."""
        terms = set(content_tokens(query))
        for p in extract_quoted_phrases(query):
            terms.update(content_tokens(p))
        if not terms:
            return []
        where, params = self._filter_sql(document_ids, collection_id)
        rows = self.db.query(
            f"SELECT c.id AS chunk_id, c.text AS text FROM document_chunks c "
            f"JOIN documents d ON d.id = c.document_id WHERE 1=1{where} LIMIT 20000",
            params)
        scored: list[KeywordHit] = []
        for r in rows:
            tokens = set(content_tokens(r["text"]))
            overlap = terms & tokens
            if not overlap:
                continue
            score = len(overlap) / (len(terms) + 0.5 * max(0, len(tokens) / 400))
            scored.append(KeywordHit(int(r["chunk_id"]), -score * 10, score))
        scored.sort(key=lambda h: -h.score)
        return scored[:limit]
