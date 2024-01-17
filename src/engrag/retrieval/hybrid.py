"""Hybrid retrieval: FTS5 BM25 + semantic vectors + deterministic rerank.

Scores are *relevance signals*, not calibrated probabilities.  Every returned
hit carries its score breakdown so the UI can show exactly why something was
retrieved.  Chunk text is re-read from the primary ``document_chunks`` table on
every query, and citations carry ``text_sha1`` so stale index rows can be
detected (and repaired by ``ensure_fts``) rather than silently trusted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from ..db import Database
from ..logging_setup import get_logger
from .embeddings import EmbeddingService, sentence_transformers_available
from .keyword import KeywordRetriever, content_tokens, extract_quoted_phrases

log = get_logger("retrieval.hybrid")

_NUMBER_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:[kMmµ]?[VWAΩ°%]|kVA|MW|kW|Hz|mm²|mm2|A|Ω)", re.I)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    page_start: int
    page_end: int
    kind: str
    text: str
    text_sha1: str
    score: float
    lexical: float
    semantic: float
    coverage: float
    matched_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "document_id": self.document_id,
            "document_title": self.document_title, "page_start": self.page_start,
            "page_end": self.page_end, "kind": self.kind, "text": self.text,
            "text_sha1": self.text_sha1, "score": round(self.score, 4),
            "lexical": round(self.lexical, 4), "semantic": round(self.semantic, 4),
            "coverage": round(self.coverage, 4), "matched_pages": self.matched_pages,
            "score_kind": "composite_relevance_not_probability",
        }


class SearchEngine:
    def __init__(self, db: Database, settings: Settings,
                 embedder: EmbeddingService | None = None):
        self.db = db
        self.s = settings
        self.keyword = KeywordRetriever(db)
        self.embedder = embedder or EmbeddingService(
            prefer_sentence_transformers=sentence_transformers_available(),
            model_id=settings.embedding_model)

    # ------------------------------------------------------------- indexing
    def index_document(self, doc_id: int, chunks: list[dict]) -> None:
        """Store chunks (primary truth) and refresh derived indexes."""
        self.db.store_chunks(doc_id, chunks)
        self.fit_idf()
        try:
            self.update_vectors(doc_id)
        except Exception as exc:
            log.warning("vector indexing failed (keyword search still works): %s", exc)

    def fit_idf(self) -> None:
        """(Re)fit the hashing embedder's IDF table on the current corpus."""
        emb = self.embedder
        emb._ensure()
        if emb.status.backend != "hashing-ngrams":
            return
        rows = self.db.query("SELECT id, text FROM document_chunks LIMIT 20000")
        if rows:
            emb.fit([r["text"] for r in rows])

    def update_vectors(self, doc_id: int | None = None) -> int:
        """Embed chunks missing vectors; returns number embedded."""
        emb = self.embedder
        tag = emb.model_tag
        if doc_id is None:
            rows = self.db.query(
                "SELECT c.id, c.text FROM document_chunks c WHERE c.id NOT IN "
                "(SELECT chunk_id FROM chunk_vectors WHERE model = ?) "
                "ORDER BY c.id", (tag,))
        else:
            rows = self.db.query(
                "SELECT c.id, c.text FROM document_chunks c WHERE c.document_id = ? "
                "AND c.id NOT IN (SELECT chunk_id FROM chunk_vectors WHERE model = ?) "
                "ORDER BY c.id", (doc_id, tag))
        if not rows:
            return 0
        vectors = emb.embed([r["text"] for r in rows])
        self.db.replace_vectors(tag, [(int(r["id"]), vectors[i])
                                      for i, r in enumerate(rows)])
        return len(rows)

    def reindex_all(self) -> dict:
        self.db.ensure_fts()
        self.fit_idf()
        embedded = 0
        try:
            embedded = self.update_vectors()
        except Exception as exc:
            log.warning("reindex vectors failed: %s", exc)
        return {"embedded_chunks": embedded,
                "embedding_backend": self.embedder.status.backend}

    # --------------------------------------------------------------- vector
    def _vector_scores(self, query: str) -> np.ndarray | None:
        tag = self.embedder.model_tag
        ids, mat = self.db.all_vectors(tag)
        if mat is None:
            return None
        try:
            qv = self.embedder.embed([query])[0]
        except Exception as exc:
            log.warning("query embedding failed: %s", exc)
            return None
        return mat @ qv                        # cosine (vectors L2-normalized)

    # ------------------------------------------------------------- query
    def search(self, query: str, *, top_k: int | None = None,
               document_ids: list[int] | None = None,
               collection_id: int | None = None,
               threshold: float | None = None) -> list[RetrievedChunk]:
        s = self.s
        top_k = min(max(1, top_k or s.default_top_k), s.max_top_k)
        threshold = s.relevance_threshold if threshold is None else threshold
        pool = s.candidate_pool

        kw_hits = self.keyword.search(query, limit=pool,
                                      document_ids=document_ids,
                                      collection_id=collection_id)
        kw_by_chunk = {h.chunk_id: h for h in kw_hits}

        # Always include chunks that contain every numeric token of the query
        # (a table row that FTS tokenizer hiccuped on must not be lost).
        extra = self._numeric_candidates(query, document_ids, collection_id, pool)
        for cid in extra:
            kw_by_chunk.setdefault(cid, KeywordHitStub(cid))

        vec_scores = self._vector_scores(query)
        tag = self.embedder.model_tag
        if vec_scores is not None and self.s.enable_semantic:
            ids, mat = self.db.all_vectors(tag)
            restrict = None
            if document_ids or collection_id is not None:
                allowed = self._allowed_ids(document_ids, collection_id)
                restrict = np.isin(np.asarray(ids), np.fromiter(allowed, dtype=np.int64,
                                                                count=len(allowed))) \
                    if ids else None
                if ids and restrict is not None:
                    vec_scores = np.where(restrict, vec_scores, -1.0)
            sem_ranked = sorted(
                ((cid, float(v)) for cid, v in zip(ids, vec_scores) if v > 0.05),
                key=lambda t: -t[1])[:pool]
        else:
            sem_ranked = []
        sem_by_chunk = {cid: max(0.0, v) for cid, v in sem_ranked}

        candidates = set(kw_by_chunk) | set(sem_by_chunk)
        q_tokens = set(content_tokens(query))
        q_phrases = extract_quoted_phrases(query)
        q_numbers = re.findall(r"\d+(?:[.,]\d+)?\s*(?:kV|MV|VA|W|var|A|Hz|°C|mm²|mm2|%|Ω)",
                               query, re.I)
        wants_table = bool(q_numbers) or "table" in query.lower()

        rows = self._load_rows(candidates)
        results: list[RetrievedChunk] = []
        for r in rows:
            cid = int(r["id"])
            text = r["text"]
            kw = kw_by_chunk.get(cid)
            lexical = kw.score if kw else 0.0
            semantic = sem_by_chunk.get(cid, 0.0)
            c_tokens = set(content_tokens(text))
            coverage = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            boost = 0.0
            if q_phrases and any(
                    phrase.lower() in " ".join(text.lower().split()) for phrase in q_phrases):
                boost += 0.25
            if r["kind"] == "table" and wants_table:
                boost += 0.15
            for n in q_numbers:
                if normalize_ws(n) in normalize_ws(text):
                    boost += 0.10
                    break
            # title match helps "compare TR-01 and TR-02" style queries
            if r["title"] and any(t in r["title"].lower() for t in q_tokens if len(t) > 3):
                boost += 0.05
            score = 0.55 * lexical + 0.30 * semantic + 0.15 * coverage + boost
            matched_pages = [r["page_start"]] if r["page_start"] == r["page_end"] \
                else locate_pages(r["page_start"], r["page_end"], text, query,
                                  r["doc_id"], self.db)
            results.append(RetrievedChunk(
                chunk_id=cid, document_id=int(r["doc_id"]),
                document_title=r["title"], page_start=int(r["page_start"]),
                page_end=int(r["page_end"]), kind=r["kind"], text=text,
                text_sha1=r["text_sha1"], score=min(1.0, score),
                lexical=lexical, semantic=semantic, coverage=coverage,
                matched_pages=matched_pages))
        results.sort(key=lambda x: -x.score)
        results = [x for x in results if x.score >= threshold]
        return results[:top_k]

    # ------------------------------------------------------------ helpers
    def _load_rows(self, candidates: set[int]) -> list[dict]:
        if not candidates:
            return []
        marks = ",".join("?" * len(candidates))
        rows = self.db.query(
            f"""SELECT c.id, c.document_id AS doc_id, c.page_start, c.page_end,
                       c.kind, c.text, c.text_sha1, d.title
                  FROM document_chunks c JOIN documents d ON d.id = c.document_id
                 WHERE c.id IN ({marks})""", tuple(candidates))
        return [dict(r) for r in rows]

    def _allowed_ids(self, document_ids, collection_id) -> set[int]:
        if document_ids:
            return set(int(d) for d in document_ids)
        rows = self.db.query("SELECT id FROM documents WHERE collection_id = ?",
                             (collection_id,))
        return {int(r["id"]) for r in rows}

    def _numeric_candidates(self, query, document_ids, collection_id, limit) -> list[int]:
        nums = [n for n in re.findall(r"\d+(?:[.,]\d+)", query) if len(n) >= 2]
        if not nums:
            return []
        conds, params = [], []
        if document_ids:
            conds.append("c.document_id IN (" + ",".join("?" * len(document_ids)) + ")")
            params.extend(int(d) for d in document_ids)
        if collection_id is not None:
            conds.append("d.collection_id = ?")
            params.append(collection_id)
        where = (" AND " + " AND ".join(conds)) if conds else ""
        like_sql = " OR ".join(["c.text LIKE ?"] * len(nums[:6]))
        params.extend(f"%{n}%" for n in nums[:6])
        rows = self.db.query(
            f"""SELECT c.id FROM document_chunks c JOIN documents d
                    ON d.id = c.document_id
                 WHERE ({like_sql}){where} LIMIT ?""", (*params, limit))
        return [int(r["id"]) for r in rows]


class KeywordHitStub:
    """Marker for chunks recalled outside FTS (numeric fallback); no BM25."""

    def __init__(self, chunk_id: int):
        self.chunk_id = chunk_id
        self.bm25_rank = 0.0
        self.score = 0.12      # weak prior; rerank/coverage can lift it


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def locate_pages(page_start: int, page_end: int, text: str, query: str,
                 doc_id: int, db: Database) -> list[int]:
    """Narrow a multi-page chunk to the pages actually matching query terms."""
    if page_start == page_end:
        return [page_start]
    terms = set(content_tokens(query))
    best: list[tuple[int, int]] = []
    for p in range(page_start, page_end + 1):
        row = db.get_page(doc_id, p)
        if not row:
            continue
        overlap = len(terms & set(content_tokens(row["text"])))
        best.append((overlap, p))
    if not best:
        return [page_start]
    best.sort(reverse=True)
    top = best[0][0]
    return sorted(p for o, p in best if o == top) or [page_start]


def build_search_excerpt(text: str, query: str, max_chars: int = 480) -> str:
    """Window of the chunk around the most query-relevant region."""
    tokens = content_tokens(query)
    low = text.lower()
    best_pos, best_score = -1, -1
    for t in tokens:
        pos = low.find(t)
        if pos < 0:
            continue
        score = sum(1 for tt in tokens if tt in low[max(0, pos - 200):pos + 200])
        if score > best_score:
            best_score, best_pos = score, pos
    if best_pos < 0:
        excerpt = text[:max_chars]
    else:
        start = max(0, best_pos - max_chars // 3)
        excerpt = text[start:start + max_chars]
        if start > 0:
            excerpt = "…" + excerpt
        if start + max_chars < len(text):
            # never cut mid-token: units like mm/s or kVA must stay whole
            tail = text[start + max_chars:]
            if tail and not tail[0].isspace() and excerpt and not excerpt[-1].isspace():
                nxt = re.search(r"\s", tail)
                excerpt += tail[: nxt.start()] if nxt else ""
            excerpt += "…"
    return excerpt
