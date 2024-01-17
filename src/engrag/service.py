"""Service layer: everything the API and UI do, in plain testable Python."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .db import Database
from .ingestion.chunking import chunk_document
from .ingestion.extract import extract_pdf, tesseract_available
from .logging_setup import get_logger
from .rag import RagService
from .retrieval.embeddings import EmbeddingService, sentence_transformers_available
from .retrieval.hybrid import SearchEngine
from .verify.claims import AnswerVerifier
from .verify.numbers import extract_values

log = get_logger("service")

MAX_INLINE_INGEST_BYTES = 12 * 1024 * 1024   # larger uploads go to background


@dataclass
class UploadOutcome:
    document_id: int | None
    status: str            # ok | duplicate | rejected
    message: str
    info: dict | None = None


class Service:
    def __init__(self, settings: Settings, db: Database | None = None):
        self.s = settings
        self.db = db or Database.from_settings(settings)
        self.embedder = EmbeddingService(
            prefer_sentence_transformers=sentence_transformers_available(),
            model_id=settings.embedding_model)
        self.engine = SearchEngine(self.db, settings, self.embedder)
        self.verifier = AnswerVerifier(self.db,
                                       rel_tolerance=settings.default_rel_tolerance)
        self.rag = RagService(self.db, settings, self.engine)
        self._pdfium_docs: dict[int, object] = {}
        self.db.ensure_fts()

    # -------------------------------------------------------------- uploads
    def upload_pdf(self, *, filename: str, data: bytes,
                   collection_id: int | None = None, title: str | None = None,
                   is_synthetic: bool = False,
                   is_public_domain: bool = False) -> UploadOutcome:
        self.s.ensure_dirs()
        name = Path(filename or "upload.pdf").name
        name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name)
        if not name.lower().endswith(".pdf"):
            return UploadOutcome(None, "rejected", "Only .pdf files are supported "
                                 "for ingestion.")
        if not data:
            return UploadOutcome(None, "rejected", "The uploaded file is empty.")
        if len(data) > self.s.max_upload_bytes:
            return UploadOutcome(None, "rejected",
                                 f"File is {len(data)/1e6:.1f} MB; the configured "
                                 f"limit is {self.s.max_upload_mb} MB "
                                 "(ENGRAG_MAX_UPLOAD_MB).")
        if not data.lstrip()[:5] in (b"%PDF-", b"%PDF"):
            return UploadOutcome(None, "rejected",
                                 "The file does not start with a %PDF header — "
                                 "it is not a PDF or is corrupt.")
        sha = hashlib.sha256(data).hexdigest()
        dup = self.db.find_duplicate(sha)
        if dup:
            return UploadOutcome(int(dup["id"]), "duplicate",
                                 f"Identical file already stored as document "
                                 f"'{dup['title']}' (id {dup['id']}, "
                                 f"{dup['status']}).")
        stored = self.s.uploads_dir / f"{sha[:20]}_{name}"
        stored.write_bytes(data)
        doc_id = self.db.insert_document(
            filename=name, title=(title or Path(name).stem)[:200], sha256=sha,
            size_bytes=len(data), stored_path=str(stored),
            collection_id=collection_id, is_synthetic=is_synthetic,
            is_public_domain=is_public_domain)
        if len(data) <= MAX_INLINE_INGEST_BYTES:
            info = self.process_document(doc_id)
            return UploadOutcome(doc_id, "ok", "uploaded and ingested", info)
        # small background task at API layer; record job status
        self.db.update_document(doc_id, status="pending",
                                status_detail="queued (large file)")
        return UploadOutcome(doc_id, "ok", "uploaded; indexing queued",
                             {"queued": True})

    # ------------------------------------------------------------ ingestion
    def process_document(self, doc_id: int, *, enable_ocr: bool | None = None) -> dict:
        row = self.db.get_document(doc_id)
        if row is None:
            raise ValueError(f"document {doc_id} does not exist")
        path = Path(row["stored_path"])
        self.db.update_document(doc_id, status="processing", status_detail="extracting")
        try:
            ocr = (enable_ocr if enable_ocr is not None
                   else self.db.get_setting("ocr_enabled", False))
            res = extract_pdf(str(path), enable_ocr=bool(ocr),
                              max_pages=self.s.max_pages,
                              max_chars=self.s.max_extracted_chars_per_doc)
        except Exception as exc:
            log.exception("extraction crash")
            self.db.update_document(doc_id, status="failed",
                                    status_detail=f"Extraction crashed: {exc}")
            return {"status": "failed", "error": str(exc)}
        if res.error and not res.pages:
            self.db.update_document(doc_id, status="failed", status_detail=res.error)
            return {"status": "failed", "error": res.error}
        pages = [{"page_number": p.page_number, "text": p.text,
                  "printed_label": p.printed_label,
                  "extraction_method": p.extraction_method,
                  "tables": p.tables} for p in res.pages]
        self.db.ensure_page_table_columns()
        self.db.store_pages(doc_id, pages)
        chunks = chunk_document(pages, min_chars=self.s.chunk_min_chars,
                                target_chars=self.s.chunk_target_chars,
                                max_chars=self.s.chunk_max_chars,
                                overlap_chars=self.s.chunk_overlap_chars)
        for c in chunks:
            c.setdefault("kind", "text")
        self.engine.index_document(doc_id, chunks)
        empty_pages = [p for p in res.pages if p.needs_ocr]
        status = "needs_ocr" if empty_pages and len(empty_pages) > len(res.pages) / 2 \
            else "ready"
        detail = res.error or ""
        if empty_pages:
            detail += (f" {len(empty_pages)} page(s) have no extractable text "
                       "(scanned image pages?). ")
            detail += ("Tesseract OCR is not installed/enabled, so these pages "
                       "are not searchable." if not tesseract_available()
                       else "OCR was attempted but could not read these pages.")
        try:
            title = res.metadata.get("title") or ""
            if title and not row["title"]:
                self.db.update_document(doc_id, title=title[:200])
        except Exception:
            pass
        self.db.update_document(
            doc_id, status=status, status_detail=detail.strip()[:500],
            page_count=res.page_count, ocr_pages=len(res.ocr_pages),
            needs_ocr_pages=len(empty_pages))
        self._pdfium_docs.pop(doc_id, None)
        return {"status": status, "pages": res.page_count, "chunks": len(chunks),
                "needs_ocr_pages": [p.page_number for p in empty_pages],
                "tables": sum(len(p["tables"]) for p in pages),
                "detail": detail.strip(), "error": res.error if status == "failed" else None}

    def reindex_document(self, doc_id: int) -> dict:
        row = self.db.get_document(doc_id)
        if row is None:
            raise ValueError("document not found")
        pages_rows = self.db.query(
            "SELECT page_number, text, printed_label, extraction_method, "
            "tables_text FROM document_pages WHERE document_id = ? "
            "ORDER BY page_number", (doc_id,))
        if pages_rows:
            # stored tables are the index truth; re-extract only to refresh them
            pages = [dict(r) | {"tables": [r["tables_text"]]
                                 if r["tables_text"] else []} for r in pages_rows]
            try:
                res = extract_pdf(row["stored_path"], enable_ocr=False,
                                  max_pages=self.s.max_pages)
                fresh = [pr.tables for pr in res.pages]
                if len(fresh) == len(pages):
                    for p, tabs in zip(pages, fresh):
                        if tabs:
                            p["tables"] = tabs
            except Exception as exc:
                log.warning("table re-extraction failed, using stored tables: %s", exc)
        else:
            return self.process_document(doc_id)
        chunks = chunk_document(pages, min_chars=self.s.chunk_min_chars,
                                target_chars=self.s.chunk_target_chars,
                                max_chars=self.s.chunk_max_chars,
                                overlap_chars=self.s.chunk_overlap_chars)
        self.engine.index_document(doc_id, chunks)
        self.db.update_document(doc_id, status="ready",
                                status_detail="reindexed")
        return {"status": "ready", "chunks": len(chunks)}

    def reindex_all(self) -> dict:
        return self.engine.reindex_all()

    def delete_document(self, doc_id: int) -> bool:
        row = self.db.get_document(doc_id)
        if not row:
            return False
        try:
            Path(row["stored_path"]).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove stored file: %s", exc)
        # invalidate render caches + derived index entries
        for f in (self.s.renders_dir / str(doc_id)).glob("*.png"):
            f.unlink(missing_ok=True)
        self._pdfium_docs.pop(doc_id, None)
        self.db.delete_document(doc_id)
        return True

    # --------------------------------------------------------------- search
    def search(self, query: str, *, top_k: int | None = None,
               document_ids: list[int] | None = None,
               collection_id: int | None = None,
               threshold: float | None = None) -> list[dict]:
        hits = self.engine.search(query, top_k=top_k, document_ids=document_ids,
                                   collection_id=collection_id, threshold=threshold)
        out = []
        for h in hits:
            d = h.to_dict()
            d["matched_page"] = h.matched_pages[0] if h.matched_pages else h.page_start
            d["excerpt"] = build_excerpt(h.text, query)
            out.append(d)
        return out

    def page_text(self, doc_id: int, page: int) -> dict | None:
        row = self.db.get_page(doc_id, page)
        if row is None:
            return None
        doc = self.db.get_document(doc_id)
        tables = row["tables_text"] if "tables_text" in row.keys() else ""
        return {"document_id": doc_id,
                "document_title": doc["title"] if doc else "",
                "page_number": page, "printed_label": row["printed_label"],
                "text": row["text"], "tables_text": tables,
                "full_text": (row["text"] + "\n\n" + tables).strip(),
                "extraction_method": row["extraction_method"]}

    def document_values(self, doc_id: int, *, page: int | None = None) -> list[dict]:
        """Every parsable engineering value on a page (Value Verification view)."""
        if page is not None:
            row = self.db.get_page(doc_id, page)
            texts = [(page, row["text"])] if row else []
        else:
            texts = [(r["page_number"], r["text"]) for r in self.db.query(
                "SELECT page_number, text FROM document_pages WHERE document_id = ? "
                "ORDER BY page_number", (doc_id,))]
        out = []
        for pno, t in texts:
            for v in extract_values(t):
                out.append({"page": pno, "raw": v.raw, "value": str(v.value),
                            "unit": v.unit_label, "category": v.category_label,
                            "qualifier": v.qualifier, "tolerance_pct": v.tolerance_pct,
                            "range": [str(x) for x in v.range_tuple] if v.range_tuple else None,
                            "temperature_c": v.temperature_c,
                            "phase_note": v.phase_note, "variant": v.variant,
                            "sentence": v.sentence[:220]})
        return out

    # ------------------------------------------------------------------ QA
    def ask(self, question: str, **kw) -> dict:
        res = self.rag.ask(question, **kw)
        return res.to_dict()

    def verify_standalone(self, text: str, *, document_ids: list[int] | None,
                          collection_id: int | None, page: int | None,
                          top_k: int = 6, rel_tolerance: float | None = None) -> dict:
        """Check a user-provided statement against the corpus."""
        tol = (self.s.default_rel_tolerance if rel_tolerance is None else
               max(0.0, min(rel_tolerance, self.s.max_rel_tolerance)))
        verifier = AnswerVerifier(self.db, rel_tolerance=tol if tol > 0 else 1e-9)
        if document_ids and page:
            rows = []
            for did in document_ids:
                pr = self.db.get_page(did, page)
                if pr:
                    rows.append({"ref_index": len(rows) + 1, "chunk_id": None,
                                 "document_id": did, "title": (
                                     self.db.get_document(did) or {}).get("title", ""),
                                 "page_start": page, "page_end": page,
                                 "text": pr["text"], "text_sha1": "", "kind": "text"})
            evidence = rows
            answer = text
        else:
            hits = self.engine.search(text, top_k=top_k,
                                      document_ids=document_ids,
                                      collection_id=collection_id)
            evidence = [{"ref_index": i, "chunk_id": h.chunk_id,
                         "document_id": h.document_id, "title": h.document_title,
                         "page_start": h.page_start, "page_end": h.page_end,
                         "matched_page": h.matched_pages[0] if h.matched_pages
                         else h.page_start,
                         "text": h.text, "text_sha1": h.text_sha1, "kind": h.kind,
                         "score": h.score}
                        for i, h in enumerate(hits, 1)]
            answer = text
        report = verifier.verify(answer, evidence, question="")
        return {"evidence": [{k: e[k] for k in e if k != "text"} for e in evidence],
                "report": report.to_dict()}

    # ------------------------------------------------------------ history
    def list_history(self, limit: int = 50) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM qa_history ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["citations"] = [dict(c) for c in self.db.query(
                "SELECT * FROM qa_citations WHERE qa_id = ?", (r["id"],))]
            d["num_checks"] = [dict(n) for n in self.db.query(
                "SELECT * FROM num_checks WHERE qa_id = ?", (r["id"],))]
            out.append(d)
        return out

    def get_history(self, qa_id: int) -> dict | None:
        r = self.db.query_one("SELECT * FROM qa_history WHERE id = ?", (qa_id,))
        if not r:
            return None
        d = dict(r)
        d["citations"] = [dict(c) for c in self.db.query(
            "SELECT * FROM qa_citations WHERE qa_id = ?", (qa_id,))]
        d["num_checks"] = [dict(n) for n in self.db.query(
            "SELECT * FROM num_checks WHERE qa_id = ?", (qa_id,))]
        return d

    def delete_history(self, qa_id: int | None = None) -> int:
        if qa_id is None:
            n = self.db.query_one("SELECT COUNT(*) AS n FROM qa_history")["n"]
            self.db.execute("DELETE FROM qa_history", ())
            self.db.execute("DELETE FROM qa_citations", ())
            self.db.execute("DELETE FROM num_checks", ())
            return int(n)
        self.db.execute("DELETE FROM qa_citations WHERE qa_id = ?", (qa_id,))
        self.db.execute("DELETE FROM num_checks WHERE qa_id = ?", (qa_id,))
        self.db.execute("DELETE FROM qa_history WHERE id = ?", (qa_id,))
        return 1

    # ---------------------------------------------------------- collections
    def list_documents(self, collection_id: int | None = None) -> list[dict]:
        docs = self.db.list_documents(collection_id)
        for d in docs:
            n = self.db.query_one(
                "SELECT COUNT(*) AS n FROM document_chunks WHERE document_id = ?",
                (d["id"],))
            d["chunk_count"] = int(n["n"]) if n else 0
        return docs

    # ------------------------------------------------------------ providers
    def provider_config(self) -> dict:
        """The provider that will actually be called (never inferred silently).

        An ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` in the environment is
        reported as a *suggestion* only: answering with a paid endpoint has to
        be enabled deliberately in this app, because the assistant must never
        spend money the user has not authorized.
        """
        cfg = dict(self.db.get_setting("active_provider") or {"kind": "offline"})
        env = _env_provider_fallback()
        if env:
            cfg["env_suggestion"] = {k: v for k, v in env.items() if k != "api_key"}
            cfg["env_key_present"] = True
        return cfg

    def save_provider(self, cfg: dict) -> dict:
        keep = {k: cfg.get(k) for k in
                ("kind", "label", "base_url", "api_key", "model", "temperature",
                 "max_tokens", "timeout_s", "retries")}
        keep = {k: v for k, v in keep.items() if v is not None and v != ""}
        self.db.set_setting("active_provider", keep)
        return keep

    # ---------------------------------------------------------------- misc
    def stats(self) -> dict:
        one = lambda sql: (self.db.query_one(sql) or {"n": 0})["n"]  # noqa: E731
        docs = self.db.query("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")
        have, total = self.db.vector_coverage(self.engine.embedder.model_tag)
        return {
            "documents": int(one("SELECT COUNT(*) AS n FROM documents")),
            "pages": int(one("SELECT COUNT(*) AS n FROM document_pages")),
            "chunks": int(one("SELECT COUNT(*) AS n FROM document_chunks")),
            "by_status": {r["status"]: r["n"] for r in docs},
            "vectors": {"modelled": have, "chunks": total,
                        "backend": self.engine.embedder.status.backend,
                        "model": self.engine.embedder.status.model_id},
            "fts5": self.db.fts5,
            "qa_rows": int(one("SELECT COUNT(*) AS n FROM qa_history")),
            "tesseract": tesseract_available(),
        }


def _env_provider_fallback() -> dict | None:
    import os
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"kind": "anthropic", "api_key": key,
                "model": os.environ.get("ANTHROPIC_MODEL", ""), "label": "Anthropic (env)"}
    return {"kind": "openai", "api_key": key,
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "model": os.environ.get("OPENAI_MODEL", ""), "label": "OpenAI (env)"}


def build_excerpt(text: str, query: str, max_chars: int = 460) -> str:
    from .retrieval.hybrid import build_search_excerpt
    return build_search_excerpt(text, query, max_chars)
