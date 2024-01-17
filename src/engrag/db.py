"""SQLite persistence layer.

Design principles
-----------------
* ``document_chunks`` is the *primary* retrieval record: text, page numbers
  and hashes are stored permanently.  FTS5 and vector rows are *derived
  indexes* that can be rebuilt from chunks at any time (``ensure_fts``,
  ``rebuild_indexes``), so citation targets can never be lost by index
  maintenance.
* Physical page numbers are 1-based and immutable; printed page labels are
  stored separately on ``document_pages``.
* Every chunk stores a ``text_sha1`` hash so citations can be validated for
  drift after any reindex.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    collection_id INTEGER,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    stored_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending|processing|ready|failed|needs_ocr
    status_detail TEXT NOT NULL DEFAULT '',
    ocr_pages INTEGER NOT NULL DEFAULT 0,
    needs_ocr_pages INTEGER NOT NULL DEFAULT 0,
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    is_public_domain INTEGER NOT NULL DEFAULT 0,
    version TEXT NOT NULL DEFAULT 'v1',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS document_pages (
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,           -- 1-based physical page
    printed_label TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    char_count INTEGER NOT NULL DEFAULT 0,
    extraction_method TEXT NOT NULL DEFAULT 'pypdf',  -- pypdf|pdfplumber|ocr|none
    tables_text TEXT NOT NULL DEFAULT '',      -- pipe-rendered tables on this page
    PRIMARY KEY (document_id, page_number),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'text',       -- text|table
    text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    text_sha1 TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model),
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS qa_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    collection_id INTEGER,
    document_ids TEXT NOT NULL DEFAULT '[]',
    abstained INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    est_cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id INTEGER NOT NULL,
    chunk_id INTEGER,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    printed_label TEXT NOT NULL DEFAULT '',
    quote TEXT NOT NULL DEFAULT '',
    quote_sha1 TEXT NOT NULL DEFAULT '',
    source_hash TEXT NOT NULL DEFAULT '',
    integrity TEXT NOT NULL DEFAULT 'unchecked', -- ok|stale|missing_quote|unchecked
    support TEXT NOT NULL DEFAULT 'unknown',     -- supported|weak|contradicted|unknown
    FOREIGN KEY (qa_id) REFERENCES qa_history(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS num_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qa_id INTEGER,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 0,
    claim TEXT NOT NULL,
    status TEXT NOT NULL,                    -- verified|unit_verified|mismatch|insufficient|ambiguous
    answer_value TEXT NOT NULL DEFAULT '',
    source_value TEXT NOT NULL DEFAULT '',
    answer_unit TEXT NOT NULL DEFAULT '',
    source_unit TEXT NOT NULL DEFAULT '',
    explanation TEXT NOT NULL DEFAULT '',
    context_label TEXT NOT NULL DEFAULT '',
    tolerance REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# FTS5 is preferred; fall back to a plain LIKE search table on exotic builds.
_FTS_SQLITE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='',            -- contentless: text lives in document_chunks
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS fts_map (
    fts_rowid INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL UNIQUE
);
"""


def _fts_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._engrag_fts_probe USING fts5(a)")
        conn.execute("DROP TABLE temp._engrag_fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


class Database:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 8000")
        self.fts5 = _fts_available(self._conn)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            if self.fts5:
                self._conn.executescript(_FTS_SQLITE)
            else:  # degraded fallback: plain tokens table
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS fts_map "
                    "(fts_rowid INTEGER PRIMARY KEY, chunk_id INTEGER NOT NULL UNIQUE)"
                )
            self._conn.commit()

    @classmethod
    def from_settings(cls, s: Settings) -> "Database":
        s.ensure_dirs()
        return cls(s.db_path)

    # ------------------------------------------------------------------ util
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur.lastrowid

    def executemany(self, sql: str, seq: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)
            self._conn.commit()

    # ----------------------------------------------------------- collections
    def get_or_create_collection(self, name: str, description: str = "",
                                 is_demo: bool = False) -> int:
        row = self.query_one("SELECT id FROM collections WHERE name = ?", (name,))
        if row:
            return int(row["id"])
        return self.execute(
            "INSERT INTO collections (name, description, is_demo, created_at) "
            "VALUES (?,?,?,?)", (name, description, int(is_demo), time.time()))

    def list_collections(self) -> list[dict]:
        rows = self.query(
            """SELECT c.*, (SELECT COUNT(*) FROM documents d
                            WHERE d.collection_id = c.id) AS doc_count
               FROM collections c ORDER BY c.name""")
        return [dict(r) for r in rows]

    def delete_collection(self, cid: int) -> None:
        for r in self.query("SELECT id FROM documents WHERE collection_id = ?", (cid,)):
            self.delete_document(int(r["id"]))
        self.execute("DELETE FROM collections WHERE id = ?", (cid,))

    # ------------------------------------------------------------- documents
    def insert_document(self, *, filename: str, title: str, sha256: str,
                        size_bytes: int, stored_path: str,
                        collection_id: int | None, is_synthetic: bool,
                        is_public_domain: bool) -> int:
        now = time.time()
        return self.execute(
            """INSERT INTO documents (filename, title, collection_id, sha256,
                   size_bytes, stored_path, status, is_synthetic,
                   is_public_domain, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (filename, title, collection_id, sha256, size_bytes, stored_path,
             "pending", int(is_synthetic), int(is_public_domain), now, now))

    def find_duplicate(self, sha256: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM documents WHERE sha256 = ?", (sha256,))

    def update_document(self, doc_id: int, **fields: Any) -> None:
        fields = {k: v for k, v in fields.items()
                  if k in {"title", "collection_id", "status", "status_detail",
                           "page_count", "ocr_pages", "needs_ocr_pages",
                           "is_synthetic", "is_public_domain", "version"}}
        if not fields:
            return
        fields["updated_at"] = time.time()
        keys = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE documents SET {keys} WHERE id = ?",
                     (*fields.values(), doc_id))

    def get_document(self, doc_id: int) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))

    def list_documents(self, collection_id: int | None = None,
                       status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM documents"
        conds, params = [], []
        if collection_id is not None:
            conds.append("collection_id = ?")
            params.append(collection_id)
        if status:
            conds.append("status = ?")
            params.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY updated_at DESC"
        return [dict(r) for r in self.query(sql, params)]

    def delete_document(self, doc_id: int) -> None:
        """Hard delete with full index/derived-data invalidation."""
        with self._lock:
            chunk_ids = [int(r["chunk_id"]) for r in self.query(
                "SELECT chunk_id FROM fts_map m JOIN document_chunks c "
                "ON c.id = m.chunk_id WHERE c.document_id = ?", (doc_id,))]
            for cid in chunk_ids:
                if self.fts5:
                    self._conn.execute("INSERT INTO chunks_fts(chunks_fts, rowid) "
                                       "VALUES ('delete', ?)", (cid,))
            self._conn.execute("DELETE FROM fts_map WHERE chunk_id IN "
                               "(SELECT id FROM document_chunks WHERE document_id = ?)",
                               (doc_id,))
            self._conn.execute("DELETE FROM chunk_vectors WHERE chunk_id IN "
                               "(SELECT id FROM document_chunks WHERE document_id = ?)",
                               (doc_id,))
            self._conn.execute("DELETE FROM document_pages WHERE document_id = ?", (doc_id,))
            self._conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
            # QA rows keep a readable record but citations are marked stale.
            self._conn.execute(
                "UPDATE qa_citations SET integrity = 'stale' WHERE document_id = ?",
                (doc_id,))
            self._conn.execute("DELETE FROM num_checks WHERE document_id = ?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()

    # ------------------------------------------------------------- pages/chunks
    def store_pages(self, doc_id: int, pages: list[dict]) -> None:
        """Persist one row per physical page (1-based) with its printed label
        and the pipe-rendered tables found on it."""
        rows = [(doc_id, p["page_number"], p.get("printed_label", ""),
                 p["text"], len(p["text"]), p.get("extraction_method", "pypdf"),
                 "\n\n".join(t for t in (p.get("tables") or []) if t))
                for p in pages]
        self.executemany(
            """INSERT INTO document_pages (document_id, page_number, printed_label,
                   text, char_count, extraction_method, tables_text)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(document_id, page_number) DO UPDATE SET
                 text = excluded.text, char_count = excluded.char_count,
                 printed_label = excluded.printed_label,
                 extraction_method = excluded.extraction_method,
                 tables_text = excluded.tables_text""",
            rows)

    def store_chunks(self, doc_id: int, chunks: list[dict]) -> list[int]:
        """Replace all chunks of a document (used by reindex) and rebuild FTS."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "DELETE FROM chunk_vectors WHERE chunk_id IN "
                "(SELECT id FROM document_chunks WHERE document_id = ?)", (doc_id,))
            self._conn.execute(
                "DELETE FROM fts_map WHERE chunk_id IN "
                "(SELECT id FROM document_chunks WHERE document_id = ?)", (doc_id,))
            if self.fts5:
                old = [int(r["id"]) for r in self._conn.execute(
                    "SELECT id FROM document_chunks WHERE document_id = ?", (doc_id,))]
                for cid in old:
                    self._conn.execute(
                        "INSERT INTO chunks_fts(chunks_fts, rowid) VALUES ('delete', ?)",
                        (cid,))
            self._conn.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
            ids: list[int] = []
            for c in chunks:
                cur = self._conn.execute(
                    """INSERT INTO document_chunks (document_id, page_start,
                           page_end, kind, text, char_count, text_sha1, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (doc_id, c["page_start"], c["page_end"], c.get("kind", "text"),
                     c["text"], len(c["text"]), c["text_sha1"], now))
                cid = int(cur.lastrowid)
                ids.append(cid)
                self._conn.execute(
                    "INSERT INTO fts_map (fts_rowid, chunk_id) VALUES (?,?)",
                    (cid, cid))
                if self.fts5:
                    self._conn.execute(
                        "INSERT INTO chunks_fts (rowid, text) VALUES (?,?)",
                        (cid, c["text"]))
            self._conn.commit()
        return ids

    def get_chunk(self, chunk_id: int) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM document_chunks WHERE id = ?", (chunk_id,))

    def get_page(self, doc_id: int, page_number: int) -> sqlite3.Row | None:
        return self.query_one(
            "SELECT * FROM document_pages WHERE document_id = ? AND page_number = ?",
            (doc_id, page_number))

    def ensure_page_table_columns(self) -> None:
        """Add columns introduced after the first release (safe to re-run)."""
        have = {r["name"] for r in self.query(
            "PRAGMA table_info(document_pages)")}
        if "tables_text" not in have:
            with self._lock:
                self._conn.execute(
                    "ALTER TABLE document_pages ADD COLUMN tables_text TEXT "
                    "NOT NULL DEFAULT ''")
                self._conn.commit()

    def ensure_fts(self) -> None:
        """Rebuild the FTS index from stored chunks (cheap consistency guard)."""
        if not self.fts5:
            return
        with self._lock:
            indexed = {int(r["chunk_id"]) for r in self._conn.execute(
                "SELECT chunk_id FROM fts_map")}
            for r in self._conn.execute(
                    "SELECT id, text FROM document_chunks"):
                if int(r["id"]) in indexed:
                    continue
                self._conn.execute(
                    "INSERT INTO chunks_fts (rowid, text) VALUES (?,?)",
                    (int(r["id"]), r["text"]))
                self._conn.execute(
                    "INSERT OR REPLACE INTO fts_map (fts_rowid, chunk_id) VALUES (?,?)",
                    (int(r["id"]), int(r["id"])))
            self._conn.commit()

    # ---------------------------------------------------------------- vectors
    def replace_vectors(self, model: str, items: list[tuple[int, "Any"]]) -> None:
        """Store numpy float32 vectors for chunks under a model tag."""
        import numpy as np
        with self._lock:
            for chunk_id, vec in items:
                v = np.asarray(vec, dtype=np.float32)
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunk_vectors (chunk_id, model, dim, vector) "
                    "VALUES (?,?,?,?)", (chunk_id, model, int(v.shape[0]), v.tobytes()))
            self._conn.commit()

    def all_vectors(self, model: str) -> tuple[list[int], "Any"]:
        import numpy as np
        rows = self.query(
            """SELECT v.chunk_id AS chunk_id, v.vector AS vector, v.dim AS dim
               FROM chunk_vectors v
               JOIN document_chunks c ON c.id = v.chunk_id
               WHERE v.model = ? ORDER BY v.chunk_id""", (model,))
        if not rows:
            return [], None
        ids = [int(r["chunk_id"]) for r in rows]
        mat = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        return ids, mat

    def vector_coverage(self, model: str) -> tuple[int, int]:
        total = self.query_one("SELECT COUNT(*) AS n FROM document_chunks")
        have = self.query_one(
            "SELECT COUNT(DISTINCT chunk_id) AS n FROM chunk_vectors WHERE model = ?",
            (model,))
        return int(have["n"] if have else 0), int(total["n"] if total else 0)

    def drop_vectors(self, model: str | None = None) -> None:
        if model:
            self.execute("DELETE FROM chunk_vectors WHERE model = ?", (model,))
        else:
            self.execute("DELETE FROM chunk_vectors", ())

    # ---------------------------------------------------------------- qa store
    def save_qa(self, *, question: str, mode: str, answer: str, model: str,
                collection_id: int | None, document_ids: list[int],
                abstained: bool, latency_ms: int, prompt_tokens: int,
                completion_tokens: int, est_cost_usd: float,
                citations: list[dict], num_checks: list[dict]) -> int:
        now = time.time()
        qa_id = self.execute(
            """INSERT INTO qa_history (question, mode, answer, model, collection_id,
                   document_ids, abstained, latency_ms, prompt_tokens,
                   completion_tokens, est_cost_usd, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (question, mode, answer, model, collection_id,
             json.dumps(document_ids), int(abstained), latency_ms, prompt_tokens,
             completion_tokens, est_cost_usd, now))
        for c in citations:
            self.execute(
                """INSERT INTO qa_citations (qa_id, chunk_id, document_id,
                       page_number, printed_label, quote, quote_sha1, source_hash,
                       integrity, support)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (qa_id, c.get("chunk_id"), c["document_id"], c["page_number"],
                 c.get("printed_label", ""), c.get("quote", ""),
                 c.get("quote_sha1", ""), c.get("source_hash", ""),
                 c.get("integrity", "unchecked"), c.get("support", "unknown")))
        for n in num_checks:
            self.execute(
                """INSERT INTO num_checks (qa_id, document_id, page_number, claim,
                       status, answer_value, source_value, answer_unit, source_unit,
                       explanation, context_label, tolerance, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (qa_id, n["document_id"], n.get("page_number", 0), n["claim"],
                 n["status"], str(n.get("answer_value", "")), str(n.get("source_value", "")),
                 n.get("answer_unit", ""), n.get("source_unit", ""),
                 n.get("explanation", ""), n.get("context_label", ""),
                 float(n.get("tolerance", 0.0)), now))
        return qa_id

    def trim_history(self, max_rows: int) -> None:
        self.execute(
            "DELETE FROM qa_history WHERE id NOT IN "
            "(SELECT id FROM qa_history ORDER BY id DESC LIMIT ?)", (max_rows,))

    # ---------------------------------------------------------------- settings
    def set_setting(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO settings (key, value) VALUES (?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (key, json.dumps(value)))

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]
