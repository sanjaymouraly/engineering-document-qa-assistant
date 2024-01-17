# Architecture

Diagrams (SVG, rendered from the descriptions below):
[component map](images/architecture.svg) · [question flow](images/request-flow.svg)

## Design rule

One service layer, many entry points. `Service` (`src/engrag/service.py`) is the only place
that touches the database and the index, and `RagService` (`src/engrag/rag.py`) is the only
place that orchestrates retrieve → gate → answer → verify. Streamlit, FastAPI, the CLI and the
evaluation harness all call those, so a measurement made by `scripts/run_eval.py` describes
exactly what a user experiences.

```
Streamlit (8 views)  ─┐
FastAPI /api/v1/*     ─┼─▶  Service ─▶ Ingestion ─▶ SQLite ─▶ Retrieval ─▶ RagService
engrag CLI            ─┘        │                                                   │
scripts/run_eval.py  ───────────┴──────────────────────────────────────────────────┴─▶ Verification
```

## Modules

| Module | Responsibility | Deliberate choices |
|---|---|---|
| `ingestion/extract.py` | pypdf for text + page labels, pdfplumber for tables (and as text fallback), scan detection, optional tesseract | physical pages are always 1-based; printed labels are captured separately and never used for navigation; tables are rendered as `cell \| cell` rows under a `TABLE (page N):` header **and** stored per page (`document_pages.tables_text`) so a quoted table row is findable on its page |
| `ingestion/chunking.py` | sentence-aware chunking with overlap | table rows are never split; a chunk that is a table stays one chunk so a row keeps its context |
| `db.py` | SQLite schema and queries, WAL, single `RLock` | FTS5 is *contentless* (`chunks_fts` + `fts_map`) so text is stored once; chunk `text_sha1` makes citation integrity checkable; `qa_citations.integrity` is set to `stale` on document delete instead of hiding the fact |
| `retrieval/keyword.py` | tokenizer + BM25 over FTS5, `LIKE` fallback when FTS5 is missing | only `ready`/`needs_ocr`/`failed` documents are searchable, and the status is echoed in results so partial indexes are visible |
| `retrieval/embeddings.py` | `sentence-transformers` when installed, otherwise a deterministic hashing n-gram embedder | vectors are persisted in `chunk_vectors` with the model id, and reused across restarts; no download is required for the offline path |
| `retrieval/hybrid.py` | fusion + reranking, page locating, excerpt building | `score = 0.55·bm25 + 0.30·cosine + 0.15·coverage + boosts`, reported as `composite_relevance_not_probability`; a numeric-recall pass guarantees passages containing a queried value reach the candidate pool |
| `rag.py` | prompt/instruction assembly, evidence gate, extractive digest, LLM call, verification, persistence | offline answers are a *numbered digest of verbatim quotes* with `[D# p#:q#]` markers; the evidence gate can refuse before any model is called (saving money and preventing invented answers) |
| `verify/numbers.py` | value extraction and comparison | `Decimal` magnitudes, `Pint` dimensions, qualifier detection (`rated`/`maximum`/`typical`/`nominal`/`short-time`/…), tolerances, ranges, temperature, phase, `kW ≠ kVA`, variant model numbers, table row labels from the first cell of a row |
| `verify/claims.py` | sentence splitting, quote verification, citation integrity, conflicts, support judgement | quotations are verified by *existence in the retrieved corpus* (a quote is not an endorsement — cross-passage disagreement belongs to the conflict report); the answer's own prose goes through the full numeric pipeline |
| `verify/textmatch.py` | whitespace/hyphen/ellipsis-tolerant matching with spans | normalized indexes are memoized (verification was 20× slower before); keys are the text itself, never hashes, so a collision can never produce a wrong span |
| `llm/providers.py` | per-vendor adapters | each protocol is spoken natively; every client is created by `BaseProvider._client()` so tests can inject `httpx.MockTransport`; no model ids are invented |
| `llm/sanitizer.py` | prompt-injection masking of document text | flags + masks instruction-override, persona-hijack, secret-ask and role-spoof patterns before anything reaches a model |
| `rendering.py` | page rasterisation + citation highlight | pypdfium2 for pixels, pdfplumber word boxes for geometry, boxes multiplied by the render scale (points→pixels) and pixel-tested in `tests/test_rendering.py` |
| `api/server.py` | HTTP surface | thin; returns the same `verification` object the UI renders |
| `frontend/` | Streamlit UI | calls the service in-process (works inside sandboxed previews where a browser cannot reach the API) |

## Data model (SQLite)

Real columns (read with `PRAGMA table_info` against `data/engrag.db`):

```
documents(id, filename, title, collection_id, sha256, size_bytes, page_count,
          stored_path, status, status_detail, ocr_pages, needs_ocr_pages,
          is_synthetic, is_public_domain, version, created_at, updated_at)
collections(id, name, description, is_demo, created_at)
document_pages(document_id, page_number, printed_label, text, char_count,
               extraction_method, tables_text)   PRIMARY KEY (document_id, page_number)
document_chunks(id, document_id, page_start, page_end, kind, text, char_count,
                text_sha1, created_at)
chunks_fts(text)  contentless FTS5  +  fts_map(rowid → chunk_id)
chunk_vectors(chunk_id, model, dim, vector)     -- vector is a float32 BLOB
qa_history(id, question, mode, answer, model, collection_id, document_ids,
           abstained, latency_ms, prompt_tokens, completion_tokens,
           est_cost_usd, created_at)
qa_citations(id, qa_id, chunk_id, document_id, page_number, printed_label, quote,
             quote_sha1, source_hash, integrity, support)
num_checks(id, qa_id, document_id, page_number, claim, status, answer_value,
           source_value, answer_unit, source_unit, explanation, context_label,
           tolerance, created_at)
settings(key, value)                            -- provider config, OCR toggle, budget
```

`integrity` on a citation takes `ok`, `missing_quote`, `invalid_ref` or `stale`. The answer-level
verdict (`grounded` / `partially_grounded` / …) is deliberately **not** stored: the History view
shows per-citation `integrity`/`support` plus a numeric-status histogram, so what you read there
is exactly what the evidence rows say, never a summary that could drift from them.

Cascade behaviour (`db.delete_document`): the stored file, pages, chunks, FTS rows,
`fts_map` entries, vectors and render cache are removed; the numeric checks belonging to that
document are deleted; `qa_history` rows survive but every `qa_citations` row pointing into the
document is set to `integrity='stale'`. History is therefore still readable for audit while
saying out loud that the evidence behind it is gone.

## Verification semantics (the important part)

1. **Split.** The answer is split into claim units. A numbered evidence block starts a new unit;
   so does each navigation bullet. Splits never occur inside `“quotes”`. The generated heading of
   a digest block (`3. Some 11 kV handbook, page 2 — …`) is *not* a claim: numbers in a document's
   own title are excluded, otherwise the title "11 kV switchgear handbook" would be checked as if
   it asserted a voltage.
2. **Quotations.** For each `“…”` run: strip display ellipsis, split on `…`, whitespace-collapse,
   then require every ≥12-character segment to appear in the retrieved evidence pool. Found →
   `verified` (with the wording *"a quotation is not an endorsement"*). Not found → `insufficient`
   with *"the quotation may be fabricated or mis-cited"*.
3. **Prose claims.** Values in the assistant's own sentences go through the full pipeline:
   context alignment, qualifier scope, row labels, unit dimensions, tolerance.
4. **Citations.** For each `[D# p#:q#]` / `[n]` marker: the cited chunk is re-read, its
   `text_sha1` compared with the stored hash, the longest verbatim fragment of the sentence must
   exist in the chunk and on the page (`text + tables_text`). Support becomes `supported` only
   when both hold; a chunk-only match degrades to `weak`, and no match at all becomes
   `insufficient`.
5. **Conflicts.** `find_conflicts()` compares same-category, same-qualifier values across the
   cited passages and lists disagreements with document, page and value — but only when both
   values carry an overlapping asset tag. It never picks a winner, and it never reports two
   different machines as a disagreement.
6. **Verdict.** `grounded` (all sentences supported/advisory) · `partially_grounded` ·
   `abstained` (a `mismatch` withholds the answer) · `unverified`.

## Why the offline mode is not a toy

Extractive mode is a deliberate design, not a stub: the numbered digest quotes real passages with
resolvable citations, the evidence gate refuses weak retrieval, and the verification engine checks
every number that appears. An LLM adds generated prose on top of the *same* evidence and is then
held to the same verification — if it fails, the extractive answer replaces it and the substitution
is stated openly.

## Failure modes and how the design reacts

| Failure | Reaction |
|---|---|
| PDF has no extractable text | status `needs_ocr`, pages listed, search reports them as non-searchable |
| Duplicate file uploaded | `409` with the existing document id; nothing re-indexed |
| Document deleted after an answer was stored | citations flagged `stale`; the file's index entries removed |
| PDF contains "ignore previous instructions" | masked before retrieval/model, `suspicious` flag raised, warning shown |
| Model returns an uncited or contradicted answer | answer withheld, extractive fallback used, reason stored |
| Model gateway down / timeout | provider `ok=False`, extractive mode answers, `llm_error_shown` recorded |
| FTS5 missing in the Python build | automatic `LIKE` fallback, `stats.fts5=false` |
| Huge page renders | `max_render_pixels` downscales before drawing |
