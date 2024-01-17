"""FastAPI application.

The API is the single functional core: the Streamlit frontend, the CLI and
the evaluation harness all call the same Service, so tests against these
endpoints exercise the real workflows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, load_settings
from ..logging_setup import get_logger, setup_logging
from ..rendering import render_page_png
from ..service import Service

log = get_logger("api")

_state: dict[str, Any] = {}


class AskPayload(BaseModel):
    """Ask endpoint body.  Defined at module level on purpose: with
    ``from __future__ import annotations`` FastAPI resolves the annotation
    against module globals, so a locally defined model would be misread as a
    query parameter and every request would 422."""

    question: str = Field(min_length=2, max_length=2000)
    document_ids: list[int] | None = None
    collection_id: int | None = None
    top_k: int | None = Field(None, ge=1, le=20)
    rel_tolerance: float | None = Field(None, ge=0.0, le=0.05)
    mode: str | None = None          # "llm" | "extractive" | None = auto


def create_app(settings: Settings | None = None,
               service: Service | None = None) -> FastAPI:
    s = settings or load_settings()
    setup_logging()
    svc = service or Service(s)
    _state["settings"] = s
    _state["service"] = svc

    app = FastAPI(title="Engineering RAG Assistant API", version=__version__,
                  docs_url="/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r".*",         # local tool: any localhost port
        allow_methods=["*"], allow_headers=["*"], allow_credentials=False)

    def service() -> Service:
        return _state["service"]

    # ------------------------------------------------------------- basic
    @app.get("/api/v1/health")
    def health() -> dict:
        svc = service()
        return {"status": "ok", "version": __version__, "data_dir": str(svc.s.data_dir),
                "fts5": svc.db.fts5, **svc.stats()}

    @app.get("/api/v1/stats")
    def stats() -> dict:
        return service().stats()

    # ------------------------------------------------------- collections
    @app.get("/api/v1/collections")
    def collections() -> list[dict]:
        return service().db.list_collections()

    @app.post("/api/v1/collections")
    def create_collection(payload: dict = Body(...)) -> dict:
        name = (payload.get("name") or "").strip()
        if not name or len(name) > 120:
            raise HTTPException(400, "Collection name required (≤120 chars).")
        cid = service().db.get_or_create_collection(
            name, (payload.get("description") or "")[:500])
        return {"id": cid, "name": name}

    @app.delete("/api/v1/collections/{cid}")
    def delete_collection(cid: int) -> dict:
        service().db.delete_collection(cid)
        return {"deleted": cid}

    # --------------------------------------------------------- documents
    @app.get("/api/v1/documents")
    def documents(collection_id: int | None = Query(None),
                  status: str | None = Query(None)) -> list[dict]:
        docs = service().list_documents(collection_id)
        if status:
            docs = [d for d in docs if d["status"] == status]
        return docs

    @app.post("/api/v1/documents")
    async def upload(file: UploadFile = File(...),
                     collection_id: int | None = Form(None),
                     title: str | None = Form(None)) -> JSONResponse:
        data = await file.read()
        limit = service().s.max_upload_bytes
        if len(data) > limit:
            return JSONResponse(status_code=413, content={
                "detail": f"File exceeds the configured limit of "
                          f"{limit // (1024 * 1024)} MB."})
        out = service().upload_pdf(filename=file.filename or "upload.pdf",
                                   data=data, collection_id=collection_id,
                                   title=title)
        code = 200 if out.status == "ok" else (409 if out.status == "duplicate"
                                               else 400)
        return JSONResponse(status_code=code, content={
            "document_id": out.document_id, "status": out.status,
            "message": out.message, "info": out.info})

    @app.get("/api/v1/documents/{doc_id}")
    def document(doc_id: int) -> dict:
        row = service().db.get_document(doc_id)
        if row is None:
            raise HTTPException(404, "document not found")
        d = dict(row)
        pages = service().db.query(
            "SELECT page_number, printed_label, char_count, extraction_method "
            "FROM document_pages WHERE document_id = ? ORDER BY page_number",
            (doc_id,))
        d["pages_meta"] = [dict(p) for p in pages]
        n = service().db.query_one(
            "SELECT COUNT(*) AS n FROM document_chunks WHERE document_id = ?",
            (doc_id,))
        d["chunk_count"] = int(n["n"]) if n else 0
        return d

    @app.post("/api/v1/documents/{doc_id}/reindex")
    def reindex(doc_id: int) -> dict:
        try:
            return service().reindex_document(doc_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @app.delete("/api/v1/documents/{doc_id}")
    def delete(doc_id: int) -> dict:
        if not service().delete_document(doc_id):
            raise HTTPException(404, "document not found")
        return {"deleted": doc_id}

    @app.get("/api/v1/documents/{doc_id}/file")
    def file_for(doc_id: int):
        row = service().db.get_document(doc_id)
        if row is None:
            raise HTTPException(404, "document not found")
        path = Path(row["stored_path"])
        if not path.exists():
            raise HTTPException(410, "stored file is missing on the server")
        return FileResponse(str(path), media_type="application/pdf",
                            filename=path.name, content_disposition_type="inline")

    @app.get("/api/v1/documents/{doc_id}/pages/{page}")
    def page_text(doc_id: int, page: int) -> dict:
        out = service().page_text(doc_id, page)
        if out is None:
            raise HTTPException(404, "page not found")
        return out

    @app.get("/api/v1/documents/{doc_id}/pages/{page}/render")
    def page_render(doc_id: int, page: int,
                    terms: str = Query("", description="comma-separated highlight terms"),
                    scale: float = Query(1.6, ge=0.5, le=3.0)) -> Response:
        row = service().db.get_document(doc_id)
        if row is None:
            raise HTTPException(404, "document not found")
        path = Path(row["stored_path"])
        png = render_page_png(path, doc_id, page,
                              renders_dir=service().s.renders_dir,
                              scale=scale,
                              max_pixels=service().s.max_render_pixels,
                              highlight_terms=[t for t in terms.split(",") if t][:40])
        if png is None:
            raise HTTPException(404, f"page {page} out of range")
        return Response(content=png, media_type="image/png")

    @app.get("/api/v1/documents/{doc_id}/values")
    def doc_values(doc_id: int, page: int | None = Query(None)) -> list[dict]:
        return service().document_values(doc_id, page=page)

    # ------------------------------------------------------------- search
    @app.post("/api/v1/search")
    def search(payload: dict = Body(...)) -> dict:
        q = (payload.get("query") or "").strip()
        if not q:
            raise HTTPException(400, "query required")
        hits = service().search(
            q, top_k=payload.get("top_k"),
            document_ids=payload.get("document_ids"),
            collection_id=payload.get("collection_id"),
            threshold=payload.get("threshold"))
        return {"query": q, "results": hits, "score_kind":
                "composite relevance (lexical BM25 + n-gram semantic + boosts); "
                "not a calibrated probability"}

    # -------------------------------------------------------------- ask
    @app.post("/api/v1/ask")
    def ask(payload: AskPayload) -> dict:
        if len(payload.question.strip()) < 2:
            raise HTTPException(400, "A non-empty question is required.")
        try:
            return service().ask(
                payload.question.strip(), document_ids=payload.document_ids,
                collection_id=payload.collection_id, top_k=payload.top_k,
                rel_tolerance=payload.rel_tolerance, mode=payload.mode)
        except Exception as exc:
            log.exception("ask failed")
            raise HTTPException(500, f"Answer pipeline failed: {exc}")

    @app.post("/api/v1/verify")
    def verify(payload: dict = Body(...)) -> dict:
        text = (payload.get("text") or "").strip()
        if len(text) < 3:
            raise HTTPException(400, "text to verify required")
        return service().verify_standalone(
            text, document_ids=payload.get("document_ids"),
            collection_id=payload.get("collection_id"), page=payload.get("page"),
            top_k=payload.get("top_k", 6),
            rel_tolerance=payload.get("rel_tolerance"))

    # ------------------------------------------------------------ history
    @app.get("/api/v1/history")
    def history(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        return service().list_history(limit)

    @app.get("/api/v1/history/{qa_id}")
    def history_one(qa_id: int) -> dict:
        d = service().get_history(qa_id)
        if d is None:
            raise HTTPException(404, "not found")
        return d

    @app.delete("/api/v1/history")
    def history_clear(qa_id: int | None = Query(None)) -> dict:
        return {"deleted": service().delete_history(qa_id)}

    # ---------------------------------------------------------- providers
    @app.get("/api/v1/providers")
    def providers() -> dict:
        from ..llm.providers import PROVIDER_KINDS
        cfg = service().provider_config()
        safe = {k: (v if k != "api_key" else
                   ("set" if v else "")) for k, v in cfg.items()}
        return {"active": safe, "kinds": PROVIDER_KINDS}

    @app.post("/api/v1/providers")
    def set_provider(payload: dict = Body(...)) -> dict:
        kind = payload.get("kind", "offline")
        if kind not in _KINDS:
            raise HTTPException(400, f"unknown provider kind {kind!r}")
        cfg = service().save_provider(payload)
        return {"saved": {k: (v if k != "api_key" else
                              ("set" if v else "")) for k, v in cfg.items()}}

    @app.post("/api/v1/providers/test")
    def test_provider(payload: dict = Body(...)) -> dict:
        from ..llm.providers import build_provider
        cfg = payload or service().provider_config()
        if not cfg.get("api_key"):
            saved = service().provider_config()
            if saved.get("kind") == cfg.get("kind") and saved.get("api_key"):
                cfg["api_key"] = saved["api_key"]
        try:
            provider = build_provider(cfg)
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}
        st = provider.status()
        return {"ok": st.ok, "detail": st.detail, "models": st.models[:80],
                "latency_ms": st.latency_ms, "kind": st.kind}

    @app.post("/api/v1/providers/chat")
    def provider_chat(payload: dict = Body(...)) -> dict:
        from ..llm.providers import ChatMessage, build_provider
        cfg = dict(payload.get("config") or service().provider_config())
        if not cfg.get("api_key"):
            saved = service().provider_config()
            if saved.get("api_key"):
                cfg["api_key"] = saved["api_key"]
        try:
            provider = build_provider(cfg)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        msgs = [ChatMessage(m["role"], m["content"]) for m in payload.get("messages", [])]
        resp = provider.chat(msgs)
        return {"ok": resp.ok, "text": resp.text[:4000], "error": resp.error,
                "model": resp.model, "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "latency_ms": resp.latency_ms}

    # ----------------------------------------------------------- settings
    @app.get("/api/v1/settings")
    def get_settings() -> dict:
        from ..config import settings_as_dict
        d = settings_as_dict(service().s)
        d["ocr_enabled"] = bool(service().db.get_setting("ocr_enabled", False))
        d["budget_warn_usd"] = service().db.get_setting(
            "budget_warn_usd", service().s.budget_warn_usd)
        return d

    @app.post("/api/v1/settings")
    def patch_settings(payload: dict = Body(...)) -> dict:
        if "ocr_enabled" in payload:
            service().db.set_setting("ocr_enabled", bool(payload["ocr_enabled"]))
        if "budget_warn_usd" in payload:
            service().db.set_setting("budget_warn_usd",
                                     float(payload["budget_warn_usd"]))
        return get_settings()

    @app.post("/api/v1/reindex")
    def reindex_all() -> dict:
        return service().reindex_all()

    # ------------------------------------------------------------ demo
    @app.post("/api/v1/demo-corpus")
    def demo_corpus(collection_id: int | None = Body(None, embed=True)) -> dict:
        from ..demo_corpus import generate_all
        tmp = service().s.data_dir / "demo_generated"
        docs = generate_all(tmp)
        created = []
        for d in docs:
            out = service().upload_pdf(filename=d.path.name,
                                       data=d.path.read_bytes(),
                                       collection_id=collection_id,
                                       title=d.title, is_synthetic=True)
            created.append({"document_id": out.document_id, "status": out.status,
                            "info": out.info, "title": d.title})
        return {"created": created}

    return app


_KINDS = {"offline", "ollama", "openai", "anthropic", "openrouter", "custom"}


def run_server(settings: Settings | None = None) -> None:
    import uvicorn
    app = create_app(settings)
    s = settings or load_settings()
    uvicorn.run(app, host=s.host, port=s.api_port, log_level="info")


def main() -> None:
    setup_logging()
    run_server()


if __name__ == "__main__":
    main()
