"""Engineering Document Q&A Assistant — Streamlit frontend.

Run:  streamlit run src/engrag/frontend/app.py
The UI calls the in-process Service directly (no browser→localhost HTTP), so
it also works inside sandboxed preview frames.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from engrag.frontend.uikit import (  # noqa: E402
    WIDTH_KW, get_service, image_bytes_to_md, sanitize_name, style,
    support_badge)
from engrag.llm.providers import PROVIDER_KINDS  # noqa: E402

st.set_page_config(page_title="Engineering RAG Assistant",
                   page_icon="📘", layout="wide", initial_sidebar_state="expanded")


VIEWS = ["ask", "documents", "search", "citations", "values", "settings",
         "evaluation", "history"]


def badge(kind: str, label: str) -> str:
    return f'<span class="badge b-{kind}">{label}</span>'


# ============================================================ page: Ask
def page_ask(svc) -> None:
    st.header("Ask Engineering AI")
    st.caption("Grounded answers with verifiable page citations. Works fully "
               "offline — the LLM is optional.")
    docs = svc.list_documents()
    cols = svc.db.list_collections()
    with st.sidebar:
        st.subheader("Sources")
        coptions = ["All documents"] + [f"collection:{c['name']}" for c in cols]
        if docs:
            coptions += [f"doc:{d['id']} {d['title']}" for d in docs]
        sel = st.selectbox("Search scope", coptions, key="ask_scope")
        top_k = st.slider("Top-k evidence passages", 2, 12,
                          svc.s.default_top_k, key="ask_topk")
        tol = st.slider("Numeric tolerance (%) — 0 means exact match", 0.0, 5.0,
                        round(svc.s.default_rel_tolerance * 100, 3) or 0.0,
                        step=0.01, key="ask_tol")
        mode = st.radio("Answer mode", ["Auto", "Offline extractive", "LLM"],
                        horizontal=True, key="ask_mode")
    document_ids, collection_id = _resolve_scope(svc, sel)
    q = st.text_area("Your question", height=80,
                     placeholder="e.g. What is the rated current on the HV side "
                                 "of DEMO-TR-01 at 40 MVA?", key="ask_q")
    examples = [
        "What are the rated voltage and frequency of DEMO-TR-01?",
        "What minimum cross-section is needed for a 25 kA 3 s fault on a copper XLPE cable?",
        "Compare the rated apparent power of DEMO-TR-01 and DEMO-TR-02.",
        "What is the current-carrying capacity of a 240 mm² cable buried in three circuits at 30 °C with soil 2.5 K·m/W?",
        "What is the vibration alarm limit for motor DEMO-M-203?",
        "What wind-speed rating does the hoverboard datasheet give?",
    ]
    st.caption("Example questions (from the bundled synthetic demo corpus):")
    for row in (examples[i:i + 2] for i in range(0, len(examples), 2)):
        c1, c2 = st.columns(2)
        for cc, ex in zip((c1, c2), row):
            if cc.button(ex, key=f"x_{ex[:18]}", **WIDTH_KW):
                st.session_state.ask_q = ex
                st.rerun()
    go = st.button("Ask", type="primary", disabled=not q.strip(), key="ask_go")
    if not docs:
        st.info("No documents yet. Open **Document Library** and either upload "
                "PDFs or click **Load synthetic demo corpus** to try the "
                "application immediately.")
        return
    if go:
        with st.spinner("Retrieving evidence, verifying citations and values…"):
            res = svc.ask(st.session_state.get("ask_q", q),
                          document_ids=document_ids, collection_id=collection_id,
                          top_k=top_k,
                          rel_tolerance=(tol / 100.0 if tol > 0 else 0.0),
                          mode=None if mode == "Auto" else
                          ("extractive" if "extractive" in mode else "llm"))
        st.session_state["ask_last"] = res
    res = st.session_state.get("ask_last")
    if not res:
        return
    _render_ask_result(svc, res, question=st.session_state.get("ask_q", q))


def _resolve_scope(svc, sel: str):
    if sel.startswith("collection:"):
        name = sel.split(":", 1)[1]
        row = svc.db.query_one("SELECT id FROM collections WHERE name = ?", (name,))
        return None, int(row["id"]) if row else None
    if sel.startswith("doc:"):
        did = int(sel[4:].split(" ", 1)[0])
        return [did], None
    return None, None


def _render_ask_result(svc, res: dict, question: str = "") -> None:
    v = res.get("verification", {})
    m = res.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(badge("info" if not res["abstained"] else "warn",
                      res["mode"]), unsafe_allow_html=True)
    c2.markdown("support: " + support_badge(v.get("overall_support", "unknown")),
                unsafe_allow_html=True)
    c3.markdown(f"retrieval {m.get('retrieval_ms', 0)} ms · total "
                f"{m.get('total_ms', 0)} ms", unsafe_allow_html=False)
    if m.get("cost_usd"):
        c4.markdown(f"est. cost ≈ ${m['cost_usd']:.4f} (estimate)",
                    unsafe_allow_html=False)
    if res.get("warnings"):
        for w in res["warnings"]:
            st.warning(w)
    st.markdown("#### Answer")
    st.markdown(res["answer"])
    btn1, btn2 = st.columns([1, 4])
    btn1.download_button("Download answer+citations",
                         data=_answer_with_citations_md(res),
                         file_name=sanitize_name((question or "answer")[:30]) + ".md",
                         mime="text/markdown", **WIDTH_KW)
    if m.get("llm_error_shown"):
        st.caption("LLM was unavailable for this request; the answer above was "
                   "composed extractively from the evidence — that is a design "
                   "fallback, not a failure.")
    tabs = st.tabs(["📎 Citations & evidence", "🔢 Numeric verification",
                    "🧾 Per-sentence checks"])
    with tabs[0]:
        ev = res.get("evidence", [])
        st.markdown(f"**Evidence passages ({len(ev)})** — click *View page* to "
                    "open the exact source page with highlighting.")
        for e in ev:
            st.markdown('<div class="evbox">', unsafe_allow_html=True)
            head = (f"**[{e['ref_index']}] {e['title']}** — physical page "
                    f"{e['page']}" + (f" (printed {e['printed_label']})"
                                      if e.get("printed_label") else "") +
                    f" · {e['kind']} · score {e['score']}")
            st.markdown(head, unsafe_allow_html=False)
            st.text_area("quote", e["excerpt"][:600], height=110,
                         disabled=True, key=f"ev_{res.get('qa_id')}_{e['ref_index']}",
                         label_visibility="collapsed")
            cc1, cc2 = st.columns([1, 4])
            if cc1.button("View page", key=f"vp_{e['ref_index']}", **WIDTH_KW):
                st.session_state["viewer"] = {"doc": e["document_id"],
                                               "page": e["page"],
                                               "terms": _terms(e["excerpt"])}
                st.session_state["nav_pending"] = "citations"  # deep link (applied in main)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    with tabs[1]:
        _render_numeric(v)
    with tabs[2]:
        for s in v.get("sentences", []):
            st.markdown(f'{support_badge(s["support"])} ' +
                        s["text"][:240].replace("\n", " "),
                        unsafe_allow_html=True)
            if s.get("note"):
                st.caption("ℹ " + s["note"])
            for c in s.get("citations", []):
                if c["integrity"] != "ok":
                    st.caption(f'⚠ citation {c["marker"]}: {c["integrity"]} — '
                               f'{c["note"]}')
            if s.get("suggested_replacement"):
                st.info("Correction candidate: " + s["suggested_replacement"])


def _terms(text: str) -> str:
    import re
    toks = re.findall(r"\d[\d,.]*\s*(?:kV|kW|kVA|mm²|Hz|kA|%|A|V|W)", text)
    return ",".join(dict.fromkeys(t.strip() for t in toks))[:120]


def _answer_with_citations_md(res: dict) -> str:
    lines = [f"# Answer\n\n{res['answer']}\n\n## Citations\n"]
    for e in res.get("evidence", []):
        lines.append(f"[{e['ref_index']}] {e['title']}, physical page "
                     f"{e['page']}{f' (printed {e['printed_label']})' if e.get('printed_label') else ''}: "
                     f"{e['excerpt'][:300]}…\n")
    return "\n".join(lines)


def _render_numeric(v: dict) -> None:
    conflicts = v.get("conflicts") or []
    if conflicts:
        st.error(f"**{len(conflicts)} conflict(s) between cited sources** — "
                 "these are reported, never auto-resolved:")
        for cf in conflicts[:8]:
            for x in cf["values"]:
                st.markdown(f"- {cf['category']} ({cf['qualifier']}): "
                            f"“{x['raw']}” in {x['document_title']} p.{x['page']}")
    checks = v.get("numeric_checks", [])
    if not checks and not conflicts:
        st.success("No numeric mismatches. Every engineering value either "
                   "appears verbatim in its cited excerpt or was confirmed "
                   "after exact unit conversion.")
        return
    for n in checks:
        cls = {"verified": "ok", "unit_verified": "ok",
               "verified_within_tolerance": "warn", "mismatch": "bad",
               "ambiguous": "warn", "insufficient": "warn"}[n["status"]]
        st.markdown(f'<span class="badge b-{cls}">{n["status"]}</span> '
                    f'**{n["claim"]}** → source “{n.get("source_value") or "—"}”'
                    f' · {n["document_title"]} p.{n["page_number"]}',
                    unsafe_allow_html=True)
        with st.expander("explanation & context", expanded=n["status"] == "mismatch"):
            st.write(n["explanation"])
            a, b = st.columns(2)
            a.caption("answer context: " + str(n.get("answer_context", {})))
            b.caption("source context: " + str(n.get("source_context", {})))
            st.caption(f"tolerance used: {n['tolerance_pct']} %")


# ====================================================== page: documents
def page_documents(svc) -> None:
    st.header("Document Library")
    t1, t2, t3 = st.tabs(["📄 Documents", "🗂 Collections", "Upload / demo"])
    with t1:
        docs = svc.list_documents()
        if not docs:
            st.info("No documents. Use the **Upload / demo** tab.")
        for d in docs:
            c1, c2, c3, c4, c5 = st.columns([4, 1.1, 1.1, 1.1, 1.1])
            kind = {"ready": "ok", "needs_ocr": "warn", "failed": "bad"}.get(
                d["status"], "info")
            c1.markdown(f"**{d['title']}**  \n"
                        f'<span class="small">{d["filename"]} · {d["sha256"][:12]}… · '
                        f'{d["size_bytes"]/1024:.0f} KB · '
                        f'{"SYNTHETIC " if d["is_synthetic"] else ""}'
                        f'{"PD " if d["is_public_domain"] else ""}</span>',
                        unsafe_allow_html=True)
            c2.markdown(badge(kind, d["status"]), unsafe_allow_html=True)
            c3.markdown(f"{d['page_count']} p\n\n{d['chunk_count']} ch",
                        help="pages / chunks")
            if c4.button("View", key=f"dv{d['id']}"):
                st.session_state["viewer"] = {"doc": d["id"], "page": 1, "terms": ""}
                st.session_state["nav_pending"] = "citations"  # deep link (applied in main)
                st.rerun()
            if c5.button("Delete", key=f"dd{d['id']}"):
                svc.delete_document(d["id"])
                st.cache_resource.clear()
                st.rerun()
            if d["status"] in ("pending", "processing"):
                st.button("Reindex now", key=f"ri{d['id']}",
                          on_click=lambda i=d["id"]: (svc.reindex_document(i),
                                                      st.rerun()))
            if d["status_detail"]:
                st.caption("ℹ " + d["status_detail"][:220])
    with t2:
        with st.form("newcol"):
            name = st.text_input("New collection name", max_chars=80)
            desc = st.text_input("Description", max_chars=200)
            if st.form_submit_button("Create collection") and name.strip():
                svc.db.get_or_create_collection(name.strip(), desc.strip())
                st.rerun()
        for c in svc.db.list_collections():
            cc1, cc2, cc3 = st.columns([5, 1, 1])
            cc1.markdown(f"**{c['name']}** · {c['doc_count']} doc(s) · "
                         f"{'demo' if c['is_demo'] else 'user'}")
            if cc2.button("Search", key=f"cs{c['id']}"):
                st.session_state["nav_pending"] = "search"  # deep link (applied in main)
                st.session_state["search_scope"] = f"collection:{c['name']}"
                st.rerun()
            if cc3.button("Delete", key=f"cd{c['id']}"):
                svc.db.delete_collection(c["id"])
                st.rerun()
    with t3:
        st.markdown("**Upload engineering PDFs** (single file or several at once)")
        up = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True,
                              key="up_pdf")
        ccols = ["No collection"] + [f"{c['id']}:{c['name']}"
                                      for c in svc.db.list_collections()]
        coll = st.selectbox("Add to collection", ccols)
        cid = int(coll.split(":")[0]) if coll != "No collection" else None
        if up and st.button("Upload & index", type="primary"):
            bar = st.progress(0.0, "extracting, chunking, indexing…")
            for i, f in enumerate(up):
                out = svc.upload_pdf(filename=f.name, data=f.getvalue(),
                                     collection_id=cid)
                bar.progress((i + 1) / len(up), f"{f.name}: {out.message}")
                if out.status == "duplicate":
                    st.warning(f"**{f.name}**: duplicate detected — already "
                               f"stored as document #{out.document_id}. "
                               "Re-upload was ignored.")
                elif out.status == "ok":
                    st.success(f"**{f.name}** indexed "
                               f"(pages={ (out.info or {}).get('pages') },"
                               f" chunks={ (out.info or {}).get('chunks') }).")
                else:
                    st.error(f"**{f.name}**: {out.message}")
            st.rerun()
        st.divider()
        st.markdown("**Synthetic demo corpus** — five fictional engineering "
                    "manuals (transformers, switchgear, cables, motors) with "
                    "internally consistent data, planted contradictions and a "
                    "prompt-injection page for testing. *Never* real standards.")
        if st.button("Load synthetic demo corpus", type="primary"):
            from engrag.demo_corpus import generate_all
            tmp = svc.s.data_dir / "demo_generated"
            docs = generate_all(tmp)
            demo_col = svc.db.get_or_create_collection(
                "Synthetic demo manuals",
                "Invented demo data for testing only — not a real standard",
                is_demo=True)
            for d in docs:
                svc.upload_pdf(filename=d.path.name, data=d.path.read_bytes(),
                               collection_id=demo_col, title=d.title,
                               is_synthetic=True)
            st.success(f"{len(docs)} demo documents ingested.")
            st.rerun()
        ocr = st.checkbox("Enable OCR (Tesseract) for scanned pages",
                          value=bool(svc.db.get_setting("ocr_enabled", False)))
        if svc.db.get_setting("ocr_enabled", False) != ocr:
            svc.db.set_setting("ocr_enabled", ocr)
            from engrag.ingestion.extract import tesseract_available
            if ocr and not tesseract_available():
                st.warning("Tesseract binary not found on this system; pages "
                           "without a text layer will be flagged `needs_ocr` "
                           "instead of silently indexed as empty.")


# ====================================================== page: search
def page_search(svc) -> None:
    st.header("Document Search")
    docs = svc.list_documents()
    cols = svc.db.list_collections()
    opts = ["All documents"] + [f"collection:{c['name']}" for c in cols] \
        + [f"doc:{d['id']} {d['title']}" for d in docs]
    c1, c2, c3 = st.columns([4, 1.2, 1.6])
    q = c1.text_input("Full-text + semantic search",
                      placeholder="e.g. derating factor soil resistivity",
                      key="search_q")
    topk = c2.slider("Top-k", 3, 20, 8, key="search_k")
    sel_def = st.session_state.pop("search_scope", "All documents")
    sel = c3.selectbox("Scope", opts,
                       index=opts.index(sel_def) if sel_def in opts else 0)
    thr = st.slider("Minimum relevance (0–1, not a probability)", 0.0, 0.6,
                    svc.s.relevance_threshold, step=0.01, key="search_thr")
    if not q.strip():
        st.info("Type a query. Hybrid search = FTS5 BM25 keyword ranking + "
                f"semantic n-gram vectors ({svc.engine.embedder.status.backend}) "
                "+ deterministic reranking of numeric/table content.")
        return
    document_ids, collection_id = _resolve_scope(svc, sel)
    hits = svc.search(q, top_k=topk, document_ids=document_ids,
                      collection_id=collection_id, threshold=thr)
    st.caption(f"{len(hits)} result(s). Score breakdown is shown per hit; "
               "scores are composite relevance signals, not calibrated "
               "probabilities.")
    for h in hits:
        st.markdown(f"**{h['document_title']}** — page **{h['matched_page']}**"
                    + (f" (printed {h['printed_label']})" if h.get("printed_label")
                       else "") +
                    f" · {h['kind']} · score {h['score']} "
                    f"(lexical {h['lexical']} · semantic {h['semantic']} · "
                    f"coverage {h['coverage']})",
                    unsafe_allow_html=False)
        st.markdown(f'<div class="quote">{h["excerpt"][:420]}</div>',
                    unsafe_allow_html=True)
        if st.button("Open page", key=f"op{h['chunk_id']}"):
            st.session_state["viewer"] = {"doc": h["document_id"],
                                          "page": h["matched_page"],
                                          "terms": _terms(q)}
            st.session_state["nav_pending"] = "citations"  # deep link (applied in main)
            st.rerun()


# ====================================================== page: citation viewer
def page_viewer(svc) -> None:
    st.header("Citation Viewer")
    docs = svc.list_documents()
    if not docs:
        st.info("Upload documents first.")
        return
    # A citation link queues the document/page/terms to display.  The widgets
    # own their own state, and Streamlit *discards the default argument* (and
    # logs a warning) if a widget key is written to before the widget is
    # created — so a queued request is applied by dropping the stale key and
    # letting the widget default (`index=` / `value=`) take effect.
    v = st.session_state.pop("viewer", None) or {}
    opts = {d["id"]: d["title"] for d in docs}
    labels = list(opts.keys())
    queued_doc = v.get("doc") if v.get("doc") in labels else None
    if queued_doc is not None:
        st.session_state.pop("cv_doc", None)
        idx = labels.index(queued_doc)
    else:
        cur = st.session_state.get("cv_doc")
        idx = labels.index(cur) if cur in labels else 0
    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
    did = c1.selectbox("Document", labels, index=idx, format_func=lambda i: opts[i],
                       key="cv_doc")
    drow = svc.db.get_document(int(did))
    npp = max(1, int(drow["page_count"] or 1))
    # The cited page opens the viewer; otherwise the page the user was last on
    # is kept.  Either way the number must stay inside the selected document.
    prev_page = st.session_state.get("cv_page")
    want = v.get("page") if queued_doc is not None else prev_page
    page_default = min(max(1, int(want or 1)), npp)
    if prev_page is None or int(prev_page) != page_default:
        st.session_state.pop("cv_page", None)
    page = c2.number_input("Physical page", 1, npp, page_default, key="cv_page")
    if v.get("terms") is not None:
        st.session_state.pop("cv_terms", None)
    terms = c3.text_input("Highlight terms",
                          value=("" if v.get("terms") is None else v["terms"]),
                          help="comma separated; exact words from the citation",
                          key="cv_terms")
    if c4.button("Render", key="vr", **WIDTH_KW):
        pass  # button press triggers the rerun that re-renders the page
    st.markdown(
        f"**{drow['title']}** · status `{drow['status']}` · sha256 "
        f"`{drow['sha256'][:16]}…` · {drow['page_count']} pages · version {drow['version']}"
        + (" · ⚠ synthetic demo data" if drow["is_synthetic"] else ""),
        unsafe_allow_html=False)
    png = None
    try:
        from engrag.rendering import render_page_png
        png = render_page_png(Path(drow["stored_path"]), int(did), int(page),
                              renders_dir=svc.s.renders_dir,
                              scale=svc.s.pdf_render_scale,
                              max_pixels=svc.s.max_render_pixels,
                              highlight_terms=[t.strip() for t in terms.split(",")
                                               if t.strip()][:40])
    except Exception as exc:
        st.error(f"Page render failed: {exc}")
    left, right = st.columns([3, 2])
    with left:
        if png:
            image_bytes_to_md(png, f"Physical page {page} "
                                  f"(highlights = cited terms)")
        cc1, cc2 = st.columns(2)
        if cc1.button("◀ Previous page", key="pv", disabled=page <= 1):
            st.session_state["viewer"] = {"doc": int(did), "page": int(page) - 1,
                                          "terms": terms}
            st.rerun()
        if cc2.button("Next page ▶", key="nx", disabled=page >= npp):
            st.session_state["viewer"] = {"doc": int(did), "page": int(page) + 1,
                                          "terms": terms}
            st.rerun()
    with right:
        pdata = svc.page_text(int(did), int(page))
        if pdata:
            st.markdown(f"**Extracted text** · method `{pdata['extraction_method']}`"
                        + (f" · printed label `{pdata['printed_label']}`"
                           if pdata["printed_label"] else ""))
            st.text_area("page text", pdata["text"][:4000], height=420,
                         disabled=True, label_visibility="collapsed")
        tables = svc.document_values(int(did), page=int(page))
        if tables:
            st.markdown("**Engineering values detected on this page**")
            import pandas as pd
            st.dataframe(pd.DataFrame([{k: v for k, v in t.items()
                                        if k != "sentence"} for t in tables]),
                         hide_index=True, **WIDTH_KW)
        svc_link = Path(drow["stored_path"])
        if svc_link.exists():
            st.download_button("Download original PDF", svc_link.read_bytes(),
                               file_name=sanitize_name(drow["filename"]),
                               mime="application/pdf")


# ====================================================== page: values
def page_values(svc) -> None:
    st.header("Value Verification")
    st.caption("Two ways to use the Engineering Value Verification Engine: "
               "(1) paste a statement and compare it against the corpus, "
               "(2) inspect every value the parser sees on a document page.")
    t1, t2 = st.tabs(["Check a statement", "Page value inspector"])
    with t1:
        stmt = st.text_area("Statement to verify", height=90, key="vs_stmt",
                            placeholder="The rated voltage of DEMO-TR-01 is 11 kV, "
                                        "which equals 11,000 V.")
        c1, c2 = st.columns([2, 1])
        tol = c1.slider("Tolerance (%)", 0.0, 5.0, 0.0, step=0.01, key="vs_tol")
        if c2.button("Verify", type="primary", disabled=not stmt.strip()):
            res = svc.verify_standalone(stmt, document_ids=None,
                                        collection_id=None, page=None,
                                        rel_tolerance=(tol / 100 if tol else 0.0))
            st.session_state["vs_last"] = res
        res = st.session_state.get("vs_last")
        if res:
            _render_numeric(res["report"])
            with st.expander("evidence used"):
                for e in res["evidence"]:
                    st.write(f"[{e['ref_index']}] {e['title']} p."
                             f"{e.get('matched_page') or e['page_start']}"
                             f" · score {e.get('score', '')}")
    with t2:
        docs = svc.list_documents()
        if not docs:
            st.info("No documents yet.")
            return
        opts = {d["id"]: d["title"] for d in docs}
        did = st.selectbox("Document", list(opts), format_func=lambda i: opts[i],
                           key="vv_doc")
        drow = svc.db.get_document(int(did))
        npp = max(1, int(drow["page_count"] or 1))
        # number_input (not slider): a one-page document would make min == max,
        # which st.slider rejects outright
        pg = st.number_input("Page", 1, npp, 1, step=1, key="vv_page")
        import pandas as pd
        vals = svc.document_values(int(did), page=pg)
        if vals:
            st.dataframe(pd.DataFrame(vals), hide_index=True, **WIDTH_KW)
        else:
            st.info("No values with recognized engineering units on this page.")


# ====================================================== page: AI settings
def page_settings(svc) -> None:
    st.header("AI Provider Settings")
    st.markdown(
        "The assistant is **fully functional without any API**: offline "
        "extractive answers, citations and numeric verification need no paid "
        "service. Connect a provider only if you want generated prose answers.")
    cfg = svc.provider_config()
    if cfg.get("env_key_present"):
        sug = cfg.get("env_suggestion") or {}
        st.info(
            f"An API key for **{sug.get('label', 'a paid provider')}** is present in "
            "the environment, but it is **not** active: nothing is sent to it until "
            "you pick a provider here and save it. This tool never calls a paid "
            "endpoint without an explicit choice.", icon="🔒")
    kind_order = ["offline", "ollama", "openai", "anthropic", "openrouter",
                  "custom"]
    labels = {k: PROVIDER_KINDS[k]["label"] for k in kind_order}
    rev = {v: k for k, v in labels.items()}
    cur = rev.get(cfg.get("kind", "offline"), "offline")
    kind = st.selectbox("Provider", [labels[k] for k in kind_order],
                        index=kind_order.index(cur))
    sel_kind = next(k for k in kind_order if labels[k] == kind)
    if sel_kind == "offline":
        svc.save_provider({"kind": "offline"})
        st.success("Offline mode active. No network calls are made.")
        _usage_panel(svc)
        return
    meta = PROVIDER_KINDS[sel_kind]
    c1, c2 = st.columns(2)
    base = c1.text_input("API Base URL", value=cfg.get("base_url")
                         or meta.get("base", ""),
                         help="Ollama: http://localhost:11434 · OpenAI: "
                              "https://api.openai.com/v1 · or your gateway URL")
    key = c2.text_input("API Key", type="password",
                        value="", placeholder=cfg.get("api_key") and "•••• (stored)" or
                        ("not required" if not meta.get("needs_key") else "required"))
    model = st.text_input("Model name", value=cfg.get("model", ""),
                          help="leave empty to discover and pick from the provider")
    if st.button("🔎 List models", key="lm"):
        probe = {"kind": sel_kind, "base_url": base,
                 "api_key": key or cfg.get("api_key", ""), "model": ""}
        from engrag.llm.providers import build_provider
        pst = build_provider(probe).status()
        if pst.models:
            # keep the list so the picker survives Streamlit reruns (a widget
            # rendered only on the click event would vanish on the next one)
            st.session_state["model_options"] = pst.models
            st.success(f"{len(pst.models)} model(s) offered by this endpoint.")
        else:
            st.session_state.pop("model_options", None)
            st.warning(pst.detail or "No model list available.")
    opts = st.session_state.get("model_options") or []
    if opts:
        model = st.selectbox(
            "Pick a model from this provider's own list", opts,
            index=opts.index(model) if model in opts else 0, key="pickm")
    c5, c6, c7, c8 = st.columns(4)
    temp = c5.number_input("Temperature", 0.0, 1.5,
                           float(cfg.get("temperature", 0.1)), step=0.05)
    maxtok = c6.number_input("Max output tokens", 128, 4096,
                             int(cfg.get("max_tokens", 900)), step=64)
    tout = c7.number_input("Timeout (s)", 10, 600,
                           int(cfg.get("timeout_s", 90)), step=5)
    retries = c8.number_input("Retries (429/5xx, backoff)", 0, 4,
                              int(cfg.get("retries", 2)))
    actions = st.columns([1, 1, 6])
    if actions[0].button("Save configuration", type="primary", key="savep"):
        new = {"kind": sel_kind, "base_url": base, "model": model,
               "temperature": temp, "max_tokens": int(maxtok),
               "timeout_s": int(tout), "retries": int(retries)}
        if key:
            new["api_key"] = key
        elif cfg.get("kind") == sel_kind and cfg.get("api_key"):
            new["api_key"] = cfg["api_key"]
        svc.save_provider(new)
        st.success("Saved (key stored in local data directory only; never "
                   "logged or sent anywhere else).")
    if actions[1].button("Test connection", key="testp"):
        probe = {"kind": sel_kind, "base_url": base,
                 "api_key": key or cfg.get("api_key", ""), "model": model or "",
                 "timeout_s": int(tout)}
        from engrag.llm.providers import build_provider
        pst = build_provider(probe).status()
        (st.success if pst.ok else st.error)(
            f"{'OK' if pst.ok else 'Not reachable'}: {pst.detail} "
            f"({pst.latency_ms} ms)")
        if pst.ok and not model:
            pick = pick_default(pst.models, sel_kind)
            if pick:
                st.info(f"Suggested model from this provider's own list: `{pick}` — "
                        "nothing is saved automatically; press Save after confirming.")
    if sel_kind in ("openai", "anthropic", "openrouter"):
        st.caption("⚠ Billing: an API key bills against your provider's **API "
                   "balance** — a ChatGPT / Claude.ai consumer *subscription* "
                   "does NOT include API access. Requests are sent only after "
                   "you Save + use LLM mode; the app will not spend money "
                   "automatically.")
    st.info("Uploading a document does **not** send it anywhere. Document "
            "text is transmitted to the configured provider only when you ask "
            "a question in LLM mode, so treat that as an explicit opt-in for "
            "the retrieved passages.")
    _usage_panel(svc)


def pick_default(models: list[str], kind: str) -> str | None:
    from engrag.llm.providers import pick_default_model
    return pick_default_model(kind, models)


def _usage_panel(svc) -> None:
    st.divider()
    st.markdown("**Usage & budget**")
    row = svc.db.query_one(
        "SELECT COUNT(*) AS n, SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c,"
        " SUM(est_cost_usd) AS cost FROM qa_history WHERE mode='llm'")
    c1, c2, c3, c4 = st.columns(4)
    n = int(row["n"] or 0)
    c1.metric("LLM Q&A requests", n)
    c2.metric("prompt tokens (reported)", int(row["p"] or 0))
    c3.metric("completion tokens", int(row["c"] or 0))
    c4.metric("estimated spend (USD)", f"{(row['cost'] or 0):.4f}")
    st.caption("Token counts are reported by the provider when available; "
               "costs use an internal *estimate* price table — check your "
               "provider's dashboard for actuals.")
    warn = float(svc.db.get_setting("budget_warn_usd", 0.0) or 0.0)
    nw = st.number_input("Warn me when estimated spend exceeds (USD, 0 = off)",
                         value=warn, min_value=0.0, step=0.50)
    if nw != warn:
        svc.db.set_setting("budget_warn_usd", float(nw))
    total = float(row["cost"] or 0)
    if nw and total >= nw:
        st.error(f"⚠ Estimated spend ${total:.4f} exceeded your warning limit "
                 f"${nw:.2f}. Consider switching to offline mode.")


# ====================================================== page: history
def page_history(svc) -> None:
    st.header("History")
    rows = svc.list_history(60)
    if not rows:
        st.info("No questions asked yet.")
        return
    import pandas as pd
    df = pd.DataFrame([{"id": r["id"], "when": _ts(r["created_at"]),
                        "mode": r["mode"], "abstained": bool(r["abstained"]),
                        "question": r["question"][:80],
                        "ms": r["latency_ms"],
                        "est_cost": round(r["est_cost_usd"], 5)} for r in rows])
    st.dataframe(df, hide_index=True, **WIDTH_KW)
    qid = st.selectbox("Inspect entry", [r["id"] for r in rows],
                       format_func=lambda i: f"#{i}: "
                       + next(r["question"][:70] for r in rows if r["id"] == i))
    entry = svc.get_history(int(qid))
    st.markdown("**" + entry["question"] + "**")
    st.markdown(entry["answer"][:2500])
    if entry["citations"]:
        st.markdown("*Citations:*")
        for c in entry["citations"]:
            st.write(f"- [{c['integrity']}/{c['support']}] doc {c['document_id']} "
                     f"p.{c['page_number']}: “{(c['quote'] or '')[:110]}…”")
    if entry["num_checks"]:
        st.markdown("*Numeric checks:*")
        import collections
        cnt = collections.Counter(n["status"] for n in entry["num_checks"])
        st.write(" · ".join(f"{k}: {v}" for k, v in cnt.items()))
    cc1, cc2 = st.columns(2)
    if cc1.button("Delete this entry", key="hd1"):
        svc.delete_history(int(qid))
        st.rerun()
    if cc2.button("Clear ALL history", key="hd2"):
        svc.delete_history(None)
        st.rerun()


def _ts(t) -> str:
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(t)


# ====================================================== page: evaluation
def run_evaluation_now(top_k: int = 8) -> dict:
    """Run the evaluation in-process on a throwaway index.

    Called from the Evaluation view.  It deliberately does *not* shell out to
    ``scripts/run_eval.py``: that script only exists in a source checkout, so a
    packaged install would break.  Running the same service-layer code here
    works from any install and touches no existing documents.
    """
    from engrag.config import load_settings
    from engrag.eval import build_service, run_evaluation
    from engrag.logging_setup import utc_stamp

    with tempfile.TemporaryDirectory(prefix="engrag-eval-") as td:
        settings = load_settings({"ENGRAG_DATA_DIR": td})
        fresh = build_service(settings, Path(td) / "gen")
        report = run_evaluation(fresh, top_k=top_k)
    report["generated_at"] = utc_stamp()
    report["engine"] = {
        "embedding_backend": fresh.engine.embedder.status.backend,
        "fts5": fresh.db.fts5,
        "mode": "offline extractive + deterministic verification",
        "corpus": "synthetic demo manuals (engrag.demo_corpus)",
    }
    return report


def page_eval(svc) -> None:
    st.header("Evaluation Dashboard")
    st.caption("Reproducible evaluation over the synthetic corpus. Metrics are "
               "measured live — nothing here is hardcoded.")
    c1, c2 = st.columns([1, 4])
    if c1.button("Run full evaluation now", type="primary"):
        with st.spinner("Running the evaluation on a fresh temporary index "
                        "(this takes a few seconds)…"):
            try:
                data = run_evaluation_now()
            except Exception as exc:                     # surface, never hide
                data = None
                st.error(f"evaluation failed: {type(exc).__name__}: {exc}")
        if data:
            svc.s.data_dir.mkdir(parents=True, exist_ok=True)
            (svc.s.data_dir / "eval_results.json").write_text(
                json.dumps(data, indent=2))
            st.success(f"evaluation finished: "
                       f"{data['summary']['passed']}/{data['summary']['cases']} "
                       "cases passed.")
            st.session_state["eval_cache"] = data
    path = svc.s.data_dir / "eval_results.json"
    if path.exists():
        data = json.loads(path.read_text())
        st.session_state["eval_cache"] = data
        st.caption(f"stored results · generated {data.get('generated_at')} · "
                   f"engine {data.get('engine')}")
    data = st.session_state.get("eval_cache")
    if not data:
        st.info("No stored results yet — run the evaluation, or run "
                "`python scripts/run_eval.py` from the repository root.")
        return
    import pandas as pd
    sm = data["summary"]
    c = st.columns(6)
    c[0].metric("Recall@k (evidence)", f"{sm['recall@k']:.0%}")
    c[1].metric("Citation accuracy", f"{sm['citation_accuracy']:.0%}")
    c[2].metric("Abstention precision", f"{sm['abstention_precision']:.0%}")
    c[3].metric("Abstention recall", f"{sm['abstention_recall']:.0%}")
    c[4].metric("Numeric detection", f"{sm['numeric_status_accuracy']:.0%}")
    c[5].metric("Median latency", f"{sm['median_latency_ms']:.0f} ms")
    st.dataframe(pd.DataFrame(data["cases"]), hide_index=True, height=420, **WIDTH_KW)


# ================================================================== main
def main() -> None:
    style()
    svc = get_service()
    with st.sidebar:
        st.markdown("## 📘 Engineering RAG")
        st.caption(f"v{__import__('engrag').__version__} · local-first · "
                   "no paid API required")
        # A view can queue a navigation change ("open the source page").  A
        # widget's state key may only be written *before* that widget is
        # instantiated, so the request is picked up here instead of at the
        # button that made it.
        pending = st.session_state.pop("nav_pending", None)
        if pending in VIEWS:
            st.session_state["navradio"] = pending
        nav = st.session_state.get("nav", "ask")
        choice = st.radio("Views", VIEWS,
                          format_func=lambda x: {
                              "ask": "💬 Ask Engineering AI",
                              "documents": "🗄 Document Library",
                              "search": "🔍 Document Search",
                              "citations": "📎 Citation Viewer",
                              "values": "🧮 Value Verification",
                              "settings": "🤖 AI Settings",
                              "evaluation": "📊 Evaluation",
                              "history": "🕘 History"}[x],
                          index=VIEWS.index(nav if nav in VIEWS else "ask"),
                          key="navradio")
        st.session_state["nav"] = choice
        st.divider()
        stats = svc.stats()
        st.caption(f"📄 {stats['documents']} docs · 📃 {stats['pages']} pages · "
                   f"🧩 {stats['chunks']} chunks · {'FTS5 ✓' if stats['fts5'] else 'FTS5 ✗ (LIKE fallback)'}"
                   f" · embed: {stats['vectors']['backend']}")
        prov = svc.db.get_setting("active_provider") or {"kind": "offline"}
        st.caption("🤖 provider: " + str(prov.get("kind", "offline")))
        if prov.get("kind") != "offline":
            st.warning("LLM mode active — answers may cost money per the "
                       "provider's API pricing.")
    pages = {"ask": page_ask, "documents": page_documents,
             "search": page_search, "citations": page_viewer,
             "values": page_values, "settings": page_settings,
             "evaluation": page_eval, "history": page_history}
    pages[choice](svc)


if __name__ == "__main__":
    # `streamlit run` executes this file as __main__; importing it as a module
    # must not start the app (that would make the CI import check impossible)
    main()
