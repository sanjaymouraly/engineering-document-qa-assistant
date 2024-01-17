"""Shared UI helpers for the Streamlit frontend."""
from __future__ import annotations

import base64
import re
from pathlib import Path

import streamlit as st

STYLE = """
<style>
.small { font-size: 0.83rem; color: #5a6b7d; }
.badge { display:inline-block; padding:0.12rem 0.5rem; border-radius:0.6rem;
         font-size:0.75rem; font-weight:600; margin-right:0.25rem;}
.b-ok { background:#e5f5e9; color:#1c6b2e;}
.b-warn { background:#fdf3d7; color:#8a6410;}
.b-bad { background:#fde4e1; color:#a02b1c;}
.b-info { background:#e7effc; color:#26528f;}
.b-mute { background:#eef1f4; color:#5a6b7d;}
.quote { border-left:3px solid #9db2cc; padding:0.4rem 0.7rem; background:#f6f9fc;
         font-size:0.85rem; white-space:pre-wrap; }
.evbox { border:1px solid #dde5ee; border-radius:8px; padding:0.5rem 0.7rem;
         margin-bottom:0.45rem; background:#fbfdff;}
kbd { font-size:0.72rem; background:#eef1f4; border-radius:4px; padding:0 4px;}
</style>
"""

STATUS_BADGE = {
    "ready": ("ok", "ready"), "needs_ocr": ("warn", "needs OCR"),
    "processing": ("info", "indexing…"), "pending": ("info", "queued"),
    "failed": ("bad", "failed"),
    "verified": ("ok", "verified"), "unit_verified": ("ok", "unit-verified"),
    "verified_within_tolerance": ("warn", "verified (tolerance)"),
    "mismatch": ("bad", "MISMATCH"), "ambiguous": ("warn", "ambiguous"),
    "insufficient": ("warn", "insufficient evidence"),
    "supported": ("ok", "supported"), "weak": ("warn", "weak"),
    "contradicted": ("bad", "contradicted"),
    "unknown": ("mute", "unknown"),
    "ok": ("ok", "ok"), "stale_chunk": ("bad", "stale chunk"),
    "missing_quote": ("bad", "quote not found"), "invalid_ref": ("bad", "invalid ref"),
    "grounded": ("ok", "grounded"), "partially_grounded": ("warn", "partially grounded"),
    "unverified": ("bad", "unverified"), "abstained": ("info", "abstained"),
    "extractive": ("info", "offline extractive"), "llm": ("info", "LLM"),
}


def badge(kind: str) -> str:
    return f'<span class="badge b-{kind}">'


def support_badge(status: str) -> str:
    cls, label = STATUS_BADGE.get(status, ("mute", status))
    return f'<span class="badge b-{cls}">{label}</span>'


def style() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


def get_service():
    """Direct in-process service (no network needed)."""
    import sys
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from engrag.config import load_settings
    from engrag.service import Service
    if "svc" not in st.session_state:
        st.session_state.svc = Service(load_settings())
    return st.session_state.svc


def md_to_download(answer: str, title: str, filename: str,
                   fmt: str = "md") -> None:
    st.download_button(("Copy answer + citations" if fmt == "copy"
                        else f"Download {fmt}"),
                       data=answer, file_name=filename, mime=f"text/{fmt}",
                       key=f"dl_{title[:20]}")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]", "_", name or "answer")


def _width_kwargs() -> dict:
    """Widget-width kwarg that works across Streamlit versions.

    ``use_container_width`` was renamed to ``width="stretch"`` in Streamlit
    1.49 and now emits a deprecation warning; older releases do not know
    ``width``.  Resolve once at import time so the UI stays quiet on both.
    """
    import inspect

    import streamlit as st
    params = inspect.signature(st.button).parameters
    if "use_container_width" not in params:
        return {"width": "stretch"}
    try:
        ver = tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        ver = (0, 0)
    return {"width": "stretch"} if ver >= (1, 49) else {"use_container_width": True}


WIDTH_KW = _width_kwargs()


def image_bytes_to_md(png: bytes, caption: str = "") -> None:
    b64 = base64.b64encode(png).decode()
    st.markdown(f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;border:1px solid #dde5ee;'
                f'border-radius:8px;" alt="rendered page">',
                unsafe_allow_html=True)
    if caption:
        st.caption(caption)
