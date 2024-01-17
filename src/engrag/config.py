"""Central configuration.

Everything is driven by environment variables (prefix ``ENGRAG_``), optionally
loaded from a ``.env`` file in the working directory.  The application has zero
mandatory external services: with no configuration at all it runs fully
offline on a local SQLite database with FTS5 keyword retrieval.
"""
from __future__ import annotations

import sys

if sys.getrecursionlimit() < 6000:      # CPython 3.13 re.compile of wide char
    sys.setrecursionlimit(8000)         # classes silently no-ops below ~1500

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _default_data_dir() -> Path:
    env = os.environ.get("ENGRAG_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / "data"


@dataclass
class Settings:
    """Runtime settings.  See ``.env.example`` for the full list."""

    data_dir: Path = field(default_factory=_default_data_dir)
    app_name: str = "Engineering RAG Assistant"
    host: str = "0.0.0.0"
    api_port: int = 8502
    web_port: int = 8501

    # Upload / ingestion safety limits
    max_upload_mb: int = 60
    max_pages: int = 1500
    max_extracted_chars_per_doc: int = 6_000_000
    allow_private_network_endpoints: bool = True  # local single-user tool

    # Chunking
    chunk_min_chars: int = 400
    chunk_target_chars: int = 1500
    chunk_max_chars: int = 2400
    chunk_overlap_chars: int = 160

    # Retrieval
    default_top_k: int = 6
    max_top_k: int = 20
    candidate_pool: int = 60
    relevance_threshold: float = 0.02
    min_evidence_score: float = 0.10   # below this, treat as weak evidence
    enable_semantic: bool = True       # only used if a semantic backend loads
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Numerical verification.  Default tolerance is exact match (so no
    # discrepancy is ever silently rounded away); users may open a small
    # engineering tolerance explicitly, capped at max_rel_tolerance.
    default_rel_tolerance: float = 1e-9
    max_rel_tolerance: float = 0.05

    # LLM defaults (offline mode is the default; nothing is called remotely
    # until the user configures and tests a provider).
    llm_mode: str = "offline"
    llm_max_output_tokens: int = 900
    llm_timeout_s: int = 90
    llm_max_retries: int = 2

    # Optional soft budget for paid providers (USD).  0 disables the warning.
    budget_warn_usd: float = 0.0

    # UI
    history_enabled: bool = True
    max_history_rows: int = 500
    pdf_render_scale: float = 1.6
    max_render_pixels: int = 22_000_000

    # Derived paths -----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "engrag.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.uploads_dir, self.renders_dir, self.export_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build settings from environment (``ENGRAG_*``) with type coercion."""
    try:  # optional convenience: load a local .env if the user created one
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is optional at runtime
        pass

    raw = dict(os.environ) if env is None else dict(env)
    s = Settings()
    coerce = {
        "data_dir": lambda v: Path(v).expanduser().resolve(),
        "app_name": str,
        "host": str,
        "api_port": int,
        "web_port": int,
        "max_upload_mb": int,
        "max_pages": int,
        "max_extracted_chars_per_doc": int,
        "allow_private_network_endpoints": lambda v: v.lower() in ("1", "true", "yes"),
        "chunk_min_chars": int,
        "chunk_target_chars": int,
        "chunk_max_chars": int,
        "chunk_overlap_chars": int,
        "default_top_k": int,
        "max_top_k": int,
        "candidate_pool": int,
        "relevance_threshold": float,
        "min_evidence_score": float,
        "enable_semantic": lambda v: v.lower() in ("1", "true", "yes"),
        "embedding_model": str,
        "default_rel_tolerance": float,
        "max_rel_tolerance": float,
        "llm_mode": str,
        "llm_max_output_tokens": int,
        "llm_timeout_s": int,
        "llm_max_retries": int,
        "budget_warn_usd": float,
        "history_enabled": lambda v: v.lower() in ("1", "true", "yes"),
        "max_history_rows": int,
        "pdf_render_scale": float,
        "max_render_pixels": int,
    }
    for key, cast in coerce.items():
        env_key = f"ENGRAG_{key.upper()}"
        if env_key in raw and raw[env_key] != "":
            try:
                setattr(s, key, cast(raw[env_key]))
            except (TypeError, ValueError):
                pass  # keep defaults on malformed values; startup check logs it
    return s


def settings_as_dict(s: Settings) -> dict:
    d = asdict(s)
    d["data_dir"] = str(s.data_dir)
    return d
