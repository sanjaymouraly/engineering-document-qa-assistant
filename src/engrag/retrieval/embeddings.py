"""Semantic embeddings with a dependency-light, always-available fallback.

Backend resolution order:
1. ``sentence-transformers`` (if installed and model cached/downloadable).
2. ``hashing-ngrams``: deterministic signed-hashing of word+character
   n-grams with IDF weighting.  No model download, works offline everywhere,
   and gives genuine lexical-semantic blending (useful for morphological
   variants like ``derating``/``derate``).  It is clearly reported as a
   *bag-of-ngrams* model, not a neural embedding, so users know its limits.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass

import numpy as np

from ..logging_setup import get_logger

log = get_logger("retrieval.embeddings")

_DIM = 512
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")


@dataclass
class EmbeddingStatus:
    backend: str = "none"        # sentence-transformers | hashing-ngrams | none
    model_id: str = ""
    dim: int = 0
    ready: bool = False
    note: str = ""


class HashingNGramEmbedder:
    """Deterministic hashed n-gram embedder (pure numpy, no downloads)."""

    backend = "hashing-ngrams"
    dim = _DIM

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._lock = threading.Lock()

    @staticmethod
    def _features(text: str) -> list[str]:
        toks = _tokenize(text)
        feats: list[str] = []
        for t in toks:
            feats.append("w:" + t)
            if len(t) >= 6:                      # suffix/prefix trigrams
                padded = f"#{t}#"
                for i in range(len(padded) - 2):
                    g = padded[i:i + 3]
                    if not g.isalpha():
                        continue
                    feats.append("c:" + g)
        for a, b in zip(toks, toks[1:]):
            feats.append("b:" + a + "_" + b)
        return feats

    @staticmethod
    def _hash(feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % _DIM
        sign = 1.0 if h[4] % 2 else -1.0
        return idx, sign

    def fit(self, texts: list[str]) -> None:
        df: dict[str, int] = {}
        with self._lock:
            self._doc_count = len(texts)
            for t in texts:
                seen = {f for f in self._features(t) if f.startswith(("w:", "b:"))}
                for f in seen:
                    df[f] = df.get(f, 0) + 1
            n = max(1, len(texts))
            self._idf = {f: math.log((n + 1) / (c + 0.5)) for f, c in df.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                idf = self._idf.get(feat, 1.0 if feat.startswith("c:") else math.log(2.0))
                idx, sign = self._hash(feat)
                out[row, idx] += sign * idf
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def sentence_transformers_available() -> bool:
    """Is the optional ``sentence-transformers`` package importable here?

    Used to decide whether it is worth *attempting* the neural backend: when
    the package is absent the service goes straight to the deterministic hashing
    model instead of logging a failure on every start.  Availability of the
    package is not the same as availability of the model — a missing or
    undownloadable model is still handled by ``EmbeddingService._ensure``.
    """
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class SentenceTransformerEmbedder:
    backend = "sentence-transformers"

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_id)
        self.model_id = model_id
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def fit(self, texts: list[str]) -> None:  # interface parity
        return None


class EmbeddingService:
    """Manages backend selection; failures degrade to hashing (never crash)."""

    def __init__(self, *, prefer_sentence_transformers: bool = True,
                 model_id: str = ""):
        self.prefer_st = prefer_sentence_transformers
        self.model_id = model_id
        self._embedder: HashingNGramEmbedder | SentenceTransformerEmbedder | None = None
        self._status = EmbeddingStatus()
        self._lock = threading.Lock()

    @property
    def status(self) -> EmbeddingStatus:
        if self._embedder is None:
            self._ensure()
        return self._status

    def _ensure(self) -> None:
        with self._lock:
            if self._embedder is not None:
                return
            if self.prefer_st and self.model_id:
                try:
                    emb = SentenceTransformerEmbedder(self.model_id)
                    self._embedder = emb
                    self._status = EmbeddingStatus(
                        backend=emb.backend, model_id=self.model_id, dim=emb.dim,
                        ready=True, note="neural embeddings")
                    return
                except Exception as exc:
                    log.info("sentence-transformers unavailable (%s); "
                             "using hashing fallback", exc)
            emb = HashingNGramEmbedder()
            self._embedder = emb
            self._status = EmbeddingStatus(
                backend=emb.backend, model_id="hashing-ngrams-v1", dim=_DIM,
                ready=True,
                note="deterministic hashed n-gram model (no downloads; "
                     "lexical-semantic, not a neural encoder)")

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        assert self._embedder is not None
        return self._embedder.embed(texts)

    def fit(self, texts: list[str]) -> None:
        self._ensure()
        assert self._embedder is not None
        self._embedder.fit(texts)

    @property
    def model_tag(self) -> str:
        st = self.status
        return st.model_id or "hashing-ngrams-v1"
