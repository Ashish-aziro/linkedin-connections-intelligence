"""Local sentence-transformers embeddings + brute-force cosine (spec §28).

The model loads lazily on first use (heavy import). Vectors are stored as
little-endian float32 bytes; a ``VectorStore`` swap to pgvector later only
touches ``search_similar``.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import settings

log = logging.getLogger("app.embeddings")

_model = None
_lock = threading.Lock()


def _hash_vector(text: str) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens vector — used when the ST model is
    disabled (fast tests) or unavailable. Not as good semantically, same shape."""
    import hashlib

    dim = settings.embedding_dim
    v = np.zeros(dim, dtype=np.float32)
    for tok in (text or "").lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
        v[(h >> 16) % dim] += 0.5
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import time as _time

                from sentence_transformers import SentenceTransformer

                log.info("loading embedding model %s", settings.embedding_model)
                t0 = _time.perf_counter()
                _model = SentenceTransformer(settings.embedding_model)
                log.info("embedding model loaded in %.1fs", _time.perf_counter() - t0)
    return _model


def preload() -> None:
    """TASK 5 — load the embedding model now (app startup) instead of paying
    the load cost inside the first search after a restart. A no-op once
    already loaded (``_get_model``'s double-checked lock). Never raises —
    startup must not fail because an optional local model couldn't load; the
    lazy path in ``embed_text``/``embed_texts`` remains the fallback."""
    if not settings.embeddings_enabled:
        return
    try:
        _get_model()
    except Exception:  # noqa: BLE001
        log.exception("embedding model preload failed — will retry lazily on first use")


def embed_text(text: str) -> bytes:
    from app.services import search_profile

    search_profile.incr("embedding_calls")
    if not settings.embeddings_enabled:
        return _hash_vector(text).tobytes()
    try:
        vec = _get_model().encode([text or ""], normalize_embeddings=True)[0]
    except Exception:  # noqa: BLE001
        log.exception("ST model unavailable — falling back to hash vectors")
        return _hash_vector(text).tobytes()
    return np.asarray(vec, dtype=np.float32).tobytes()


def embed_texts(texts: list[str]) -> list[bytes]:
    if not texts:
        return []
    if not settings.embeddings_enabled:
        return [_hash_vector(t).tobytes() for t in texts]
    vecs = _get_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return [np.asarray(v, dtype=np.float32).tobytes() for v in vecs]


def to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_scores(query: bytes, rows: list[tuple[str, bytes]]) -> list[tuple[str, float]]:
    """Vectors are already L2-normalized → cosine == dot product."""
    if not rows:
        return []
    q = to_array(query)
    out: list[tuple[str, float]] = []
    for pid, blob in rows:
        v = to_array(blob)
        if v.shape != q.shape:
            continue
        out.append((pid, float(np.dot(q, v))))
    out.sort(key=lambda t: t[1], reverse=True)
    return out
