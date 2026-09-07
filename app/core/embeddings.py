"""
Local ONNX embeddings.

Embeddings are deliberately never sent to an API. The model is a 90 MB ONNX
build of all-MiniLM-L6-v2 that runs on CPU, which buys three things that matter
for this deployment:

  * Retrieval works with zero API keys, so the app is never a blank screen.
  * Query cost is zero, so a public demo cannot run up a bill.
  * The prebuilt index and the live query encoder can never drift apart, which
    is the usual way a swapped embedding provider silently breaks a RAG app.

The model loads lazily on first use and is shared across requests behind a lock.
"""
from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np

from . import config

log = logging.getLogger("nyaya.embeddings")

_model = None
_load_lock = threading.Lock()
_embed_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from fastembed import TextEmbedding

                log.info("Loading embedding model %s", config.EMBED_MODEL)
                _model = TextEmbedding(config.EMBED_MODEL)
                log.info("Embedding model ready")
    return _model


def warm_up() -> bool:
    """Load the model ahead of the first request. Returns True on success."""
    try:
        embed_query("warm up")
        return True
    except Exception as exc:  # noqa: BLE001 - the app still serves without it
        log.error("Embedding model failed to warm up: %s", exc)
        return False


def is_ready() -> bool:
    return _model is not None


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Embed a batch of texts into L2 normalised float32 rows."""
    items = [t if t.strip() else " " for t in texts]
    if not items:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    model = _get_model()
    # onnxruntime sessions are not safe to call concurrently from many threads
    with _embed_lock:
        vectors = np.array(list(model.embed(items, batch_size=batch_size)), dtype=np.float32)
    return _normalise(vectors)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query into one L2 normalised float32 vector."""
    return embed_texts([text])[0]
