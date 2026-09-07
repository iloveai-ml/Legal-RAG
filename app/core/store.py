"""
The search index, sized to run inside a 512 MB container.

ChromaDB is gone. In its place:

  * vectors.npy is memory mapped, so the operating system pages the 29 MB of
    float32 embeddings in and out instead of holding them all resident.
  * chunk text lives in a sqlite file and is read only for the handful of rows
    that actually make it into a result set.
  * BM25 is a precomputed sparse CSC matrix, so a keyword query is one column
    slice and a row sum rather than a scan over every document.

Everything is read only at runtime, which means workers can share the files and
a restart costs nothing but a few file opens.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from . import config

log = logging.getLogger("nyaya.store")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Chunk:
    """One retrievable passage with everything needed to cite it."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    section_marker: str
    paragraph_index: int
    text: str
    jurisdiction: str = "IN"
    source_url: str = ""
    source_note: str = ""
    source_task: str = ""
    outcome: int = -1
    origin: str = "corpus"  # "corpus" or "upload"

    def citation_label(self) -> str:
        parts = [self.doc_title]
        if self.section_marker:
            parts.append(self.section_marker)
        parts.append(f"para {self.paragraph_index}")
        return ", ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "section_marker": self.section_marker,
            "paragraph_index": self.paragraph_index,
            "text": self.text,
            "jurisdiction": self.jurisdiction,
            "source_url": self.source_url,
            "source_note": self.source_note,
            "source_task": self.source_task,
            "outcome": self.outcome,
            "origin": self.origin,
        }


@dataclass
class Scored:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


class IndexUnavailable(RuntimeError):
    """The prebuilt index files are missing or unreadable."""


_COLUMNS = (
    "chunk_id, doc_id, doc_title, doc_type, section_marker, paragraph_index, "
    "text, jurisdiction, source_url, source_note, source_task, outcome"
)


def _row_to_chunk(row: Sequence[Any]) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        doc_id=row[1],
        doc_title=row[2],
        doc_type=row[3],
        section_marker=row[4],
        paragraph_index=int(row[5]),
        text=row[6],
        jurisdiction=row[7],
        source_url=row[8],
        source_note=row[9],
        source_task=row[10],
        outcome=int(row[11]),
    )


class CorpusIndex:
    """Hybrid dense plus sparse search over the prebuilt corpus."""

    def __init__(self, index_dir: Path | None = None):
        self.dir = Path(index_dir or config.INDEX_DIR)
        self.manifest: dict[str, Any] = {}
        self.ready = False
        self.error: str | None = None

        self._vectors: np.ndarray | None = None
        self._bm25: sparse.csc_matrix | None = None
        self._vocab: dict[str, int] = {}
        self._local = threading.local()
        self._doc_types: np.ndarray | None = None
        self._source_tasks: np.ndarray | None = None

        self._load()

    # ------------------------------------------------------------ loading

    def _load(self) -> None:
        try:
            manifest_path = self.dir / "manifest.json"
            if not manifest_path.exists():
                raise IndexUnavailable(
                    f"No index found at {self.dir}. Run scripts/build_index.py first."
                )
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self._vectors = np.load(self.dir / "vectors.npy", mmap_mode="r")
            self._bm25 = sparse.load_npz(self.dir / "bm25.npz").tocsc()
            self._vocab = json.loads(
                (self.dir / "bm25_vocab.json").read_text(encoding="utf-8")
            )

            # Small per row arrays kept resident so filters never touch sqlite
            with self._connect() as con:
                rows = con.execute(
                    "SELECT doc_type, source_task FROM chunks ORDER BY row_id"
                ).fetchall()
            self._doc_types = np.array([r[0] for r in rows], dtype=object)
            self._source_tasks = np.array([r[1] for r in rows], dtype=object)

            if len(self._doc_types) != self._vectors.shape[0]:
                raise IndexUnavailable(
                    f"Index is inconsistent: {self._vectors.shape[0]} vectors "
                    f"but {len(self._doc_types)} chunk rows."
                )

            self.ready = True
            log.info(
                "Index ready: %s chunks, %s documents",
                self.manifest.get("n_chunks"),
                self.manifest.get("n_documents"),
            )
        except Exception as exc:  # noqa: BLE001 - the app must still boot
            self.error = str(exc)
            log.error("Index failed to load: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        """One read only connection per thread, since sqlite objects are not shared."""
        con = getattr(self._local, "con", None)
        if con is None:
            path = self.dir / "chunks.sqlite"
            con = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True, check_same_thread=False
            )
            self._local.con = con
        return con

    # ------------------------------------------------------------ stats

    @property
    def n_chunks(self) -> int:
        return int(self.manifest.get("n_chunks", 0))

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "error": self.error,
            "n_chunks": self.n_chunks,
            "n_documents": int(self.manifest.get("n_documents", 0)),
            "doc_types": self.manifest.get("doc_types", {}),
            "outcome_tasks": self.manifest.get("outcome_tasks", {}),
            "embed_model": self.manifest.get("embed_model", config.EMBED_MODEL),
            "built_at": self.manifest.get("built_at", ""),
        }

    # ------------------------------------------------------------ fetching

    def fetch(self, row_ids: Sequence[int]) -> dict[int, Chunk]:
        if not row_ids:
            return {}
        placeholders = ",".join("?" * len(row_ids))
        con = self._connect()
        rows = con.execute(
            f"SELECT row_id, {_COLUMNS} FROM chunks WHERE row_id IN ({placeholders})",
            [int(r) for r in row_ids],
        ).fetchall()
        return {int(r[0]): _row_to_chunk(r[1:]) for r in rows}

    # ------------------------------------------------------------ masking

    def _mask(
        self,
        doc_types: Sequence[str] | None,
        source_task: str | None,
    ) -> np.ndarray | None:
        mask: np.ndarray | None = None
        if doc_types:
            wanted = {t.lower() for t in doc_types}
            mask = np.array(
                [str(t).lower() in wanted for t in self._doc_types], dtype=bool
            )
        if source_task:
            task_mask = np.array(
                [str(t) == source_task for t in self._source_tasks], dtype=bool
            )
            mask = task_mask if mask is None else (mask & task_mask)
        return mask

    # ------------------------------------------------------------ search

    def vector_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity against every chunk. Vectors are pre normalised."""
        assert self._vectors is not None
        return np.asarray(self._vectors @ query_vector, dtype=np.float32)

    def keyword_scores(self, query: str) -> np.ndarray:
        """BM25 score for every chunk, as one sparse column slice plus a row sum."""
        assert self._bm25 is not None
        columns = [self._vocab[t] for t in set(tokenize(query)) if t in self._vocab]
        n_docs = self._bm25.shape[0]
        if not columns:
            return np.zeros(n_docs, dtype=np.float32)
        sliced = self._bm25[:, columns]
        return np.asarray(sliced.sum(axis=1), dtype=np.float32).ravel()

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k: int = 8,
        doc_types: Sequence[str] | None = None,
        boost_doc_types: Sequence[str] | None = None,
        source_task: str | None = None,
    ) -> list[Scored]:
        """
        Hybrid retrieval fused with reciprocal rank fusion.

        RRF is used rather than a weighted score blend because the dense and
        sparse scores live on completely different scales, and rank based
        fusion is robust to that without needing per query calibration.

        doc_types is a hard filter and should only carry a choice the user made
        explicitly. boost_doc_types is a soft preference that nudges matching
        passages up without ever removing anything, which is what a guess from
        the query router should be allowed to do.
        """
        if not self.ready or not query.strip():
            return []

        mask = self._mask(doc_types, source_task)
        if mask is not None and not mask.any():
            return []

        pool = max(top_k * config.CANDIDATE_MULTIPLIER, top_k)

        dense = self.vector_scores(query_vector)
        sparse_scores = self.keyword_scores(query)
        if mask is not None:
            dense = np.where(mask, dense, -np.inf)
            sparse_scores = np.where(mask, sparse_scores, -np.inf)

        dense_top = _top_indices(dense, pool)
        sparse_top = _top_indices(sparse_scores, pool)

        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_top)}
        sparse_rank = {int(idx): rank for rank, idx in enumerate(sparse_top)}

        fused: dict[int, float] = {}
        for idx, rank in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)
        for idx, rank in sparse_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)

        if boost_doc_types:
            preferred = {t.lower() for t in boost_doc_types}
            assert self._doc_types is not None
            for idx in fused:
                if str(self._doc_types[idx]).lower() in preferred:
                    fused[idx] *= config.DOC_TYPE_BOOST

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        chunks = self.fetch([idx for idx, _ in ordered])

        results: list[Scored] = []
        for idx, score in ordered:
            chunk = chunks.get(idx)
            if chunk is None:
                continue
            results.append(
                Scored(
                    chunk=chunk,
                    score=float(score),
                    vector_rank=dense_rank.get(idx),
                    keyword_rank=sparse_rank.get(idx),
                )
            )
        return results

    def similar_by_task(
        self,
        query_vector: np.ndarray,
        *,
        source_task: str,
        top_k: int = 40,
    ) -> list[Scored]:
        """Pure dense similarity within one labelled outcome dataset."""
        if not self.ready:
            return []
        mask = self._mask(None, source_task)
        if mask is None or not mask.any():
            return []

        dense = self.vector_scores(query_vector)
        dense = np.where(mask, dense, -np.inf)
        top = _top_indices(dense, top_k)
        chunks = self.fetch([int(i) for i in top])

        out: list[Scored] = []
        for rank, idx in enumerate(top):
            chunk = chunks.get(int(idx))
            if chunk is None or not np.isfinite(dense[idx]):
                continue
            out.append(Scored(chunk=chunk, score=float(dense[idx]), vector_rank=rank))
        return out


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, ordered best first, skipping -inf."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    part = part[np.isfinite(scores[part])]
    return part[np.argsort(-scores[part])]


# ---------------------------------------------------------------- singleton

_index: CorpusIndex | None = None
_index_lock = threading.Lock()


def get_index() -> CorpusIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = CorpusIndex()
    return _index


# ---------------------------------------------------------------- uploads

@dataclass
class UploadedDocument:
    """A document a visitor uploaded, embedded and searchable for their session."""

    doc_id: str
    filename: str
    title: str
    chunks: list[Chunk]
    vectors: np.ndarray
    created_at: float
    n_pages: int = 0
    truncated: bool = False
    bm25_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def search(
        self, query: str, query_vector: np.ndarray, top_k: int
    ) -> list[Scored]:
        if not self.chunks:
            return []
        dense = np.asarray(self.vectors @ query_vector, dtype=np.float32)

        # A short in memory BM25 over just this document, so an exact clause
        # match still surfaces even when the wording is unusual
        terms = set(tokenize(query))
        keyword = np.zeros(len(self.chunks), dtype=np.float32)
        if terms:
            for i, chunk in enumerate(self.chunks):
                tokens = tokenize(chunk.text)
                if not tokens:
                    continue
                hits = sum(1 for t in tokens if t in terms)
                keyword[i] = hits / (len(tokens) ** 0.5)

        dense_rank = {int(i): r for r, i in enumerate(_top_indices(dense, len(self.chunks)))}
        keyword_rank = {int(i): r for r, i in enumerate(_top_indices(keyword, len(self.chunks)))}

        fused: dict[int, float] = {}
        for i, r in dense_rank.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + r)
        for i, r in keyword_rank.items():
            if keyword[i] > 0:
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + r)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Scored(
                chunk=self.chunks[i],
                score=float(score),
                vector_rank=dense_rank.get(i),
                keyword_rank=keyword_rank.get(i),
            )
            for i, score in ordered
        ]
