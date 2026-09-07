"""
Build the compact, deploy-ready search index for Nyaya.

Reads the legacy ChromaDB sqlite corpus, re-embeds every chunk with the local
ONNX MiniLM model, and writes small artifacts that the web app loads at boot
without needing ChromaDB at all:

    data/index/vectors.npy      float32 (N, 384), L2 normalised, memory mapped at runtime
    data/index/chunks.sqlite    chunk text and provenance, queried on demand
    data/index/bm25.npz         precomputed BM25 weights as a sparse CSC matrix
    data/index/bm25_vocab.json  term to column mapping
    data/index/manifest.json    corpus statistics shown in the UI

Usage:
    python scripts/build_index.py --source ../legal-rag/data/chroma/chroma.sqlite3
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "index"

BM25_K1 = 1.5
BM25_B = 0.75
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# The original ingest wrote a handful of document titles through the wrong
# codec, so an em dash separator survives as U+FFFD. Repair those here and
# normalise every dash separator to a plain hyphen so nothing downstream has to
# deal with mojibake or with dash characters the interface does not use.
def clean_title(title: str) -> str:
    """Titles and provenance notes, normalised then trimmed of dangling dashes."""
    return clean_text(title).strip(" -")


# Source documents carry typographic characters that the interface does not
# use: em and en dashes, non breaking hyphens and spaces, zero width joiners.
# Normalising them here, before embedding, means the vectors always match the
# text that is displayed and that citation quotes are checked against.
_CHAR_MAP = {
    "�": " - ",   # replacement character, an em dash lost to a bad codec
    "—": " - ",   # em dash
    "–": " - ",   # en dash
    "‒": " - ",   # figure dash
    "―": " - ",   # horizontal bar
    "‑": "-",     # non breaking hyphen
    " ": " ",     # no break space
    " ": " ",     # narrow no break space
    " ": " ",     # thin space
    "​": "",      # zero width space
    "﻿": "",      # byte order mark
}
_CHAR_RE = re.compile("|".join(map(re.escape, _CHAR_MAP)))
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    out = _CHAR_RE.sub(lambda m: _CHAR_MAP[m.group(0)], text)
    return _SPACE_RUN_RE.sub(" ", out).strip()


# ---------------------------------------------------------------- extraction

def read_chroma(sqlite_path: Path) -> list[dict]:
    """Pull every chunk with its text and metadata out of a ChromaDB sqlite file."""
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # embedding_metadata.id joins back to embeddings.id for the public chunk id
    id_map: dict[int, str] = {
        row["id"]: row["embedding_id"]
        for row in con.execute("SELECT id, embedding_id FROM embeddings")
    }

    records: dict[int, dict] = {}
    rows = con.execute(
        "SELECT id, key, string_value, int_value, float_value FROM embedding_metadata"
    )
    for row in rows:
        rec = records.setdefault(row["id"], {})
        if row["string_value"] is not None:
            rec[row["key"]] = row["string_value"]
        elif row["int_value"] is not None:
            rec[row["key"]] = row["int_value"]
        elif row["float_value"] is not None:
            rec[row["key"]] = row["float_value"]
    con.close()

    out: list[dict] = []
    for internal_id, rec in records.items():
        text = clean_text(rec.get("chroma:document") or "")
        chunk_id = id_map.get(internal_id)
        if not text or not chunk_id:
            continue

        extra: dict = {}
        raw_extra = rec.get("extra_json")
        if raw_extra:
            try:
                extra = json.loads(raw_extra)
            except (json.JSONDecodeError, TypeError):
                extra = {}
        if "source_task" in rec:
            extra["source_task"] = rec["source_task"]
        if "outcome" in rec:
            extra["outcome"] = int(rec["outcome"])

        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": rec.get("doc_id") or "",
                "doc_title": clean_title(rec.get("doc_title") or ""),
                "doc_type": rec.get("doc_type") or "",
                "section_marker": clean_title(rec.get("section_marker") or ""),
                "paragraph_index": int(rec.get("paragraph_index") or 0),
                "text": text,
                "jurisdiction": rec.get("jurisdiction") or "IN",
                "source_url": rec.get("source_url") or "",
                # Provenance notes carry the same dash separators as the titles
                "source_note": clean_title(rec.get("source_note") or ""),
                "source_task": str(extra.get("source_task") or ""),
                "outcome": int(extra.get("outcome", -1)),
            }
        )

    # Stable order so row index N in vectors.npy always means the same chunk
    out.sort(key=lambda r: (r["doc_id"], r["paragraph_index"], r["chunk_id"]))
    return out


# ---------------------------------------------------------------- embeddings

def embed_all(texts: list[str], batch_size: int = 128) -> np.ndarray:
    from fastembed import TextEmbedding

    print(f"Loading embedding model {EMBED_MODEL} ...", flush=True)
    model = TextEmbedding(EMBED_MODEL)

    vectors = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    started = time.perf_counter()
    written = 0
    for vec in model.embed(texts, batch_size=batch_size):
        vectors[written] = vec
        written += 1
        if written % 2000 == 0 or written == len(texts):
            rate = written / max(time.perf_counter() - started, 1e-6)
            remaining = (len(texts) - written) / max(rate, 1e-6)
            print(
                f"  embedded {written:,}/{len(texts):,} "
                f"({rate:.0f}/s, about {remaining:.0f}s left)",
                flush=True,
            )
    if written != len(texts):
        raise SystemExit(f"Embedder returned {written} vectors for {len(texts)} texts.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------- bm25

def build_bm25(texts: list[str]) -> tuple[sparse.csc_matrix, dict[str, int]]:
    """
    Precompute the document side of the BM25 score as a sparse matrix.

        W[d, t] = idf(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len_d / avgdl))

    At query time a document score is the sum of the columns for the query
    terms, which is one sparse column slice plus a row sum.
    """
    print("Tokenising for BM25 ...", flush=True)
    doc_terms: list[Counter] = []
    doc_lengths = np.zeros(len(texts), dtype=np.float32)
    vocab: dict[str, int] = {}

    for i, text in enumerate(texts):
        toks = tokenize(text)
        doc_lengths[i] = len(toks)
        counts = Counter(toks)
        doc_terms.append(counts)
        for term in counts:
            if term not in vocab:
                vocab[term] = len(vocab)

    n_docs = len(texts)
    avgdl = float(doc_lengths.mean()) if n_docs else 1.0
    print(f"  {n_docs:,} docs, {len(vocab):,} terms, average length {avgdl:.0f}", flush=True)

    doc_freq = np.zeros(len(vocab), dtype=np.int64)
    for counts in doc_terms:
        for term in counts:
            doc_freq[vocab[term]] += 1
    idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)

    nnz = sum(len(c) for c in doc_terms)
    rows = np.zeros(nnz, dtype=np.int32)
    cols = np.zeros(nnz, dtype=np.int32)
    vals = np.zeros(nnz, dtype=np.float32)

    at = 0
    for i, counts in enumerate(doc_terms):
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * doc_lengths[i] / avgdl)
        for term, freq in counts.items():
            j = vocab[term]
            rows[at] = i
            cols[at] = j
            vals[at] = idf[j] * (freq * (BM25_K1 + 1.0)) / (freq + norm)
            at += 1

    matrix = sparse.csc_matrix((vals, (rows, cols)), shape=(n_docs, len(vocab)))
    print(f"  {matrix.nnz:,} non-zero weights", flush=True)
    return matrix, vocab


# ---------------------------------------------------------------- persistence

SCHEMA = """
CREATE TABLE chunks (
    row_id          INTEGER PRIMARY KEY,
    chunk_id        TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    doc_title       TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    section_marker  TEXT NOT NULL,
    paragraph_index INTEGER NOT NULL,
    text            TEXT NOT NULL,
    jurisdiction    TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    source_note     TEXT NOT NULL,
    source_task     TEXT NOT NULL,
    outcome         INTEGER NOT NULL
)
"""


def write_chunks_db(records: list[dict], path: Path) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode = OFF")
    con.execute(SCHEMA)
    con.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                i,
                r["chunk_id"],
                r["doc_id"],
                r["doc_title"],
                r["doc_type"],
                r["section_marker"],
                r["paragraph_index"],
                r["text"],
                r["jurisdiction"],
                r["source_url"],
                r["source_note"],
                r["source_task"],
                r["outcome"],
            )
            for i, r in enumerate(records)
        ],
    )
    con.execute("CREATE INDEX idx_chunk_id ON chunks(chunk_id)")
    con.execute("CREATE INDEX idx_doc_type ON chunks(doc_type)")
    con.execute("CREATE INDEX idx_source_task ON chunks(source_task)")
    con.commit()
    con.execute("VACUUM")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Nyaya search index.")
    parser.add_argument(
        "--source",
        default=str(ROOT.parent / "legal-rag" / "data" / "chroma" / "chroma.sqlite3"),
        help="Path to the source ChromaDB sqlite file.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap chunk count.")
    parser.add_argument(
        "--reuse-vectors",
        action="store_true",
        help=(
            "Skip embedding and keep the existing vectors.npy. Safe only when "
            "chunk text and ordering are unchanged, for example after a metadata "
            "only fix."
        ),
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"Source corpus not found: {source}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading corpus from {source}", flush=True)
    records = read_chroma(source)
    if args.limit:
        records = records[: args.limit]
    n_docs = len({r["doc_id"] for r in records})
    print(f"  {len(records):,} chunks across {n_docs:,} documents", flush=True)

    texts = [r["text"] for r in records]

    if args.reuse_vectors:
        vectors = np.load(OUT_DIR / "vectors.npy", mmap_mode="r")
        if vectors.shape[0] != len(records):
            raise SystemExit(
                f"Existing vectors.npy holds {vectors.shape[0]} rows but the corpus "
                f"has {len(records)} chunks. Rebuild without --reuse-vectors."
            )
        print(f"Reusing vectors.npy {vectors.shape}", flush=True)
    else:
        vectors = embed_all(texts)
        np.save(OUT_DIR / "vectors.npy", vectors)
        print(f"Wrote vectors.npy {vectors.shape}", flush=True)

    matrix, vocab = build_bm25(texts)
    sparse.save_npz(OUT_DIR / "bm25.npz", matrix, compressed=True)
    (OUT_DIR / "bm25_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print("Wrote bm25.npz and bm25_vocab.json", flush=True)

    write_chunks_db(records, OUT_DIR / "chunks.sqlite")
    print("Wrote chunks.sqlite", flush=True)

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_chunks": len(records),
        "n_documents": n_docs,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "bm25": {"k1": BM25_K1, "b": BM25_B, "vocab_size": len(vocab)},
        "doc_types": dict(Counter(r["doc_type"] for r in records)),
        "outcome_tasks": dict(
            Counter(r["source_task"] for r in records if r["source_task"])
        ),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    total_mb = sum(p.stat().st_size for p in OUT_DIR.iterdir()) / 1e6
    print(f"\nIndex complete. Total artifact size: {total_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
