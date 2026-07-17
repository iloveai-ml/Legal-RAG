"""
Rebuild the ChromaDB index using Gemini embeddings (768-dim).

Reads all existing chunks (text + metadata) from the current collection,
resets the collection, then re-embeds everything with Gemini and re-inserts.
Run once after switching LEGAL_RAG_EMBEDDER=gemini in .env.

Usage:
    python scripts/rebuild_gemini.py
"""
from __future__ import annotations
import os
import sys
import time

# Must be set BEFORE importing core modules so the embedder picks up gemini
os.environ["LEGAL_RAG_EMBEDDER"] = "gemini"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import index, config
from core.embeddings import embed_documents

print("=" * 60)
print("Legal-RAG Index Rebuild — Gemini embeddings (768-dim)")
print("=" * 60)
print(f"ChromaDB dir:     {config.CHROMA_DIR}")
print(f"Collection:       {config.CHROMA_COLLECTION}")
print(f"Gemini embed model: {config.GEMINI_EMBEDDING_MODEL}")
print()

# Step 1: Read all existing chunks
print("Step 1/3: Reading existing chunks from ChromaDB...")
t0 = time.time()
chunks = index.all_chunks()
print(f"  {len(chunks):,} chunks loaded in {time.time()-t0:.1f}s")

if not chunks:
    print("ERROR: No chunks found. Nothing to rebuild.")
    sys.exit(1)

from collections import Counter
types = Counter(c.doc_type for c in chunks)
tasks = Counter(c.extra.get("source_task", "none") for c in chunks if c.extra)
print(f"  By doc_type:    {dict(types)}")
print(f"  By source_task: {dict(tasks)}")
print()

# Step 2: Reset collection (delete + recreate with new dimensions)
print("Step 2/3: Resetting collection for 768-dim Gemini vectors...")
index.reset_index()
print("  Collection reset OK")
print()

# Step 3: Re-embed and re-insert in batches
BATCH = 50  # Gemini batch limit
total = len(chunks)
n_batches = (total + BATCH - 1) // BATCH

print(f"Step 3/3: Re-embedding {total:,} chunks in {n_batches} batches of {BATCH}...")
print("  (This takes ~8-12 minutes. Do not close this window.)")
print()

from core.citation import Chunk
import json

coll = index.get_collection()
inserted = 0
errors = 0
t_start = time.time()

for i in range(0, total, BATCH):
    batch = chunks[i : i + BATCH]
    batch_num = i // BATCH + 1

    texts = [c.text for c in batch]
    ids = [c.chunk_id for c in batch]
    metas = []
    for c in batch:
        md = {
            "doc_id": c.doc_id,
            "doc_title": c.doc_title,
            "doc_type": c.doc_type,
            "section_marker": c.section_marker,
            "paragraph_index": c.paragraph_index,
            "jurisdiction": c.jurisdiction,
            "source_url": c.source_url,
            "source_note": c.source_note,
        }
        if c.extra:
            md["extra_json"] = json.dumps(c.extra, ensure_ascii=False)
            if "source_task" in c.extra:
                md["source_task"] = c.extra["source_task"]
            if "outcome" in c.extra and c.extra["outcome"] != -1:
                md["outcome"] = int(c.extra["outcome"])
        metas.append(md)

    last_err = None
    for attempt in range(3):
        try:
            vecs = embed_documents(texts)
            coll.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)
            inserted += len(batch)
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    else:
        print(f"  WARN: Batch {batch_num} failed after retries: {last_err!r}")
        errors += len(batch)

    elapsed = time.time() - t_start
    rate = inserted / elapsed if elapsed > 0 else 0
    remaining = (total - inserted - errors)
    eta = remaining / rate if rate > 0 else 0
    pct = 100 * (inserted + errors) / total

    print(
        f"  [{pct:5.1f}%] {inserted:,}/{total:,} inserted | "
        f"errors: {errors} | {rate:.0f} chunks/s | ETA {eta/60:.1f}min",
        end="\r",
    )

print()
print()
print(f"Done. {inserted:,} chunks inserted, {errors} errors.")
print(f"Total time: {(time.time()-t_start)/60:.1f} minutes")
print()
print("Next step: set LEGAL_RAG_EMBEDDER=gemini in .env and restart the app.")
