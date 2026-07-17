"""
Central configuration. Loads env vars from the project root .env or the LegalTech parent .env.

Provider priority (set in llm.py):
  1. Siemens  — primary LLM (SIEMENS_API_KEY required)
  2. Groq     — fallback for routing calls (GROQ_API_KEY optional)
  3. Gemini   — fallback for synthesis calls (GOOGLE_API_KEY optional)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARENT_DIR = PROJECT_ROOT.parent

for candidate in (PROJECT_ROOT / ".env", PARENT_DIR / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)


def _require(name: str, *aliases: str) -> str:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError(
        f"Missing required env var. Set one of: {[name, *aliases]} in the .env file."
    )


def _optional(name: str, *aliases: str, default: str | None = None) -> str | None:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value:
            return value
    return default


# ── Primary LLM: Siemens (required) ──────────────────────────────────────────
SIEMENS_API_KEY = _require("SIEMENS_API_KEY", "OPENAI_API_KEY")
SIEMENS_BASE_URL = _optional("SIEMENS_BASE_URL", default="https://api.siemens.com/llm/v1")
SIEMENS_MODEL = _optional("SIEMENS_MODEL", default="gpt-oss-120b-onprem")

# ── Fallback LLMs (optional — app degrades gracefully without them) ───────────
GOOGLE_API_KEY = _optional("GOOGLE_API_KEY", "GEMINI_API_KEY")
GROQ_API_KEY = _optional("GROQ_API_KEY")

HF_TOKEN = _optional("HF_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACE_TOKEN")
VOYAGE_API_KEY = _optional("VOYAGE_API_KEY")

# ── Model names ───────────────────────────────────────────────────────────────
GEMINI_SYNTHESIS_MODEL = _optional("GEMINI_SYNTHESIS_MODEL", default="gemini-2.5-flash")
GEMINI_HEAVY_MODEL = _optional("GEMINI_HEAVY_MODEL", default="gemini-2.5-pro")
GEMINI_EMBEDDING_MODEL = _optional("GEMINI_EMBEDDING_MODEL", default="gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(_optional("GEMINI_EMBEDDING_DIM", default="768") or "768")
GROQ_ROUTER_MODEL = _optional("GROQ_ROUTER_MODEL", default="llama-3.3-70b-versatile")

# ── Embedding backend (set LEGAL_RAG_EMBEDDER in .env) ───────────────────────
#   local   — sentence-transformers, fully offline (default — no API cost)
#   gemini  — Gemini embedding-001 (requires GOOGLE_API_KEY)
#   voyage  — voyage-law-2 (requires VOYAGE_API_KEY)
#   hf_api  — HuggingFace Inference API (optional HF_TOKEN)
LEGAL_RAG_VOYAGE_MODEL = _optional("LEGAL_RAG_VOYAGE_MODEL", default="voyage-law-2")
LEGAL_RAG_HF_API_MODEL = _optional(
    "LEGAL_RAG_HF_API_MODEL", default="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
_chroma_raw = _optional("LEGAL_RAG_CHROMA_DIR") or str(DATA_DIR / "chroma")
_chroma_path = Path(_chroma_raw)
CHROMA_DIR = _chroma_path if _chroma_path.is_absolute() else PROJECT_ROOT / _chroma_path
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

for d in (DATA_DIR, RAW_DIR, CHROMA_DIR, EVAL_DIR, EVAL_RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHROMA_COLLECTION = _optional("LEGAL_RAG_CHROMA_COLLECTION") or "indian_legal_corpus"
EMBEDDING_DIM = 768

CHUNK_TOKENS_TARGET = 350
CHUNK_OVERLAP_TOKENS = 60

RETRIEVAL_TOP_K = 8
RERANK_KEEP = 5
