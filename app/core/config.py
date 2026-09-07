"""
Central configuration for Nyaya.

Everything is read from the environment so the same image runs locally and on
Render. Nothing here is required. If no provider key is set the app still boots
and serves retrieval only results, and visitors can supply their own key from
the browser.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional in production
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if load_dotenv is not None:
    for candidate in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------- app

APP_NAME = _env("APP_NAME", default="Nyaya") or "Nyaya"
APP_TAGLINE = (
    _env("APP_TAGLINE", default="Citation grounded research on Indian law")
    or "Citation grounded research on Indian law"
)
PORT = _env_int("PORT", 8000)

# Allow visitors to paste their own provider key in the browser. The key is
# used for that single request and is never written to disk or logged.
ALLOW_BYO_KEY = _env_bool("ALLOW_BYO_KEY", True)

# ---------------------------------------------------------------- provider keys

GROQ_API_KEY = _env("GROQ_API_KEY")
GEMINI_API_KEY = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
HF_API_KEY = _env("HF_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY")

# Cloudflare Workers AI needs two values, not one: the account id is part of the
# request URL rather than a header, so a token on its own addresses nothing.
CLOUDFLARE_API_KEY = _env("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "CF_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = _env("CLOUDFLARE_ACCOUNT_ID", "CF_ACCOUNT_ID")

# Optional override for OpenAI compatible gateways such as an on premise
# deployment or a proxy. Only applies to the openai provider.
OPENAI_BASE_URL = _env("OPENAI_BASE_URL")

# ---------------------------------------------------------------- models

# Every model name is overridable so a provider rename never needs a code change.
GROQ_FAST_MODEL = _env("GROQ_FAST_MODEL", default="openai/gpt-oss-20b")
GROQ_DEEP_MODEL = _env("GROQ_DEEP_MODEL", default="openai/gpt-oss-120b")

GEMINI_FAST_MODEL = _env("GEMINI_FAST_MODEL", default="gemini-2.5-flash")
# gemini-2.5-pro was the deep default and now returns 404 on a free AI Studio
# key, which made the Deep toggle fail outright. 2.5-flash is verified working,
# so it is the default here and the deep choice stays overridable.
GEMINI_DEEP_MODEL = _env("GEMINI_DEEP_MODEL", default="gemini-2.5-flash")

OPENAI_FAST_MODEL = _env("OPENAI_FAST_MODEL", default="gpt-4o-mini")
OPENAI_DEEP_MODEL = _env("OPENAI_DEEP_MODEL", default="gpt-4o")

HF_FAST_MODEL = _env("HF_FAST_MODEL", default="meta-llama/Llama-3.3-70B-Instruct")
HF_DEEP_MODEL = _env("HF_DEEP_MODEL", default="Qwen/Qwen2.5-72B-Instruct")

# Workers AI model ids are the full namespaced form, taken from Cloudflare's
# own model pages rather than guessed.
CLOUDFLARE_FAST_MODEL = _env(
    "CLOUDFLARE_FAST_MODEL", default="@cf/meta/llama-3.3-70b-instruct-fp8-fast"
)
CLOUDFLARE_DEEP_MODEL = _env("CLOUDFLARE_DEEP_MODEL", default="@cf/openai/gpt-oss-120b")

# Order in which providers are tried when the caller does not name one.
PROVIDER_PRIORITY = [
    p.strip().lower()
    for p in (
        _env("PROVIDER_PRIORITY", default="groq,gemini,cloudflare,openai,huggingface") or ""
    ).split(",")
    if p.strip()
]

# ---------------------------------------------------------------- retrieval

INDEX_DIR = Path(
    _env("NYAYA_INDEX_DIR", default=str(PROJECT_ROOT / "data" / "index"))
    or str(PROJECT_ROOT / "data" / "index")
)

EMBED_MODEL = _env(
    "NYAYA_EMBED_MODEL", default="sentence-transformers/all-MiniLM-L6-v2"
) or "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 8)
RRF_K = _env_int("RRF_K", 60)
CANDIDATE_MULTIPLIER = 3

# How strongly the router's document type guess nudges matching passages up.
# Kept mild on purpose: it should break ties, never overrule relevance.
DOC_TYPE_BOOST = 1.15

# ---------------------------------------------------------------- uploads

MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024)
MAX_UPLOAD_CHUNKS = _env_int("MAX_UPLOAD_CHUNKS", 400)
UPLOAD_SESSION_TTL_SECONDS = _env_int("UPLOAD_SESSION_TTL_SECONDS", 3600)
MAX_UPLOAD_SESSIONS = _env_int("MAX_UPLOAD_SESSIONS", 24)

# ---------------------------------------------------------------- limits

REQUEST_TIMEOUT_SECONDS = _env_int("REQUEST_TIMEOUT_SECONDS", 90)
MAX_QUESTION_CHARS = _env_int("MAX_QUESTION_CHARS", 4000)
RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 20)
