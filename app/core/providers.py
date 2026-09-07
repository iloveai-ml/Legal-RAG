"""
One unified LLM layer over Groq, Gemini, OpenAI and Hugging Face.

All four expose an OpenAI compatible chat completions endpoint, so a single
client class covers every provider and the only thing that changes is the base
URL, the key and the model name. That keeps the dependency list to one SDK and
the fallback logic to one code path.

Design rules that matter for deployment:

  * The app never requires a key. Retrieval runs on local embeddings, so with
    zero keys configured the app still answers with ranked cited passages.
  * Any single key is enough. Whichever providers are configured are tried in
    priority order, and a failing provider falls through to the next one.
  * A visitor can supply their own key per request. That key lives only for the
    duration of the request. It is never logged, cached or written to disk.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from openai import OpenAI

from . import config

log = logging.getLogger("nyaya.providers")

# Provider ids used across the API and the UI
GROQ = "groq"
GEMINI = "gemini"
CLOUDFLARE = "cloudflare"
OPENAI = "openai"
HUGGINGFACE = "huggingface"

# Cloudflare is the one provider whose account id sits in the URL path, so its
# base URL has to be built per request rather than being a constant.
_CF_BASE_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
_SAFE_ACCOUNT = re.compile(r"^[0-9a-zA-Z]{8,64}$")


def clean_account(value: str | None) -> str:
    """An account id safe to interpolate into a URL path, or empty."""
    candidate = (value or "").strip()
    return candidate if _SAFE_ACCOUNT.match(candidate) else ""


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    base_url: str
    fast_model: str
    deep_model: str
    key_hint: str
    console_url: str
    supports_json_mode: bool = True
    # True when a token alone is not enough, because the account id forms part
    # of the request URL. base_url is then a template holding {account_id}.
    needs_account: bool = False


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    GROQ: ProviderSpec(
        id=GROQ,
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        fast_model=config.GROQ_FAST_MODEL or "llama-3.3-70b-versatile",
        deep_model=config.GROQ_DEEP_MODEL or "openai/gpt-oss-120b",
        key_hint="gsk_...",
        console_url="https://console.groq.com/keys",
    ),
    GEMINI: ProviderSpec(
        id=GEMINI,
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        fast_model=config.GEMINI_FAST_MODEL or "gemini-2.5-flash",
        deep_model=config.GEMINI_DEEP_MODEL or "gemini-2.5-pro",
        key_hint="AIza...",
        console_url="https://aistudio.google.com/apikey",
    ),
    CLOUDFLARE: ProviderSpec(
        id=CLOUDFLARE,
        label="Cloudflare Workers AI",
        base_url=_CF_BASE_TEMPLATE,
        fast_model=config.CLOUDFLARE_FAST_MODEL
        or "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        deep_model=config.CLOUDFLARE_DEEP_MODEL or "@cf/openai/gpt-oss-120b",
        key_hint="Cloudflare API token, plus an account id",
        console_url="https://dash.cloudflare.com/profile/api-tokens",
        needs_account=True,
    ),
    OPENAI: ProviderSpec(
        id=OPENAI,
        label="OpenAI",
        base_url=config.OPENAI_BASE_URL or "https://api.openai.com/v1",
        fast_model=config.OPENAI_FAST_MODEL or "gpt-4o-mini",
        deep_model=config.OPENAI_DEEP_MODEL or "gpt-4o",
        key_hint="sk-...",
        console_url="https://platform.openai.com/api-keys",
    ),
    HUGGINGFACE: ProviderSpec(
        id=HUGGINGFACE,
        label="Hugging Face",
        base_url="https://router.huggingface.co/v1",
        fast_model=config.HF_FAST_MODEL or "meta-llama/Llama-3.3-70B-Instruct",
        deep_model=config.HF_DEEP_MODEL or "Qwen/Qwen2.5-72B-Instruct",
        key_hint="hf_...",
        console_url="https://huggingface.co/settings/tokens",
        # The router fronts many model servers and not all honour response_format
        supports_json_mode=False,
    ),
}

_SERVER_KEYS: dict[str, str | None] = {
    GROQ: config.GROQ_API_KEY,
    GEMINI: config.GEMINI_API_KEY,
    CLOUDFLARE: config.CLOUDFLARE_API_KEY,
    OPENAI: config.OPENAI_API_KEY,
    HUGGINGFACE: config.HF_API_KEY,
}

# The second half of a Cloudflare credential, from the environment.
_SERVER_ACCOUNTS: dict[str, str | None] = {
    CLOUDFLARE: clean_account(config.CLOUDFLARE_ACCOUNT_ID),
}

_KEY_PREFIXES: list[tuple[str, str]] = [
    ("gsk_", GROQ),
    ("hf_", HUGGINGFACE),
    ("sk-proj-", OPENAI),
    ("sk-", OPENAI),
    ("AIza", GEMINI),
    # Cloudflare user API tokens. Verified against a live token.
    ("cfut_", CLOUDFLARE),
]


class NoProviderError(RuntimeError):
    """Raised when no provider is configured or every configured one failed."""


@dataclass
class ProviderCall:
    """What actually happened during a generation, surfaced to the UI."""

    provider: str = ""
    model: str = ""
    attempts: list[str] = field(default_factory=list)
    byo_key: bool = False


# ---------------------------------------------------------------- key handling

def detect_provider_from_key(key: str) -> str | None:
    """Infer which provider a pasted key belongs to from its prefix."""
    key = (key or "").strip()
    for prefix, provider in _KEY_PREFIXES:
        if key.startswith(prefix):
            return provider
    return None


def server_configured_providers() -> list[str]:
    """Provider ids the server can actually call, in priority order.

    A provider needing an account id counts only when both halves are set,
    because a token with no account cannot address anything.
    """
    ordered = [p for p in config.PROVIDER_PRIORITY if p in PROVIDER_SPECS]
    for pid in PROVIDER_SPECS:
        if pid not in ordered:
            ordered.append(pid)
    usable = []
    for pid in ordered:
        if not _SERVER_KEYS.get(pid):
            continue
        if PROVIDER_SPECS[pid].needs_account and not _SERVER_ACCOUNTS.get(pid):
            continue
        usable.append(pid)
    return usable


def provider_catalogue() -> list[dict[str, Any]]:
    """Public description of every provider, for the settings panel."""
    configured = set(server_configured_providers())
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "configured": spec.id in configured,
            "fast_model": spec.fast_model,
            "deep_model": spec.deep_model,
            "key_hint": spec.key_hint,
            "console_url": spec.console_url,
        }
        for spec in PROVIDER_SPECS.values()
    ]


# ---------------------------------------------------------------- clients

_client_cache: dict[tuple[str, str], OpenAI] = {}
_client_lock = threading.Lock()


def resolve_base_url(spec: ProviderSpec, account_id: str | None = None) -> str:
    """The base URL for one call. Empty when a needed account id is missing."""
    if not spec.needs_account:
        return spec.base_url
    account = clean_account(account_id) or (_SERVER_ACCOUNTS.get(spec.id) or "")
    return spec.base_url.format(account_id=account) if account else ""


def _client(spec: ProviderSpec, api_key: str, cacheable: bool, base_url: str) -> OpenAI:
    """Build (and for server keys, reuse) an OpenAI compatible client."""
    if cacheable:
        # The account id can change the URL, so it is part of the cache key.
        cache_key = (spec.id, base_url)
        with _client_lock:
            cached = _client_cache.get(cache_key)
            if cached is not None:
                return cached
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(config.REQUEST_TIMEOUT_SECONDS),
                max_retries=0,
            )
            _client_cache[cache_key] = client
            return client
    # Visitor supplied keys get a throwaway client that is never cached
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(config.REQUEST_TIMEOUT_SECONDS),
        max_retries=0,
    )


def _is_permanent(exc: Exception) -> bool:
    """Auth and permission failures should fall through instead of retrying."""
    status = getattr(exc, "status_code", None)
    return status in (400, 401, 403, 404)


def _redact(message: str) -> str:
    """Strip anything that looks like a key out of an error string."""
    return re.sub(
        r"\b(gsk_|hf_|sk-|AIza|cfut_)[A-Za-z0-9_\-]{6,}", r"\1<redacted>", message
    )


# ---------------------------------------------------------------- resolution

def _resolve_chain(
    *,
    preferred: str | None,
    byo_key: str | None,
    byo_account: str | None = None,
) -> list[tuple[ProviderSpec, str, bool, str]]:
    """
    Work out which (spec, key, cacheable) combinations to try, in order.

    A visitor key always goes first. After that come the server configured
    providers in priority order, with any explicitly preferred provider hoisted
    to the front.
    """
    chain: list[tuple[ProviderSpec, str, bool, str]] = []

    if byo_key:
        pid = preferred if preferred in PROVIDER_SPECS else detect_provider_from_key(byo_key)
        if pid in PROVIDER_SPECS:
            spec = PROVIDER_SPECS[pid]
            base = resolve_base_url(spec, byo_account)
            # A provider that needs an account id and has none cannot be
            # addressed at all, so it is left out rather than failing later.
            if base:
                chain.append((spec, byo_key, False, base))

    server = server_configured_providers()
    if preferred in PROVIDER_SPECS and preferred in server:
        server = [preferred] + [p for p in server if p != preferred]

    for pid in server:
        key = _SERVER_KEYS.get(pid)
        if not key:
            continue
        spec = PROVIDER_SPECS[pid]
        base = resolve_base_url(spec, byo_account)
        if base:
            chain.append((spec, key, True, base))
    return chain


def has_any_provider(
    byo_key: str | None = None, byo_account: str | None = None
) -> bool:
    return bool(
        _resolve_chain(preferred=None, byo_key=byo_key, byo_account=byo_account)
    )


# ---------------------------------------------------------------- generation

def chat(
    messages: list[dict[str, str]],
    *,
    deep: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    json_mode: bool = False,
    preferred_provider: str | None = None,
    byo_key: str | None = None,
    byo_account: str | None = None,
    call: ProviderCall | None = None,
) -> str:
    """
    Run a chat completion against the first provider that answers.

    Raises NoProviderError when nothing is configured or every provider failed.
    """
    chain = _resolve_chain(
        preferred=preferred_provider, byo_key=byo_key, byo_account=byo_account
    )
    if not chain:
        raise NoProviderError(
            "No language model provider is configured. Add a Groq, Gemini, "
            "Cloudflare, OpenAI or Hugging Face key, or paste one in the "
            "settings panel. A Cloudflare token also needs its account id."
        )

    record = call if call is not None else ProviderCall()
    errors: list[str] = []

    for spec, api_key, cacheable, base_url in chain:
        model = spec.deep_model if deep else spec.fast_model
        base_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Reasoning models sometimes emit a preamble that trips a provider's
        # strict JSON validator and comes back as a 400. When that happens the
        # request is worth one more try with the constraint removed, because
        # chat_json can recover an object from ordinary prose anyway.
        use_json_mode = json_mode and spec.supports_json_mode
        attempts = 3 if use_json_mode else 2

        for attempt in range(attempts):
            kwargs = dict(base_kwargs)
            drop_json_mode = use_json_mode and attempt > 0
            if use_json_mode and not drop_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                client = _client(spec, api_key, cacheable, base_url)
                response = client.chat.completions.create(**kwargs)
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise RuntimeError("Provider returned an empty response.")
                record.provider = spec.id
                record.model = model
                record.byo_key = not cacheable
                return content
            except Exception as exc:  # noqa: BLE001 - fall through to next provider
                detail = _redact(f"{type(exc).__name__}: {exc}")[:240]
                last_attempt = attempt == attempts - 1
                # An auth or missing model error will not improve on a retry,
                # but a JSON validation 400 might once the constraint is gone.
                fatal = _is_permanent(exc) and not (
                    use_json_mode and attempt == 0 and getattr(exc, "status_code", None) == 400
                )
                if fatal or last_attempt:
                    errors.append(f"{spec.label}: {detail}")
                    record.attempts.append(f"{spec.label} failed")
                    log.warning("Provider %s failed: %s", spec.id, detail)
                    break
                if not drop_json_mode:
                    time.sleep(1.0 + attempt)

    raise NoProviderError("Every configured provider failed. " + " | ".join(errors))


# ---------------------------------------------------------------- json helpers

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json(raw: str) -> dict:
    """Parse a JSON object out of a model response, tolerating stray prose."""
    text = _FENCE_RE.sub("", raw.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced brace span
    start = text.find("{")
    if start == -1:
        raise ValueError("Model response contained no JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                break
    raise ValueError("Model response contained no complete JSON object.")


def chat_json(
    *,
    system: str,
    prompt: str,
    schema: dict,
    deep: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    preferred_provider: str | None = None,
    byo_key: str | None = None,
    byo_account: str | None = None,
    call: ProviderCall | None = None,
) -> dict:
    """Structured generation that works the same way on every provider."""
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    system_message = (
        f"{system}\n\n"
        "Respond with a single valid JSON object and nothing else. "
        "No markdown fences, no commentary, no keys outside the schema.\n\n"
        f"JSON schema:\n{schema_text}"
    )
    raw = chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        deep=deep,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
        preferred_provider=preferred_provider,
        byo_key=byo_key,
        byo_account=byo_account,
        call=call,
    )
    return _extract_json(raw)


# Models reach for typographic characters that a prompt cannot reliably talk
# them out of: em and en dashes, narrow no break spaces, non breaking hyphens,
# smart quotes. Normalising on the way out guarantees the house style instead of
# hoping every provider honoured it, and keeps quote matching against source
# text from failing on an invisible character difference.
_PUNCT_MAP = {
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
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_MAP))
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")


def normalise_text(value: str) -> str:
    """Apply house punctuation to anything a model wrote before it is displayed."""
    if not value:
        return ""
    out = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], value)
    out = _SPACE_RUN_RE.sub(" ", out)
    # A dash that ended up against a word boundary reads worse than one with air
    out = re.sub(r"\s+-\s+", " - ", out)
    return out.strip()


def coerce_str_list(value: Any, limit: int = 12) -> list[str]:
    """Models occasionally return a string where the schema asks for a list."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(normalise_text(item))
        elif isinstance(item, dict):
            for key in ("text", "factor", "value", "description"):
                if isinstance(item.get(key), str) and item[key].strip():
                    out.append(normalise_text(item[key]))
                    break
        if len(out) >= limit:
            break
    return [s for s in out if s]
