"""
LLM clients — three providers, one priority order:

  1. Siemens (gpt-oss-120b-onprem, OpenAI-compatible)  ← PRIMARY
  2. Groq    (llama-3.3-70b-versatile)                 ← fallback for routing
  3. Gemini  (2.5-flash / 2.5-pro)                     ← fallback for synthesis

Fallback logic:
- On Siemens 401/403 (permanent auth/permission error): skip retries, fall back immediately.
- On Siemens 429/5xx (transient): retry up to 3×, then fall back.
- On Groq/Gemini failure after retries: raise RuntimeError (caught by app.py).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import openai as _openai_module
from openai import OpenAI

from . import config

# ── Siemens — always required (primary) ───────────────────────────────────────
_siemens = OpenAI(
    api_key=config.SIEMENS_API_KEY,
    base_url=config.SIEMENS_BASE_URL,
)

# ── Groq — optional fallback for routing ──────────────────────────────────────
_groq = None
if config.GROQ_API_KEY:
    try:
        from groq import Groq
        _groq = Groq(api_key=config.GROQ_API_KEY)
    except Exception:
        pass

# ── Gemini — optional fallback for synthesis ──────────────────────────────────
_gemini = None
_genai_types = None
if config.GOOGLE_API_KEY:
    try:
        from google import genai as _genai_mod
        from google.genai import types as _genai_types
        _gemini = _genai_mod.Client(api_key=config.GOOGLE_API_KEY)
    except Exception:
        pass


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_permanent_error(exc: Exception) -> bool:
    """True for 401/403 — don't retry, fail fast so fallback fires immediately."""
    return isinstance(exc, (_openai_module.AuthenticationError, _openai_module.PermissionDeniedError))


def _siemens_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    response_format_json: bool = False,
) -> str:
    """Call Siemens API. Raises RuntimeError on all failures (permanent or after retries).

    Qwen models: automatically disables thinking mode so it doesn't consume
    output tokens and interfere with JSON responses.
    """
    model_name = config.SIEMENS_MODEL or "gpt-oss-120b-onprem"
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": min(max_tokens, 4096),
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    # Qwen models support thinking mode; disable it so token budget goes to output
    if "qwen" in model_name.lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = _siemens.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if _is_permanent_error(e):
                # 401/403 — no point retrying
                break
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Siemens call failed: {last_err!r}")


def _gemini_generate_direct(
    prompt: str,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
    thinking_budget: int | None = 0,
) -> str:
    """Gemini-only generation (no Siemens attempt — used as fallback path)."""
    if _gemini is None:
        raise RuntimeError(
            "Siemens unavailable and Gemini not configured. "
            "Add GOOGLE_API_KEY to .env as a fallback."
        )

    cfg_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        cfg_kwargs["response_mime_type"] = response_mime_type
    if response_schema:
        cfg_kwargs["response_schema"] = response_schema
    if thinking_budget is not None and _genai_types is not None:
        cfg_kwargs["thinking_config"] = _genai_types.ThinkingConfig(thinking_budget=thinking_budget)

    assert _genai_types is not None
    cfg = _genai_types.GenerateContentConfig(**cfg_kwargs)

    mn = (model or config.GEMINI_SYNTHESIS_MODEL or "gemini-2.5-flash").strip()
    mn = mn[len("models/"):] if mn.startswith("models/") else mn

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = _gemini.models.generate_content(model=mn, contents=prompt, config=cfg)
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after retries: {last_err!r}")


# ── Public API ────────────────────────────────────────────────────────────────

def groq_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    response_format_json: bool = False,
) -> str:
    """Routing/fast call — Siemens primary, Groq fallback."""
    # 1. Try Siemens
    try:
        return _siemens_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=response_format_json,
        )
    except RuntimeError:
        pass  # fall through to Groq

    # 2. Groq fallback
    if _groq is None:
        raise RuntimeError(
            "Siemens unavailable and GROQ_API_KEY not configured. "
            "Add GROQ_API_KEY to .env as a fallback."
        )
    kwargs: dict[str, Any] = {
        "model": model or config.GROQ_ROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = _groq.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Siemens and Groq both failed. Groq error: {last_err!r}")


def gemini_generate(
    prompt: str,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
    thinking_budget: int | None = 0,
) -> str:
    """Text generation — Siemens primary, Gemini fallback."""
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    # 1. Try Siemens (ignores response_mime_type/schema — gemini_json handles those)
    try:
        return _siemens_chat(messages, temperature=temperature, max_tokens=max_output_tokens)
    except RuntimeError:
        pass

    # 2. Gemini fallback
    return _gemini_generate_direct(
        prompt,
        model=model,
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        thinking_budget=thinking_budget,
    )


def gemini_json(prompt: str, schema: dict, *, model: str | None = None, **kwargs) -> dict:
    """
    Structured JSON generation — Siemens primary, Gemini fallback.

    Siemens path: schema embedded in prompt + response_format=json_object.
    Gemini path:  native response_schema (guaranteed-valid JSON).
    """
    system_instruction = kwargs.get("system_instruction")
    temperature = float(kwargs.get("temperature", 0.1))
    max_output_tokens = int(kwargs.get("max_output_tokens", 4096))

    # Build Siemens messages with schema hint injected
    json_rule = "Return ONLY valid JSON. No markdown fences, no commentary, no extra keys."
    schema_str = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    siemens_prompt = f"{prompt}\n\nRequired JSON schema:\n{schema_str}"

    sys_content = (f"{system_instruction}\n\n{json_rule}") if system_instruction else json_rule
    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": siemens_prompt},
    ]

    # 1. Try Siemens
    try:
        raw = _siemens_chat(
            messages,
            temperature=temperature,
            max_tokens=min(max_output_tokens, 4096),
            response_format_json=True,
        )
        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        return json.loads(raw)
    except RuntimeError:
        pass  # Siemens unavailable — fall through to Gemini
    except json.JSONDecodeError:
        pass  # Siemens returned bad JSON — fall through to Gemini

    # 2. Gemini fallback with native response_schema (guarantees valid JSON)
    gemini_kwargs = {k: v for k, v in kwargs.items() if k not in ("response_mime_type", "response_schema")}
    text = _gemini_generate_direct(
        prompt,
        model=model,
        response_mime_type="application/json",
        response_schema=schema,
        **gemini_kwargs,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON: {text[:200]!r}") from e
