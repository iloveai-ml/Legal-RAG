"""HTTP routes.

A visitor supplied key arrives in the X-Provider-Key header rather than the JSON
body, so it never lands in a request log that records payloads, and it is passed
straight through to the provider call without being stored anywhere. Cloudflare
needs an account id as well, which travels the same way in X-Provider-Account.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile

from ..core import config, documents, embeddings, pipeline, providers
from ..core.store import Scored, get_index
from .schemas import (
    AnswerOut,
    AskRequest,
    CitationOut,
    OutcomeOut,
    OutcomeRequest,
    SearchRequest,
    SimilarCaseOut,
    SourceOut,
    StatusOut,
    UploadOut,
)

log = logging.getLogger("nyaya.api")
router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- rate limit

_hits: dict[str, deque[float]] = defaultdict(deque)
_hits_lock = Lock()


def _rate_limit(request: Request) -> None:
    """A small fixed window limiter so one visitor cannot drain the demo's quota."""
    limit = config.RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return
    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not client:
        client = request.client.host if request.client else "unknown"

    now = time.time()
    with _hits_lock:
        bucket = _hits[client]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"That is more than {limit} requests in a minute. "
                    "Wait a moment and try again."
                ),
            )
        bucket.append(now)
        if len(_hits) > 4096:
            for key in [k for k, v in _hits.items() if not v]:
                _hits.pop(key, None)


def _clean_key(raw: str | None) -> str | None:
    if not config.ALLOW_BYO_KEY:
        return None
    key = (raw or "").strip()
    return key if 12 <= len(key) <= 400 else None


# ---------------------------------------------------------------- rendering

def _sources_out(hits: list[Scored], cited_ids: set[str]) -> list[SourceOut]:
    return [
        SourceOut(
            tag=f"S{i}",
            chunk_id=hit.chunk.chunk_id,
            doc_title=hit.chunk.doc_title,
            doc_type=hit.chunk.doc_type,
            section_marker=hit.chunk.section_marker,
            paragraph_index=hit.chunk.paragraph_index,
            text=hit.chunk.text,
            source_url=hit.chunk.source_url,
            source_note=hit.chunk.source_note,
            origin=hit.chunk.origin,
            score=round(hit.score, 6),
            vector_rank=hit.vector_rank,
            keyword_rank=hit.keyword_rank,
            cited=hit.chunk.chunk_id in cited_ids,
        )
        for i, hit in enumerate(hits, start=1)
    ]


# ---------------------------------------------------------------- endpoints

@router.get("/status", response_model=StatusOut)
def status() -> StatusOut:
    configured = providers.server_configured_providers()
    return StatusOut(
        app_name=config.APP_NAME,
        tagline=config.APP_TAGLINE,
        index=get_index().stats(),
        embedder_ready=embeddings.is_ready(),
        providers=providers.provider_catalogue(),
        active_provider=configured[0] if configured else None,
        has_provider=bool(configured),
        allow_byo_key=config.ALLOW_BYO_KEY,
        uploads_enabled=True,
        max_upload_mb=round(config.MAX_UPLOAD_BYTES / 1e6, 1),
    )


@router.get("/health")
def health() -> dict:
    """Liveness probe for Render. Cheap on purpose."""
    index = get_index()
    return {
        "status": "ok" if index.ready else "degraded",
        "index_ready": index.ready,
        "chunks": index.n_chunks,
    }


@router.post("/ask", response_model=AnswerOut)
def ask(
    payload: AskRequest,
    request: Request,
    x_provider_key: str | None = Header(default=None),
    x_provider_account: str | None = Header(default=None),
) -> AnswerOut:
    _rate_limit(request)

    result = pipeline.answer_question(
        payload.question,
        top_k=payload.top_k,
        doc_types=payload.doc_types or None,
        deep=payload.deep,
        skip_router=payload.skip_router,
        upload_ids=payload.upload_ids,
        preferred_provider=payload.provider,
        byo_key=_clean_key(x_provider_key),
        byo_account=providers.clean_account(x_provider_account),
    )

    cited_ids = {c.chunk_id for c in result.citations if c.verified}
    return AnswerOut(
        question=result.question,
        answer=result.answer,
        key_points=result.key_points,
        citations=[
            CitationOut(
                source_tag=c.source_tag,
                chunk_id=c.chunk_id,
                quote=c.quote,
                support_reason=c.support_reason,
                verified=c.verified,
                quote_match=c.quote_match,
            )
            for c in result.citations
        ],
        sources=_sources_out(result.hits, cited_ids),
        intent=result.route.intent,
        intent_label=result.route.label,
        rewritten_query=result.route.query,
        keywords=result.route.keywords,
        confidence=result.confidence,
        caveats=result.caveats,
        mode=result.mode,
        provider=result.provider,
        model=result.model,
        notice=result.notice,
        elapsed_ms=result.elapsed_ms,
        verified_count=sum(1 for c in result.citations if c.verified),
        unverified_count=sum(1 for c in result.citations if not c.verified),
    )


@router.post("/outcome", response_model=OutcomeOut)
def outcome(
    payload: OutcomeRequest,
    request: Request,
    x_provider_key: str | None = Header(default=None),
    x_provider_account: str | None = Header(default=None),
) -> OutcomeOut:
    _rate_limit(request)

    report = pipeline.predict_outcome(
        payload.case_description,
        task=payload.task,
        max_cases=payload.max_cases,
        deep=payload.deep,
        preferred_provider=payload.provider,
        byo_key=_clean_key(x_provider_key),
        byo_account=providers.clean_account(x_provider_account),
    )
    return OutcomeOut(
        case_description=report.case_description,
        task=report.task,
        task_label=report.task_label,
        similar_cases=[
            SimilarCaseOut(
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                outcome=c.outcome,
                outcome_label=c.outcome_label,
                score=c.score,
                snippet=c.snippet,
                source_note=c.source_note,
            )
            for c in report.similar_cases
        ],
        favorable_count=report.favorable_count,
        unfavorable_count=report.unfavorable_count,
        favorable_label=report.favorable_label,
        unfavorable_label=report.unfavorable_label,
        favorable_pct=report.favorable_pct,
        similar_case_summary=report.similar_case_summary,
        key_factors=report.key_factors,
        assessment=report.assessment,
        confidence=report.confidence,
        caveats=report.caveats,
        mode=report.mode,
        provider=report.provider,
        model=report.model,
        notice=report.notice,
        elapsed_ms=report.elapsed_ms,
    )


@router.post("/search")
def search(payload: SearchRequest, request: Request) -> dict:
    """Retrieval with no language model involved. Free, fast, and useful for checking coverage."""
    _rate_limit(request)

    route = pipeline.Route(query=payload.query)
    started = time.perf_counter()
    hits = pipeline.retrieve(
        route,
        payload.query,
        top_k=payload.top_k,
        doc_types=payload.doc_types or None,
        upload_ids=payload.upload_ids,
    )
    return {
        "query": payload.query,
        "sources": [s.model_dump() for s in _sources_out(hits, set())],
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


@router.post("/upload", response_model=UploadOut)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadOut:
    _rate_limit(request)

    filename = file.filename or "document"
    if not any(filename.lower().endswith(ext) for ext in documents.SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF, DOCX, TXT or MD file.",
        )

    payload = await file.read()
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That file is larger than the {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB limit.",
        )

    try:
        document = documents.ingest(filename, payload)
    except documents.DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Could not process that file: {exc}") from exc

    return UploadOut(
        doc_id=document.doc_id,
        filename=document.filename,
        title=document.title,
        n_chunks=len(document.chunks),
        n_pages=document.n_pages,
        truncated=document.truncated,
    )


@router.delete("/upload/{doc_id}")
def drop_upload(doc_id: str) -> dict:
    return {"removed": documents.upload_store.drop(doc_id)}
