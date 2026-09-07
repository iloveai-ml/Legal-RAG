"""Request and response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core import config


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    doc_types: list[str] = Field(default_factory=list, max_length=4)
    top_k: int = Field(default=config.RETRIEVAL_TOP_K, ge=3, le=20)
    deep: bool = False
    skip_router: bool = False
    upload_ids: list[str] = Field(default_factory=list, max_length=4)
    provider: str | None = None


class OutcomeRequest(BaseModel):
    case_description: str = Field(min_length=20, max_length=config.MAX_QUESTION_CHARS)
    task: Literal["cjpe", "bail"] = "cjpe"
    max_cases: int = Field(default=10, ge=4, le=20)
    deep: bool = False
    provider: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    doc_types: list[str] = Field(default_factory=list, max_length=4)
    top_k: int = Field(default=10, ge=3, le=25)
    upload_ids: list[str] = Field(default_factory=list, max_length=4)


class SourceOut(BaseModel):
    tag: str
    chunk_id: str
    doc_title: str
    doc_type: str
    section_marker: str
    paragraph_index: int
    text: str
    source_url: str
    source_note: str
    origin: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    cited: bool = False


class CitationOut(BaseModel):
    source_tag: str
    chunk_id: str
    quote: str
    support_reason: str
    verified: bool
    quote_match: float


class AnswerOut(BaseModel):
    question: str
    answer: str
    key_points: list[str]
    citations: list[CitationOut]
    sources: list[SourceOut]
    intent: str
    intent_label: str
    rewritten_query: str
    keywords: list[str]
    confidence: str
    caveats: list[str]
    mode: str
    provider: str
    model: str
    notice: str
    elapsed_ms: int
    verified_count: int
    unverified_count: int


class SimilarCaseOut(BaseModel):
    doc_id: str
    doc_title: str
    outcome: int
    outcome_label: str
    score: float
    snippet: str
    source_note: str


class OutcomeOut(BaseModel):
    case_description: str
    task: str
    task_label: str
    similar_cases: list[SimilarCaseOut]
    favorable_count: int
    unfavorable_count: int
    favorable_label: str
    unfavorable_label: str
    favorable_pct: int
    similar_case_summary: str
    key_factors: list[str]
    assessment: str
    confidence: str
    caveats: list[str]
    mode: str
    provider: str
    model: str
    notice: str
    elapsed_ms: int


class UploadOut(BaseModel):
    doc_id: str
    filename: str
    title: str
    n_chunks: int
    n_pages: int
    truncated: bool


class StatusOut(BaseModel):
    app_name: str
    tagline: str
    index: dict[str, Any]
    embedder_ready: bool
    providers: list[dict[str, Any]]
    active_provider: str | None
    has_provider: bool
    allow_byo_key: bool
    uploads_enabled: bool
    max_upload_mb: float
