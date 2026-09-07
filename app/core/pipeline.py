"""
The question answering pipeline.

    question -> router -> hybrid retrieval -> synthesis -> citation verification

The last step is what makes the output trustworthy. The model is asked to attach
the chunk_id of the passage behind every claim, and each returned citation is
checked against the passages that were actually retrieved. A citation pointing
at anything else is marked unverified rather than quietly rendered, so a
fabricated source is visible in the interface instead of hidden inside prose.

Every stage degrades rather than fails. No provider key means retrieval only
results. A router failure means the raw question is used. A synthesis failure
still returns the retrieved passages.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Sequence

from . import config, embeddings, prompts, providers
from .documents import upload_store
from .store import Scored, get_index

log = logging.getLogger("nyaya.pipeline")

VALID_INTENTS = {
    "statute_lookup",
    "case_research",
    "legal_concept",
    "procedure",
    "document_review",
    "general_legal_qa",
    "out_of_scope",
}
VALID_DOC_TYPES = {"constitution", "statute", "rule", "case"}

INTENT_LABELS = {
    "statute_lookup": "Statute lookup",
    "case_research": "Case research",
    "legal_concept": "Concept explanation",
    "procedure": "Procedure",
    "document_review": "Document review",
    "general_legal_qa": "General legal question",
    "out_of_scope": "Outside the corpus",
}


@dataclass
class Route:
    intent: str = "general_legal_qa"
    doc_types: list[str] = field(default_factory=list)
    query: str = ""
    keywords: list[str] = field(default_factory=list)
    used_model: bool = False

    @property
    def label(self) -> str:
        return INTENT_LABELS.get(self.intent, "General legal question")


@dataclass
class VerifiedCitation:
    source_tag: str
    chunk_id: str
    quote: str
    support_reason: str
    verified: bool
    quote_match: float


@dataclass
class Answer:
    question: str
    answer: str
    key_points: list[str]
    citations: list[VerifiedCitation]
    hits: list[Scored]
    route: Route
    confidence: str
    caveats: list[str]
    mode: str                 # "synthesised" or "retrieval_only"
    provider: str = ""
    model: str = ""
    notice: str = ""
    elapsed_ms: int = 0


# ---------------------------------------------------------------- routing

def route_question(
    question: str,
    *,
    preferred_provider: str | None,
    byo_key: str | None,
    has_uploads: bool,
    byo_account: str | None = None,
) -> Route:
    """Classify and rewrite the query. Falls back to the raw question on failure."""
    fallback = Route(query=question.strip())
    if not providers.has_any_provider(byo_key, byo_account):
        return fallback

    try:
        data = providers.chat_json(
            system=prompts.ROUTER_SYSTEM,
            prompt=f"Q: {question.strip()}\nA:",
            schema=prompts.ROUTER_SCHEMA,
            temperature=0.0,
            max_tokens=400,
            preferred_provider=preferred_provider,
            byo_key=byo_key,
            byo_account=byo_account,
        )
    except Exception as exc:  # noqa: BLE001 - routing is an optimisation
        log.info("Router unavailable, using the raw question: %s", exc)
        return fallback

    intent = str(data.get("intent") or "").strip()
    if intent not in VALID_INTENTS:
        intent = "general_legal_qa"

    doc_types = [
        t.lower()
        for t in providers.coerce_str_list(data.get("doc_type_filter"), limit=4)
        if t.lower() in VALID_DOC_TYPES
    ]
    rewritten = str(data.get("rewritten_query") or "").strip() or question.strip()

    # An uploaded document makes a corpus doc type filter actively harmful,
    # because the upload is not any of those types
    if has_uploads:
        doc_types = []

    return Route(
        intent=intent,
        doc_types=doc_types,
        query=rewritten,
        keywords=providers.coerce_str_list(data.get("keywords"), limit=8),
        used_model=True,
    )


# ---------------------------------------------------------------- retrieval

def retrieve(
    route: Route,
    original_question: str,
    *,
    top_k: int,
    doc_types: Sequence[str] | None,
    upload_ids: list[str],
    boost_doc_types: Sequence[str] | None = None,
) -> list[Scored]:
    """Search the corpus, plus any uploaded documents, and merge the results."""
    query = route.query or original_question
    query_vector = embeddings.embed_query(query)

    index = get_index()
    corpus_hits: list[Scored] = []
    if index.ready:
        corpus_hits = index.search(
            query,
            query_vector,
            top_k=top_k,
            doc_types=list(doc_types) if doc_types else None,
            boost_doc_types=list(boost_doc_types) if boost_doc_types else None,
        )

    documents = upload_store.get_many(upload_ids)
    if not documents:
        return corpus_hits

    # Uploaded passages compete on their own merit but are guaranteed a share of
    # the window, so a question about the user's own document is never crowded
    # out by a large corpus.
    reserved = max(2, top_k // 2)
    upload_hits: list[Scored] = []
    for document in documents:
        upload_hits.extend(document.search(query, query_vector, reserved))
    upload_hits.sort(key=lambda h: -h.score)
    upload_hits = upload_hits[:reserved]

    merged = upload_hits + corpus_hits[: max(top_k - len(upload_hits), 0)]
    merged.sort(key=lambda h: -h.score)
    return merged[:top_k]


# ---------------------------------------------------------------- verification

def _quote_match(quote: str, source_text: str) -> float:
    """
    How much of the model's quote actually appears in the cited passage.

    An exact substring is the common case. Otherwise fall back to a similarity
    ratio against the best aligned window, which catches light paraphrase and
    whitespace differences without accepting an invented quote.
    """
    quote = " ".join(quote.split()).strip()
    source = " ".join(source_text.split())
    if not quote:
        return 0.0
    if quote.lower() in source.lower():
        return 1.0
    matcher = SequenceMatcher(None, quote.lower(), source.lower())
    block = matcher.find_longest_match(0, len(quote), 0, len(source))
    return block.size / len(quote) if quote else 0.0


_TAG_RE = re.compile(r"\[S\d+\]")


def ensure_inline_tags(
    answer: str, citations: list[VerifiedCitation], hits: list[Scored]
) -> str:
    """
    Guarantee the answer carries at least one clickable source tag.

    Most models follow the instruction to tag each sentence, but not every model
    does it every time. Rather than guess which sentence a citation belongs to,
    which would attribute a source to a claim it may not support, append an
    explicit trailing line naming the passages the model said it relied on. It
    is accurate, it is visibly a list rather than a per sentence attribution,
    and it keeps the citations reachable from the answer.
    """
    if not answer or _TAG_RE.search(answer):
        return answer

    tag_by_chunk = {hit.chunk.chunk_id: f"S{i}" for i, hit in enumerate(hits, start=1)}
    tags: list[str] = []
    for citation in citations:
        if not citation.verified:
            continue
        tag = tag_by_chunk.get(citation.chunk_id)
        if tag and tag not in tags:
            tags.append(tag)

    if not tags:
        return answer

    tags.sort(key=lambda t: int(t[1:]))
    joined = "".join(f"[{t}]" for t in tags)
    return f"{answer}\n\nDrawn from {joined}."


def verify_citations(raw_citations: Any, hits: list[Scored]) -> list[VerifiedCitation]:
    """
    Check each citation against the passages that were actually retrieved.

    A citation counts as verified only when its chunk_id names a retrieved
    passage and its quote genuinely comes from that passage.
    """
    by_id = {hit.chunk.chunk_id: hit.chunk for hit in hits}
    by_tag = {f"S{i}": hit.chunk for i, hit in enumerate(hits, start=1)}

    out: list[VerifiedCitation] = []
    if not isinstance(raw_citations, list):
        return out

    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        source_tag = str(item.get("source_tag") or "").strip()
        # Normalise the quote the same way the corpus text was normalised at
        # build time, so a stray em dash cannot fail an otherwise exact match
        quote = providers.normalise_text(str(item.get("quote") or ""))

        chunk = by_id.get(chunk_id)
        if chunk is None:
            # Models sometimes return the [S#] tag in place of the chunk_id
            chunk = by_tag.get(source_tag.strip("[]"))
            if chunk is not None:
                chunk_id = chunk.chunk_id

        match = _quote_match(quote, chunk.text) if chunk else 0.0
        out.append(
            VerifiedCitation(
                source_tag=source_tag or (f"S{len(out) + 1}"),
                chunk_id=chunk_id,
                quote=quote,
                support_reason=providers.normalise_text(str(item.get("support_reason") or "")),
                verified=chunk is not None and match >= 0.6,
                quote_match=round(match, 3),
            )
        )
    return out


# ---------------------------------------------------------------- answering

_NO_PROVIDER_NOTICE = (
    "No language model is connected, so this is a retrieval only result. "
    "The passages below are the highest ranking matches in the corpus for your "
    "question. Add a provider key in Settings to get a written answer with "
    "verified citations."
)


def _retrieval_only(
    question: str,
    hits: list[Scored],
    route: Route,
    notice: str,
    started: float,
) -> Answer:
    if hits:
        summary = (
            f"Found {len(hits)} passages in the corpus that match this question. "
            "They are ranked by hybrid semantic and keyword relevance and shown in full below."
        )
    else:
        summary = (
            "Nothing in the indexed corpus matches this question closely enough to show. "
            "Try different wording, or check the corpus coverage in the sidebar."
        )
    return Answer(
        question=question,
        answer=summary,
        key_points=[],
        citations=[],
        hits=hits,
        route=route,
        confidence="low",
        caveats=[],
        mode="retrieval_only",
        notice=notice,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    doc_types: Sequence[str] | None = None,
    deep: bool = False,
    skip_router: bool = False,
    upload_ids: list[str] | None = None,
    preferred_provider: str | None = None,
    byo_key: str | None = None,
    byo_account: str | None = None,
) -> Answer:
    started = time.perf_counter()
    question = question.strip()
    upload_ids = upload_ids or []
    top_k = top_k or config.RETRIEVAL_TOP_K
    has_provider = providers.has_any_provider(byo_key, byo_account)

    if skip_router or not has_provider:
        route = Route(query=question)
    else:
        route = route_question(
            question,
            preferred_provider=preferred_provider,
            byo_key=byo_key,
            byo_account=byo_account,
            has_uploads=bool(upload_ids),
        )

    if route.intent == "out_of_scope":
        return Answer(
            question=question,
            answer=(
                "This does not look like a question about Indian law. "
                "Ask about a statute, a judgment, a doctrine or a procedure, "
                "or upload a document and ask about that."
            ),
            key_points=[],
            citations=[],
            hits=[],
            route=route,
            confidence="high",
            caveats=["No retrieval was run for an out of scope question."],
            mode="synthesised",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # Only a filter the user set explicitly is allowed to remove passages. The
    # router's guess is applied as a preference instead, because a wrong guess
    # would otherwise hide the one source that answers the question. Asking
    # whether privacy is a fundamental right, for example, reads as a concept
    # question and gets filtered to statutes, which drops the very judgment
    # that decided it.
    try:
        hits = retrieve(
            route,
            question,
            top_k=top_k,
            doc_types=list(doc_types) if doc_types else None,
            boost_doc_types=None if doc_types else route.doc_types,
            upload_ids=upload_ids,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Retrieval failed")
        return Answer(
            question=question,
            answer=(
                "Retrieval failed, so there are no passages to answer from. "
                f"The search index reported: {exc}"
            ),
            key_points=[],
            citations=[],
            hits=[],
            route=route,
            confidence="low",
            caveats=["The search index is unavailable."],
            mode="retrieval_only",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    if not has_provider:
        return _retrieval_only(question, hits, route, _NO_PROVIDER_NOTICE, started)

    if not hits:
        return Answer(
            question=question,
            answer=(
                "Nothing in the indexed corpus addresses this question, so there is "
                "no grounded answer to give. Rather than guess from memory, the app "
                "stops here. Try rephrasing the question, clearing any document type "
                "filter, or uploading the document you have in mind."
            ),
            key_points=[],
            citations=[],
            hits=[],
            route=route,
            confidence="low",
            caveats=["Retrieval returned no passages."],
            mode="synthesised",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    call = providers.ProviderCall()
    prompt = prompts.SYNTH_PROMPT.format(
        question=question, sources=prompts.render_sources(hits)
    )
    try:
        data = providers.chat_json(
            system=prompts.SYNTH_SYSTEM,
            prompt=prompt,
            schema=prompts.SYNTH_SCHEMA,
            deep=deep,
            temperature=0.15,
            max_tokens=3000,
            preferred_provider=preferred_provider,
            byo_key=byo_key,
            byo_account=byo_account,
            call=call,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Synthesis failed: %s", exc)
        notice = (
            "The language model could not be reached, so this is a retrieval only "
            f"result. The passages below are still the best matches. Details: {exc}"
        )
        return _retrieval_only(question, hits, route, notice, started)

    citations = verify_citations(data.get("citations"), hits)
    answer_text = providers.normalise_text(str(data.get("answer") or ""))
    answer_text = ensure_inline_tags(answer_text, citations, hits)
    if not answer_text:
        return _retrieval_only(
            question, hits, route, "The model returned an empty answer.", started
        )

    return Answer(
        question=question,
        answer=answer_text,
        key_points=providers.coerce_str_list(data.get("key_points"), limit=6),
        citations=citations,
        hits=hits,
        route=route,
        confidence=str(data.get("confidence") or "medium").lower(),
        caveats=providers.coerce_str_list(data.get("caveats"), limit=6),
        mode="synthesised",
        provider=call.provider,
        model=call.model,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


# ---------------------------------------------------------------- outcomes

TASK_META = {
    "cjpe": {
        "label": "Court judgment",
        "description": (
            "Court judgment prediction. Find Indian court decisions with facts "
            "similar to the user's matter and show how each was decided."
        ),
        "favorable": "Appeal accepted",
        "unfavorable": "Appeal rejected",
        "language_note": "",
    },
    "bail": {
        "label": "Bail application",
        "description": (
            "Bail prediction. Find Indian bail applications with facts similar to "
            "the user's matter and show whether bail was granted."
        ),
        "favorable": "Bail granted",
        "unfavorable": "Bail denied",
        "language_note": (
            "The bail corpus is written in Hindi, so an English description "
            "retrieves it less reliably than the judgment corpus."
        ),
    },
}


@dataclass
class SimilarCase:
    doc_id: str
    doc_title: str
    outcome: int
    outcome_label: str
    score: float
    snippet: str
    source_note: str


@dataclass
class OutcomeReport:
    case_description: str
    task: str
    task_label: str
    similar_cases: list[SimilarCase]
    favorable_count: int
    unfavorable_count: int
    favorable_label: str
    unfavorable_label: str
    similar_case_summary: str
    key_factors: list[str]
    assessment: str
    confidence: str
    caveats: list[str]
    mode: str
    provider: str = ""
    model: str = ""
    notice: str = ""
    elapsed_ms: int = 0

    @property
    def favorable_pct(self) -> int:
        total = self.favorable_count + self.unfavorable_count
        return int(round(100 * self.favorable_count / total)) if total else 0


def predict_outcome(
    case_description: str,
    *,
    task: str = "cjpe",
    max_cases: int = 10,
    deep: bool = False,
    preferred_provider: str | None = None,
    byo_key: str | None = None,
    byo_account: str | None = None,
) -> OutcomeReport:
    started = time.perf_counter()
    task = task if task in TASK_META else "cjpe"
    meta = TASK_META[task]
    case_description = case_description.strip()

    index = get_index()
    query_vector = embeddings.embed_query(case_description)
    # Over fetch because several chunks usually belong to the same judgment
    scored = index.similar_by_task(query_vector, source_task=task, top_k=max_cases * 6)

    seen: set[str] = set()
    similar: list[SimilarCase] = []
    for hit in scored:
        chunk = hit.chunk
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        if chunk.outcome == 1:
            label = meta["favorable"]
        elif chunk.outcome == 0:
            label = meta["unfavorable"]
        else:
            label = "Not recorded"
        similar.append(
            SimilarCase(
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                outcome=chunk.outcome,
                outcome_label=label,
                score=round(float(hit.score), 4),
                snippet=chunk.text[:700],
                source_note=chunk.source_note,
            )
        )
        if len(similar) >= max_cases:
            break

    favorable = sum(1 for c in similar if c.outcome == 1)
    unfavorable = sum(1 for c in similar if c.outcome == 0)

    def build(
        *,
        summary: str,
        factors: list[str],
        assessment: str,
        confidence: str,
        caveats: list[str],
        mode: str,
        provider: str = "",
        model: str = "",
        notice: str = "",
    ) -> OutcomeReport:
        return OutcomeReport(
            case_description=case_description,
            task=task,
            task_label=meta["label"],
            similar_cases=similar,
            favorable_count=favorable,
            unfavorable_count=unfavorable,
            favorable_label=meta["favorable"],
            unfavorable_label=meta["unfavorable"],
            similar_case_summary=summary,
            key_factors=factors,
            assessment=assessment,
            confidence=confidence,
            caveats=caveats,
            mode=mode,
            provider=provider,
            model=model,
            notice=notice,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    if not similar:
        return build(
            summary="No comparable cases were found in this dataset.",
            factors=[],
            assessment=(
                "The similarity search returned nothing for this description. "
                "Try describing the facts and the charges in more detail."
            ),
            confidence="low",
            caveats=[],
            mode="retrieval_only",
        )

    total = favorable + unfavorable
    pct = int(round(100 * favorable / total)) if total else 0
    statistical = (
        f"Across the {total} closest comparable matters, {favorable} ended in "
        f"{meta['favorable'].lower()} ({pct} percent) and {unfavorable} ended in "
        f"{meta['unfavorable'].lower()}."
    )

    base_caveats = [
        "This is a similarity search over past cases, not a prediction of your result.",
        "A court weighs the specific facts and the record before it.",
    ]
    if meta["language_note"]:
        base_caveats.append(meta["language_note"])

    if not providers.has_any_provider(byo_key, byo_account):
        return build(
            summary=statistical,
            factors=[],
            assessment=(
                "The comparable cases and their recorded outcomes are listed below. "
                "Connect a provider key in Settings to get a written analysis of what "
                "separates the two groups."
            ),
            confidence="low",
            caveats=base_caveats,
            mode="retrieval_only",
            notice="No language model is connected, so this is a statistical result only.",
        )

    call = providers.ProviderCall()
    prompt = prompts.OUTCOME_PROMPT.format(
        task_description=meta["description"]
        + (f"\n{meta['language_note']}" if meta["language_note"] else ""),
        case_description=case_description,
        sources=prompts.render_outcome_sources(similar),
        favorable_label=meta["favorable"],
        unfavorable_label=meta["unfavorable"],
        n_favorable=favorable,
        n_unfavorable=unfavorable,
        n_total=len(similar),
    )
    try:
        data = providers.chat_json(
            system=prompts.OUTCOME_SYSTEM,
            prompt=prompt,
            schema=prompts.OUTCOME_SCHEMA,
            deep=deep,
            temperature=0.2,
            max_tokens=2400,
            preferred_provider=preferred_provider,
            byo_key=byo_key,
            byo_account=byo_account,
            call=call,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Outcome analysis failed: %s", exc)
        return build(
            summary=statistical,
            factors=[],
            assessment=(
                "The comparable cases and their outcomes are listed below. "
                f"The written analysis could not be generated. Details: {exc}"
            ),
            confidence="low",
            caveats=base_caveats,
            mode="retrieval_only",
        )

    caveats = providers.coerce_str_list(data.get("caveats"), limit=6)
    for caveat in base_caveats:
        if caveat not in caveats:
            caveats.append(caveat)

    return build(
        summary=providers.normalise_text(str(data.get("similar_case_summary") or statistical)),
        factors=providers.coerce_str_list(data.get("key_factors"), limit=6),
        assessment=providers.normalise_text(str(data.get("assessment") or "")),
        confidence=str(data.get("confidence") or "medium").lower(),
        caveats=caveats,
        mode="synthesised",
        provider=call.provider,
        model=call.model,
    )
