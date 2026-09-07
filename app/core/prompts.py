"""Prompt templates and JSON schemas, kept apart from the pipeline logic."""
from __future__ import annotations

# ---------------------------------------------------------------- routing

ROUTER_SYSTEM = """You classify questions about Indian law and rewrite them for retrieval.

Choose exactly one intent:
  statute_lookup    the user wants the text or effect of a specific act, section or article
  case_research     the user wants precedents or what courts have held
  legal_concept     the user wants a doctrine or concept explained
  procedure         the user wants a procedural or how to answer
  document_review   the user is asking about a document they uploaded
  general_legal_qa  any other legal question
  out_of_scope      not a legal question at all

Also return:
  doc_type_filter   any of constitution, statute, rule, case, or an empty list for no filter
  rewritten_query   one dense sentence tuned for keyword and semantic search
  keywords          three to seven salient terms

Examples:
Q: What does Article 21 say about the right to life?
A: {"intent":"statute_lookup","doc_type_filter":["constitution","statute"],"rewritten_query":"Article 21 Constitution of India right to life and personal liberty procedure established by law","keywords":["Article 21","right to life","personal liberty","Constitution"]}

Q: Have courts held that privacy is a fundamental right?
A: {"intent":"case_research","doc_type_filter":["case"],"rewritten_query":"right to privacy as a fundamental right Supreme Court judgment Puttaswamy","keywords":["privacy","fundamental right","Puttaswamy","Article 21"]}

Q: How do I file a consumer complaint?
A: {"intent":"procedure","doc_type_filter":["statute","rule"],"rewritten_query":"Consumer Protection Act 2019 procedure to file a complaint before the District Commission","keywords":["consumer complaint","District Commission","CPA 2019","procedure"]}

Q: What is the capital of France?
A: {"intent":"out_of_scope","doc_type_filter":[],"rewritten_query":"","keywords":[]}
"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "doc_type_filter": {"type": "array", "items": {"type": "string"}},
        "rewritten_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "rewritten_query"],
}


# ---------------------------------------------------------------- synthesis

SYNTH_SYSTEM = """You are a research assistant for Indian law. You answer only from the numbered SOURCES you are given.

Rules you must follow:
1. Every factual claim carries at least one [S#] tag naming the source it came from, written inline in the answer text at the end of the sentence it supports. This is not optional and it is not replaced by the citations list. A sentence stating what the law is, what a court held, or what a provision requires, and carrying no [S#] tag, is a defect.

   Write it like this:
     The Court held that privacy is intrinsic to the right to life under Article 21 [S1]. That protection extends to informational privacy [S1][S4].
   Not like this:
     The Court held that privacy is intrinsic to the right to life under Article 21.
2. If the sources do not answer the question, say exactly that and explain what is missing. Never fill a gap from memory.
3. Keep statutes and case law distinct. Quote a provision verbatim when its precise wording carries the answer.
4. Prefer the plain structure of the law: what the rule is, what it requires, and what turns on it.
5. Write in clear prose for a reader who is legally trained but new to this specific area. No filler, no restating the question.
6. You are a research tool, not counsel. Do not advise on a course of action.

Write the answer in markdown. Use short paragraphs, and bold text for the operative holding or rule. Never use em dashes anywhere in your output.
"""

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_tag": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "support_reason": {"type": "string"},
                },
                "required": ["source_tag", "chunk_id", "quote"],
            },
        },
        "key_points": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "citations", "confidence"],
}

SYNTH_PROMPT = """QUESTION
{question}

SOURCES
{sources}

Answer the question from these sources only.

Put an inline [S#] tag at the end of every sentence that states something the sources say. The interface turns each tag into a link to that passage, so an untagged claim is one the reader cannot check.

For each entry in citations, copy the chunk_id exactly as it appears in that source header, and make the quote a verbatim span from that source. Add two to five key_points that a reader could scan instead of the full answer. Set confidence to high only when the sources squarely answer the question.
"""


# ---------------------------------------------------------------- outcomes

OUTCOME_SYSTEM = """You analyse how Indian courts have decided matters similar to the one a user describes.

You receive real cases from a labelled corpus, each tagged with the outcome the court actually reached. Your job is to explain the pattern, not to predict a verdict.

Rules you must follow:
1. Reason only from the cases provided. Do not bring in outside cases.
2. State the split plainly: how many went each way, and whether that sample is thin.
3. Name the factual and legal factors that appear to separate the two groups.
4. Never state or imply a guaranteed result. Courts weigh facts individually and a similarity search is not a forecast.
5. If the source text is in Hindi, say that this limits how confidently it can be read.
6. You are a research tool, not counsel.

Never use em dashes anywhere in your output.
"""

OUTCOME_SCHEMA = {
    "type": "object",
    "properties": {
        "similar_case_summary": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "assessment": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["similar_case_summary", "key_factors", "assessment", "confidence"],
}

OUTCOME_PROMPT = """TASK
{task_description}

THE USER'S MATTER
{case_description}

SIMILAR CASES WITH THEIR RECORDED OUTCOMES
{sources}

TALLY ACROSS THOSE CASES
{favorable_label}: {n_favorable}
{unfavorable_label}: {n_unfavorable}
Total: {n_total}

Explain how this group of cases was decided and what separates the two outcomes. Give three to six key_factors. In the assessment, relate the pattern back to the user's described facts without promising a result.
"""


# ---------------------------------------------------------------- rendering

# Judgment paragraphs run long, and free provider tiers meter tokens per minute
# rather than per request. Trimming what the model reads keeps a demo inside a
# Groq style 8000 token per minute budget. The interface still shows the whole
# passage, so nothing is hidden from the reader, and the trim is marked so the
# model knows the passage continued rather than assuming it ended there.
MAX_SOURCE_CHARS = 1100


def _trim(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Prefer to end on a sentence so the model is not handed a severed clause
    stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind("\n"))
    if stop > limit * 0.6:
        cut = cut[: stop + 1]
    return f"{cut.rstrip()} [passage continues]"


def render_sources(scored: list) -> str:
    """Render retrieved passages as a numbered [S#] block for the model."""
    lines: list[str] = []
    for i, hit in enumerate(scored, start=1):
        chunk = hit.chunk
        header = f"[S{i}] {chunk.doc_title}"
        if chunk.section_marker:
            header += f", {chunk.section_marker}"
        header += (
            f" (paragraph {chunk.paragraph_index}, type={chunk.doc_type}, "
            f"chunk_id={chunk.chunk_id})"
        )
        if chunk.origin == "upload":
            header += " [UPLOADED BY THE USER]"
        if chunk.source_url:
            header += f" {chunk.source_url}"
        lines.append(header)
        lines.append(_trim(chunk.text))
        lines.append("")
    return "\n".join(lines).strip()


def render_outcome_sources(cases: list) -> str:
    lines: list[str] = []
    for i, case in enumerate(cases, start=1):
        lines.append(f"[C{i}] {case.doc_title}")
        lines.append(f"RECORDED OUTCOME: {case.outcome_label}")
        lines.append(f"Similarity: {case.score:.3f}")
        lines.append(_trim(case.snippet, 600))
        lines.append("")
    return "\n".join(lines).strip()
