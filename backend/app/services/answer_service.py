"""The online RAG pipeline for one ticket: retrieve -> (maybe) generate -> validate.

Possible outcomes (DraftResult.status):
  "grounded"              all checks passed; still reviewed by an agent
  "needs_attention"       a draft exists but a check raised a warning
  "insufficient_evidence" retrieval or the model found the docs don't cover it;
                          a fixed hand-off reply is used, never a made-up answer
  "generation_failed"     the LLM could not be reached or never returned valid JSON
"""
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.llm.base import LLMClient, LLMError
from app.llm.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, LLMAnswer, build_user_prompt
from app.rag.retriever import RetrievalResult, RetrievedChunk, Retriever
from app.services.citations import validate_citations
from app.services.pii import mask_pii

HANDOFF_REPLY = (
    "Thank you for contacting StreamNest support. We couldn't find documentation that fully "
    "answers your question, so a support specialist will review your request and get back to you."
)


@dataclass
class Citation:
    source_id: str
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    similarity: float
    outdated: bool


@dataclass
class DraftResult:
    status: str
    answer: str
    citations: list[Citation]
    warnings: list[str]
    retrieval: RetrievalResult
    sources_sent: list[str] = field(default_factory=list)  # chunk_ids shown to the LLM
    llm_called: bool = False
    llm: dict | None = None  # provider, model, tokens, latency, attempts
    pii_masked: dict[str, int] = field(default_factory=dict)  # what was hidden from retrieval + LLM


def select_sources(result: RetrievalResult, top_k: int) -> tuple[list[RetrievedChunk], list[str]]:
    """Pick the top_k chunks to send to the LLM.

    1. Drop outdated chunks when the document that replaced them was also
       retrieved. Doing this in code (not only via a prompt rule) means a small
       model can't accidentally quote the old refund window.
    2. THEN cut to top_k, so current chunks fill the freed slots.
    3. Warn about an outdated document only if it is actually being sent.
    """
    notes, drop = [], set()
    for conflict in result.conflicts:
        if conflict["current_doc_retrieved"]:
            drop.add(conflict["outdated_doc_id"])
            notes.append(f"Excluded outdated document {conflict['outdated_doc_id']} "
                         f"(replaced by {conflict['current_doc_id']}).")
    kept = [rc for rc in result.chunks if rc.chunk.doc_id not in drop][:top_k]

    sent_docs = {rc.chunk.doc_id for rc in kept}
    for conflict in result.conflicts:
        if not conflict["current_doc_retrieved"] and conflict["outdated_doc_id"] in sent_docs:
            notes.append(f"Only the outdated document {conflict['outdated_doc_id']} was retrieved; "
                         f"its replacement {conflict['current_doc_id']} was not. Check the policy manually.")
    return kept, notes


def generate_draft(ticket_text: str, retriever: Retriever, llm: LLMClient,
                   top_k: int = 5, max_retries: int = 1) -> DraftResult:
    # Mask PII first: this function is the single path to the LLM, so masking
    # here protects every caller (the /answer endpoint and stored tickets).
    masked = mask_pii(ticket_text)
    ticket_text = masked.text

    # Stages 6-7: retrieval + evidence check (no LLM involved).
    # Fetch 2x the chunks we need: if outdated documents fill some slots and are
    # dropped below, current chunks take their place instead of the prompt
    # silently shrinking (a real bug seen with the 2024 vs 2026 refund policy).
    retrieval = retriever.retrieve(ticket_text, top_k=top_k * 2)
    if not retrieval.evidence_sufficient:
        # Weak evidence: don't even ask the LLM. Saves time and removes the
        # chance of a confident answer built on irrelevant passages.
        return DraftResult("insufficient_evidence", HANDOFF_REPLY, [], [retrieval.reason], retrieval,
                           pii_masked=masked.found)

    chunks, warnings = select_sources(retrieval, top_k)
    sources ={f"S{i}": rc for i, rc in enumerate(chunks, start=1)}
    user_prompt = build_user_prompt(ticket_text, sources)                       # stage 8

    # Stage 9: call the LLM. Retry only if the reply is not valid JSON for our
    # schema; connection errors and timeouts won't fix themselves instantly.
    parsed, result, attempts, error = None, None, 0, None
    while attempts <= max_retries:
        attempts += 1
        try:
            result = llm.generate(SYSTEM_PROMPT, user_prompt, ANSWER_SCHEMA)
            parsed = LLMAnswer.model_validate_json(result.content)
            break
        except ValidationError:
            error = "The model's reply was not valid JSON for the answer schema."
        except LLMError as exc:
            error = str(exc)
            break

    llm_info = {"provider": llm.provider, "model": llm.model, "attempts": attempts}
    if result is not None:
        llm_info.update(model=result.model, input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens, latency_ms=result.latency_ms)
    sent = [rc.chunk.chunk_id for rc in chunks]

    if parsed is None:
        return DraftResult("generation_failed", "", [], warnings + [error], retrieval, sent, True,
                           llm_info, masked.found)

    if not parsed.evidence_sufficient:
        # The model, after reading the sources, says they don't answer the ticket.
        note = parsed.missing_information or "no reason given"
        return DraftResult("insufficient_evidence", HANDOFF_REPLY, [],
                           warnings + [f"Model reported missing information: {note}"],
                           retrieval, sent, True, llm_info, masked.found)

    # Stage 10: validate citations against what we actually sent.
    check = validate_citations(parsed.answer, parsed.cited_source_ids, set(sources))
    citations = [
        Citation(source_id=sid, chunk_id=sources[sid].chunk.chunk_id, doc_id=sources[sid].chunk.doc_id,
                 title=sources[sid].chunk.title, section=sources[sid].chunk.section,
                 text=sources[sid].chunk.text, similarity=sources[sid].similarity,
                 outdated=bool(sources[sid].chunk.superseded_by))
        for sid in check.valid_ids
    ]
    warnings += check.warnings
    if any(c.outdated for c in citations):
        warnings.append("The answer cites an outdated policy document.")

    # Notes about excluded outdated docs are informational, not problems.
    problems = [w for w in warnings if not w.startswith("Excluded outdated")]
    status = "needs_attention" if problems else "grounded"
    return DraftResult(status, check.cleaned_answer, citations, warnings, retrieval, sent, True,
                       llm_info, masked.found)
