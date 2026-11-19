"""Ticket workflow rules, kept out of the HTTP layer so they are easy to test.

    create ticket ──> open ──(generate draft)──> drafted ──(approve / edit)──> resolved
                        ^                           │
                        └──────(reject draft)───────┘

Routes translate the errors below into HTTP codes (404 / 409); this module
never imports FastAPI.
"""
from dataclasses import asdict
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import Draft, Ticket
from app.llm.base import LLMClient
from app.rag.retriever import Retriever
from app.services.answer_service import generate_draft
from app.services.edit_distance import edit_ratio, strip_citation_markers
from app.services.pii import mask_pii


class NotFoundError(Exception):
    """The ticket or draft id does not exist (-> HTTP 404)."""


class ConflictError(Exception):
    """The action is not allowed in the current state (-> HTTP 409)."""


def merge_counts(*counts: dict[str, int]) -> dict[str, int]:
    total: dict[str, int] = {}
    for c in counts:
        for kind, n in c.items():
            total[kind] = total.get(kind, 0) + n
    return total


def create_ticket(db: Session, subject: str, body: str, customer_email: str | None) -> Ticket:
    subject_mask, body_mask = mask_pii(subject), mask_pii(body)
    ticket = Ticket(subject=subject, body=body,
                    subject_masked=subject_mask.text, body_masked=body_mask.text,
                    pii_found=merge_counts(subject_mask.found, body_mask.found),
                    customer_email=customer_email)
    db.add(ticket)
    db.commit()  # INSERT happens here; ticket.id is filled in afterwards
    return ticket


def get_ticket(db: Session, ticket_id: int) -> Ticket:
    ticket = db.get(Ticket, ticket_id)  # lookup by primary key
    if ticket is None:
        raise NotFoundError(f"Ticket {ticket_id} not found.")
    return ticket


def create_draft_for_ticket(db: Session, ticket: Ticket, retriever: Retriever, llm: LLMClient,
                            top_k: int = 5, max_retries: int = 1) -> Draft:
    if ticket.status == "resolved":
        raise ConflictError(f"Ticket {ticket.id} is already resolved.")

    # The original text goes in; generate_draft masks it before retrieval and the LLM.
    result = generate_draft(f"{ticket.subject}\n\n{ticket.body}", retriever, llm,
                            top_k=top_k, max_retries=max_retries)
    llm_info = result.llm or {}
    draft = Draft(
        ticket_id=ticket.id,
        status=result.status,
        answer=result.answer,
        citations=[asdict(c) for c in result.citations],
        warnings=result.warnings,
        sources_sent=result.sources_sent,
        pii_masked=result.pii_masked,
        evidence_sufficient=result.retrieval.evidence_sufficient,
        best_similarity=result.retrieval.best_similarity,
        llm_called=result.llm_called,
        llm_provider=llm_info.get("provider"),
        llm_model=llm_info.get("model"),
        input_tokens=llm_info.get("input_tokens"),
        output_tokens=llm_info.get("output_tokens"),
        latency_ms=llm_info.get("latency_ms"),
    )
    ticket.status = "drafted"
    db.add(draft)
    db.commit()  # draft INSERT + ticket UPDATE in ONE transaction: both happen or neither
    return draft


def get_draft(db: Session, draft_id: int) -> Draft:
    draft = db.get(Draft, draft_id)
    if draft is None:
        raise NotFoundError(f"Draft {draft_id} not found.")
    return draft


def review_draft(db: Session, draft: Draft, decision: str,
                 final_reply: str | None = None, note: str | None = None) -> Draft:
    """Record the agent's decision. Each draft can be reviewed exactly once."""
    if draft.review_decision is not None:
        raise ConflictError(f"Draft {draft.id} was already reviewed ({draft.review_decision}).")
    latest_id = max(d.id for d in draft.ticket.drafts)
    if draft.id != latest_id:
        raise ConflictError(f"Draft {draft.id} is outdated; review the latest draft ({latest_id}).")

    if decision == "approved":
        if not draft.answer:
            raise ConflictError("This draft has no text to approve (generation failed). Edit or reject it.")
        draft.final_reply = strip_citation_markers(draft.answer)
        draft.edit_ratio = 0.0
        draft.ticket.status = "resolved"
    elif decision == "edited":
        draft.final_reply = final_reply
        draft.edit_ratio = edit_ratio(draft.answer, final_reply or "")
        draft.ticket.status = "resolved"
    else:  # rejected: the draft was not usable; the ticket needs another attempt
        draft.review_note = note
        draft.ticket.status = "open"

    draft.review_decision = decision
    draft.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    return draft
