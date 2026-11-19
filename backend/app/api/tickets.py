"""Ticket REST API: create/list/read tickets, generate AI drafts, record agent reviews.

    POST /api/tickets                    create a ticket (PII is masked on save)
    GET  /api/tickets?status=open        list tickets (escalated first, then newest), paginated;
                                         filters: status, category, escalated
    GET  /api/tickets/{id}               one ticket with all its drafts
    POST /api/tickets/{id}/drafts        run the RAG pipeline and store the draft
    POST /api/drafts/{id}/review         agent approves / edits / rejects a draft
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.answers import get_llm
from app.api.deps import get_db, get_triager
from app.api.retrieval import get_retriever
from app.config import get_settings
from app.db.models import Ticket
from app.llm.base import LLMClient
from app.ml.classifier import CATEGORIES
from app.rag.retriever import Retriever
from app.schemas import (DraftOut, DraftRequest, ReviewRequest, TicketCreate, TicketDetail,
                         TicketList, TicketOut)
from app.services import ticket_service as svc
from app.services.triage import Triager

router = APIRouter(tags=["tickets"])


def http_error(exc: Exception) -> HTTPException:
    """Map service-layer errors to HTTP status codes in one place."""
    code = status.HTTP_404_NOT_FOUND if isinstance(exc, svc.NotFoundError) else status.HTTP_409_CONFLICT
    return HTTPException(code, str(exc))


@router.post("/tickets", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def create_ticket(body: TicketCreate, db: Session = Depends(get_db),
                  triager: Triager = Depends(get_triager)) -> Ticket:
    return svc.create_ticket(db, body.subject, body.body, body.customer_email, triager)


@router.get("/tickets", response_model=TicketList)
def list_tickets(status_filter: str | None = Query(default=None, alias="status",
                                                   pattern="^(open|drafted|resolved)$"),
                 category: str | None = Query(default=None, pattern=f"^({'|'.join(CATEGORIES)})$"),
                 escalated: bool | None = None,
                 limit: int = Query(default=20, ge=1, le=100),
                 offset: int = Query(default=0, ge=0),
                 db: Session = Depends(get_db)) -> dict:
    query = select(Ticket)
    if status_filter:
        query = query.where(Ticket.status == status_filter)
    if category:
        query = query.where(Ticket.category == category)
    if escalated is not None:
        query = query.where(Ticket.escalated == escalated)
    # Count with the same filters, so the UI can show "page 2 of 5".
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    # Work queue order: escalated tickets first, then newest first.
    order = (Ticket.escalated.desc(), Ticket.id.desc())
    items = db.scalars(query.order_by(*order).limit(limit).offset(offset)).all()
    return {"total": total, "items": items}


@router.get("/tickets/{ticket_id}", response_model=TicketDetail)
def read_ticket(ticket_id: int, db: Session = Depends(get_db)) -> Ticket:
    try:
        return svc.get_ticket(db, ticket_id)
    except svc.NotFoundError as exc:
        raise http_error(exc)


@router.post("/tickets/{ticket_id}/drafts", response_model=DraftOut, status_code=status.HTTP_201_CREATED)
def create_draft(ticket_id: int, body: DraftRequest | None = None, db: Session = Depends(get_db),
                 retriever: Retriever = Depends(get_retriever), llm: LLMClient = Depends(get_llm)):
    top_k = body.top_k if body else 5
    try:
        ticket = svc.get_ticket(db, ticket_id)
        return svc.create_draft_for_ticket(db, ticket, retriever, llm, top_k=top_k,
                                           max_retries=get_settings().llm_max_retries)
    except (svc.NotFoundError, svc.ConflictError) as exc:
        raise http_error(exc)


@router.post("/drafts/{draft_id}/review", response_model=DraftOut)
def review(draft_id: int, body: ReviewRequest, db: Session = Depends(get_db)):
    try:
        draft = svc.get_draft(db, draft_id)
        return svc.review_draft(db, draft, body.decision, body.final_reply, body.note)
    except (svc.NotFoundError, svc.ConflictError) as exc:
        raise http_error(exc)
