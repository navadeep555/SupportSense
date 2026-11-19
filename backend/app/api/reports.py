"""Reports built from stored drafts and agent reviews.

GET /api/reports/draft-metrics   how useful are the AI drafts? (feature 3)
GET /api/reports/knowledge-gaps  which questions does the documentation not cover? (feature 2)

Both read only masked ticket text, so personal data never appears in reports.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.models import Draft, Ticket
from app.schemas import DraftMetrics, KnowledgeGapReport

router = APIRouter(prefix="/reports", tags=["reports"])


def _counts(db: Session, column) -> dict[str, int]:
    """SELECT column, COUNT(*) FROM drafts GROUP BY column  ->  {value: count}"""
    rows = db.execute(select(column, func.count()).where(column.is_not(None)).group_by(column))
    return {value: n for value, n in rows}


@router.get("/draft-metrics", response_model=DraftMetrics)
def draft_metrics(db: Session = Depends(get_db)) -> dict:
    by_status = _counts(db, Draft.status)
    by_decision = _counts(db, Draft.review_decision)
    reviewed = sum(by_decision.values())
    approved, edited = by_decision.get("approved", 0), by_decision.get("edited", 0)

    avg_edit = db.scalar(select(func.avg(Draft.edit_ratio)).where(Draft.review_decision == "edited"))
    avg_latency = db.scalar(select(func.avg(Draft.latency_ms)).where(Draft.latency_ms.is_not(None)))
    return {
        "total_drafts": sum(by_status.values()),
        "by_status": by_status,
        "reviewed": reviewed,
        "by_decision": by_decision,
        # None (not 0) when nothing is reviewed yet: "no data" is different from "0%".
        "approval_rate": round(approved / reviewed, 3) if reviewed else None,
        "acceptance_rate": round((approved + edited) / reviewed, 3) if reviewed else None,
        "avg_edit_ratio_when_edited": round(avg_edit, 3) if avg_edit is not None else None,
        "avg_llm_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
    }


@router.get("/knowledge-gaps", response_model=KnowledgeGapReport)
def knowledge_gaps(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)) -> dict:
    """Drafts where the knowledge base failed: no evidence found, or an agent rejected the draft.

    Lowest similarity first: those are the topics most clearly missing from the docs.
    """
    gap_filter = or_(Draft.status == "insufficient_evidence", Draft.review_decision == "rejected")
    total = db.scalar(select(func.count()).select_from(Draft).where(gap_filter))
    rows = db.execute(
        select(Draft, Ticket).join(Ticket, Draft.ticket_id == Ticket.id)
        .where(gap_filter).order_by(Draft.best_similarity.asc(), Draft.id.desc()).limit(limit)
    ).all()

    items = []
    for draft, ticket in rows:
        if draft.review_decision == "rejected":
            reason, detail = "agent_rejected", draft.review_note or ""
        elif not draft.llm_called:
            reason, detail = "no_matching_docs", draft.warnings[0] if draft.warnings else ""
        else:
            reason, detail = "model_found_docs_insufficient", draft.warnings[-1] if draft.warnings else ""
        items.append({"ticket_id": ticket.id, "draft_id": draft.id, "reason": reason, "detail": detail,
                      "best_similarity": draft.best_similarity, "subject_masked": ticket.subject_masked,
                      "body_masked": ticket.body_masked, "created_at": draft.created_at})
    return {"total": total, "items": items}
