"""Table definitions (ORM models). One class = one table, one attribute = one column.

Ticket 1 ---- * Draft
A ticket can have several drafts (e.g. the agent rejects one and asks for
another). Each draft stores what the AI produced AND what the agent did with
it; the review fields feed the draft-quality metrics and knowledge-gap report.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)  # auto-increments: 1, 2, 3...
    subject: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    # Masked copies: what the LLM and the reports see. The originals stay for the
    # agent, who may need e.g. the real phone number to call the customer back.
    subject_masked: Mapped[str] = mapped_column(String(200))
    body_masked: Mapped[str] = mapped_column(Text)
    pii_found: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"email": 1, "phone": 2}
    customer_email: Mapped[str | None] = mapped_column(String(254))
    # Lifecycle: open -> drafted -> resolved (rejecting a draft sends it back to open)
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)
    # ---- Triage (Phase 6), computed once from the masked text at creation ----
    category: Mapped[str | None] = mapped_column(String(50), index=True)  # None if no classifier
    category_confidence: Mapped[float | None]
    category_low_confidence: Mapped[bool] = mapped_column(default=False)
    escalated: Mapped[bool] = mapped_column(default=False, index=True)
    escalation_score: Mapped[float | None]
    escalation_reasons: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)

    # Not a column: lets Python code use ticket.drafts. cascade = deleting a
    # ticket also deletes its drafts.
    drafts: Mapped[list["Draft"]] = relationship(
        back_populates="ticket", order_by="Draft.id", cascade="all, delete-orphan")


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)

    # ---- What the AI pipeline produced (copied from DraftResult) ----
    status: Mapped[str] = mapped_column(String(30), index=True)  # grounded / needs_attention / ...
    answer: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    sources_sent: Mapped[list] = mapped_column(JSON, default=list)
    pii_masked: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_sufficient: Mapped[bool]
    best_similarity: Mapped[float]
    llm_called: Mapped[bool]
    llm_provider: Mapped[str | None] = mapped_column(String(30))
    llm_model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    latency_ms: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # ---- What the human agent did (all empty until the draft is reviewed) ----
    review_decision: Mapped[str | None] = mapped_column(String(20), index=True)  # approved/edited/rejected
    final_reply: Mapped[str | None] = mapped_column(Text)  # the text actually sent to the customer
    edit_ratio: Mapped[float | None]  # 0.0 = sent unchanged ... 1.0 = completely rewritten
    review_note: Mapped[str | None] = mapped_column(Text)  # why it was rejected
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    ticket: Mapped[Ticket] = relationship(back_populates="drafts")
